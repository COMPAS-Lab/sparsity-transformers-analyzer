import numpy as np
import random
from scipy import optimize
from scipy.stats import pearsonr
import hw_modeling
from roofline import Roofline
from sparsemat_hw_modeling import (
    distance_of_dense_vals_per_row, 
    PerfData, 
    transfer_attn_to_bprune_dense_idx,
    bprune_sweep_rrspan,
    rr2spmm_latency_overlap_analysis,
    spmm_non_rr_latency_analysis,
    get_tccore_config,
    rr2spmm_fifo_latency_overlap_analysis, 
    sigma_latency_analysis,
    naive_roundrobin_latency_analysis)
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import os, json, util, pathlib, math
from itertools import product, chain
import multiprocessing
import pandas as pd
import seaborn as sns
import heapq
from util import find_positive_integer_pairs

MODEL_PALETTE = {
    "chatglm2-6b-32k": ("#1f77b4", "#84cdff"), 
    "llama2-7b-chat-4k": ("#ff7f0e", "#ffc85f"), 
    "mixtral-8x7b": ("#2ca02c", "#76ce55")
}

NX10_RESOURCES = {"tensor block": 3960.0, "m20k": 6840.0, "alm": 702720.0}

def sublist_creator(lst, n):
    lists = [[] for _ in range(n)]
    totals = [(0, i) for i in range(n)]
    heapq.heapify(totals)
    for value in lst:
        total, index = heapq.heappop(totals)
        lists[index].append(value)
        heapq.heappush(totals, (total + value, index))
    return lists

def suboptimal_sublist_creator(lst, n, return_indices=False) -> tuple:
    """
    naive load balancing algorithm to create 2 sublists
    """
    assert n == 2, "Only supports 2 sublists for now"
    lists, sched_indices = [[], []],  [[], []]
    lists[0].append(lst[0])
    sched_indices[0].append(0)

    for idx, value in enumerate(lst[1:]):
        if sum(lists[0]) <= sum(lists[1]):
            lists[0].append(value)
            sched_indices[0].append(idx + 1)
        else:
            lists[1].append(value)
            sched_indices[1].append(idx + 1)

    if return_indices:
        return lists, sched_indices
    else:
        return lists, None

def gen_spmat_by_sparsity(ref_mat: np.array, 
                          target_seqlen: int, 
                          fake_dense_val = 0.4, 
                          is_plot_figure = False) -> np.array:
    row_spar = np.count_nonzero(ref_mat, axis=-1)
    row_dist, _ = distance_of_dense_vals_per_row(ref_mat, row_size=1)
    row_first_dval = np.argmax(ref_mat > 0, axis=-1)
    ref_mat_len = int(ref_mat.shape[-1])

    #scale each row distribution attribute to target len:
    def get_curve_fit_res(y, cur, appro_fn=np.ceil) -> list:
        coeff, _ = optimize.curve_fit(cur, list(range(ref_mat_len)), y)
        thres = int(50 / 1024 * 2048)
        coeff = list(coeff).append(thres)
        # alternate b
        coeff[1] += 50 - thres
        coeff[2] = int(coeff[2] / 1024 * 2048)
        print(f"coeff: {coeff}")
        new_x = np.arange(0, target_seqlen, 1).astype(np.int)
        pred_y = appro_fn(cur(new_x, *coeff)).astype(np.int)
        return pred_y
    
    def row_abs_func(x, a, b, c, thres=50):
        return np.piecewise(x, [x < thres, x >= thres], [lambda x: a * x + b, c])

    def row_first_func(x, a, b, c, thres=50):
        y = np.piecewise(x, [x < thres, x >= thres], [c, lambda x: a * x + b])
        return y

    scaled_row_dist = get_curve_fit_res(row_dist, row_abs_func, np.floor)
    scaled_row_first_dval = get_curve_fit_res(row_first_dval, row_first_func, np.floor)
    num_dvals_row = get_curve_fit_res(row_spar, row_abs_func, np.floor)
    
    if is_plot_figure:
        plt.plot(row_dist, color="r", label="attn span")
        plt.plot(row_first_dval, color="g", label="first dval")
        plt.plot(row_spar, color="b", label="#dval")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.legend()
        plt.savefig("./res_fig/temp/row_spar_feature.pdf")
        plt.clf()
    
        plt.plot(scaled_row_dist, color="r", label="attn span")
        plt.plot(scaled_row_first_dval, color="g", label="first dval")
        plt.plot(num_dvals_row, color="b", label="#dval")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.legend()
        plt.savefig("./res_fig/temp/row_spar_feature_scaled.pdf")
        plt.clf()

    res_mat = np.zeros((target_seqlen, target_seqlen))
    for r in range(int(target_seqlen)):
        scaled_row_last_dval = min(scaled_row_dist[r] + scaled_row_first_dval[r], target_seqlen)
        cand_cols = np.arange(
            scaled_row_first_dval[r], scaled_row_last_dval, 1).astype(np.int)
        selected_num_cols = min(len(cand_cols), num_dvals_row[r])
        # if len(cand_cols) < num_dvals_row[r]:
        #     print("#candidates not enough!")
        dval_cols = random.sample(list(cand_cols), selected_num_cols)
        for c in dval_cols:
            res_mat[r][c] = fake_dense_val

    # confirm sparsity
    ori_spar = np.count_nonzero(ref_mat) / ref_mat.size
    new_spar = np.count_nonzero(res_mat) / res_mat.size
    print(f"ori: {ori_spar:.3f}, new: {new_spar:.3f}")
    return res_mat


def explore_row_features(mats: list, inst_idx: int):
    row_spar_list, row_dist_list, row_first_dval_list = [], [], []
    mats = mats[:4,:4,:,:]
    for ref_mat_l in mats:
        print(f"mat shape: {ref_mat_l.shape}")
        for ref_mat_h in ref_mat_l:
            row_spar = np.count_nonzero(ref_mat_h, axis=-1)
            row_dist, _ = distance_of_dense_vals_per_row(ref_mat_h, row_size=1)
            row_first_dval = np.argmax(ref_mat_h > 0, axis=-1)

            row_spar_list.append(row_spar)
            row_dist_list.append(row_dist)
            row_first_dval_list.append(row_first_dval)

    for i in row_spar_list:
        plt.plot(i, color="b", linewidth=1, alpha=0.4)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=mats.shape[-1])
    plt.savefig(f"./res_fig/temp/row_num_dval_i{inst_idx}l.png")
    plt.clf()

    for i in row_dist_list:
        plt.plot(i, color="r", linewidth=1, alpha=0.4)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=mats.shape[-1])
    plt.savefig(f"./res_fig/temp/row_span_i{inst_idx}l.png")
    plt.clf()

    all_row_dists = np.array(row_dist_list).flatten()
    plt.hist(all_row_dists, bins=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.savefig(f"./res_fig/temp/row_span_dist_i{inst_idx}.png")
    plt.clf()

    for i in row_first_dval_list:
        plt.plot(i, color="g", linewidth=1, alpha=0.4)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=mats.shape[-1])
    plt.savefig(f"./res_fig/temp/row_first_dval_i{inst_idx}l.png")
    plt.clf()


def get_unique_colidx_ratio(idx_list: list[tuple[int, int]], n_shared_chans: int, omit_last_iter=False):
    '''
    get unique colidx ratio of a head
    '''
    res_redidx_count_list, res_total_idx_count_list = [], []
    res_effec_area = []
    res_inner_area_util_list, res_total_area_util_list = [], []
    res_n_route_ratio = []

    def get_unique_idx_rate(curr_blk_list: list) -> tuple:
        # redundant index count computing method
        all_col_idx = list(chain(*curr_blk_list))
        total_idx_count = len(all_col_idx)
        sorted_idx_res = list(np.unique(all_col_idx))
        n_removed_redidx = total_idx_count - len(sorted_idx_res)
        n_unique_idx = len(sorted_idx_res)

        return n_removed_redidx, total_idx_count, n_unique_idx * n_shared_chans
    
    def get_agg_util_area(curr_blk_list: list, seq_len: int) -> list[tuple]:
        # compute aggregated area / total area
        all_col_idx = list(chain(*curr_blk_list))
        unique_col_idx_res = list(np.unique(all_col_idx))
        total_area_util = float(len(unique_col_idx_res)) / float(seq_len)
        # compute inner area /aggregated area
        inner_area_util = float(len(all_col_idx)) / float((max(unique_col_idx_res)+1) * n_shared_chans)
        return inner_area_util, total_area_util

    def get_n_route_ratio(curr_blk_list: list, cl: int) -> float:
        # compute number of max iterations, pad each row to the same length
        n_iters = max([math.ceil(len(r) / cl) for r in curr_blk_list])
        padded_blk_list = []
        for r in curr_blk_list:
            padded = []
            if len(r) < (n_iters * cl):
                padded = r + [-1] * (n_iters * cl - len(r))
            else:
                padded = r
            padded_blk_list.append(padded)

        # compute shared n routes
        all_col_idx = list(chain(*curr_blk_list))
        n_unique_col_idx = len(list(np.unique(all_col_idx)))
        shared_routes = float(cl) if n_unique_col_idx > cl else float(n_unique_col_idx)

        # calculate number of routing paths for each iter
        n_routes_ratio = []
        for i in range(n_iters):
            # get current iter's input
            iter_idx_grp = [r[i * cl : (i+1) * cl] for r in padded_blk_list]
            n_unique_access = 0
            for col_idx in range(cl):
                # check how many unique access in each iteration
                uniq_colidx = list(np.sort(np.unique([r[col_idx] for r in iter_idx_grp])))
                n_unique_access += len(uniq_colidx) - 1 if uniq_colidx[0] == -1 else len(uniq_colidx)
            
            n_routes_ratio.append(shared_routes / float(n_unique_access))
            
        return np.mean(n_routes_ratio)

    last_ridx = idx_list[0][0] if idx_list else None
    curr_seq_len = last_ridx
    curr_idx_blk = [[] for i in range(n_shared_chans)]
    curr_rowblk_counter = 0
    # bfp conversion and idx attaching
    for rec_idx in range(len(idx_list)):
        curr_ridx = idx_list[rec_idx][0]
        curr_seq_len = (last_ridx+1) * 3
        if curr_ridx > last_ridx:
            last_ridx = curr_ridx
            curr_rowblk_counter += 1

        if curr_rowblk_counter == n_shared_chans:
            curr_redidx_count, curr_total_idx_count, curr_effective_area = get_unique_idx_rate(curr_idx_blk)
            curr_inner_area_util, curr_total_area_util = get_agg_util_area(curr_idx_blk, math.ceil(curr_seq_len / 20.0))
            curr_n_route_ratio = get_n_route_ratio(curr_idx_blk, 8)

            res_redidx_count_list.append(curr_redidx_count)
            res_total_idx_count_list.append(curr_total_idx_count)
            res_effec_area.append(curr_effective_area)
            res_inner_area_util_list.append(curr_inner_area_util)
            res_total_area_util_list.append(curr_total_area_util)
            res_n_route_ratio.append(curr_n_route_ratio)
            curr_idx_blk = [[] for i in range(n_shared_chans)]
            curr_rowblk_counter = 0

        curr_idx_blk[curr_rowblk_counter].append(idx_list[rec_idx][1])

        # tail
        if (rec_idx == len(idx_list)-1) and (not omit_last_iter):
            curr_redidx_count, curr_total_idx_count, curr_effective_area = get_unique_idx_rate(curr_idx_blk)
            curr_inner_area_util, curr_total_area_util = get_agg_util_area(curr_idx_blk, curr_seq_len+1)
            curr_n_route_ratio = get_n_route_ratio(curr_idx_blk, 8)

            res_redidx_count_list.append(curr_redidx_count)
            res_total_idx_count_list.append(curr_total_idx_count)
            res_effec_area.append(curr_effective_area)
            res_inner_area_util_list.append(curr_inner_area_util)
            res_total_area_util_list.append(curr_total_area_util)
            res_n_route_ratio.append(curr_n_route_ratio)
            curr_idx_blk = [[] for i in range(n_shared_chans)]
            curr_rowblk_counter = 0

    # print(f"total area use: {['{0:.2f}'.format(i) for i in res_total_area_util_list]}")
    # print(f"inner area use: {['{0:.2f}'.format(i) for i in res_inner_area_util_list]}")
    return {
        "head_common_ratio_list": res_redidx_count_list, 
        "head_total_idx_count_list": res_total_idx_count_list, 
        "head_effec_area": res_effec_area, 
        "head_inner_area_util": res_inner_area_util_list, 
        "head_total_area_util": res_total_area_util_list,
        "head_mean_route_ratio": res_n_route_ratio,
    }

def get_unique_colidx_ratio_inst(
        inst_ridx_list: list[str], 
        inst_cidx_list: list[str],
        n_shared_chans: int, 
        aggre_method = None,
        selected_heads: dict[str, list[int]] = None):
    
    all_inst_res_redidx_count, all_inst_res_total_idx_count = [], []
    all_inst_res_iarea_util, all_inst_res_tarea_util = [], []
    all_inst_raw_density, all_inst_effec_density = [], []
    all_inst_mean_route_ratio = []
    all_inst_hidx, all_inst_seq_ids = [], []

    inst_ridx_list.sort()
    inst_cidx_list.sort()
    for rinst, cinst in zip(inst_ridx_list, inst_cidx_list):
        print(f"working on {rinst} and {cinst}")
        inst = rinst.split("/")[-1].split("_ridx.npy")[0]
        # load seq len for current inst
        # read seq len from inst profile
        profile_path = pathlib.Path(rinst).parent / f"{inst}.json"
        curr_seqlen = 0
        with open(profile_path, "r") as profile_f:
            curr_profile = json.load(profile_f)
            curr_seqlen = curr_profile["seq_len"]

        # load and extract all heads data
        src_ridx = np.load(rinst)
        src_cidx = np.load(cinst)
        # extract one head
        headgrp_idx = []
        curr_head_idx = []
        for ridx, cidx in zip(src_ridx, src_cidx):
            if ridx == -1 and cidx == -1:
                headgrp_idx.append(curr_head_idx.copy())
                curr_head_idx = []
            else:
                curr_head_idx.append((ridx, cidx))

        # fetch a head
        print(f"get {len(headgrp_idx)} heads in total")
        curr_inst_res_redidx_count, curr_inst_res_total_idx_count = [], []
        curr_inst_iarea_util, curr_inst_tarea_util = [], []
        curr_inst_raw_density, curr_effec_density = [], []
        curr_inst_mean_route_ratio = []
        curr_hidx, curr_seq_id = [], []
        for h_idx in range(len(headgrp_idx)):
            curr_hidx_selected = selected_heads is None or h_idx in selected_heads[inst]
            if curr_hidx_selected:
                print(f"get head {h_idx} in {inst}")
                total_nblks = math.ceil(math.ceil(curr_seqlen / 3.) * math.ceil(curr_seqlen / 20.0) / 2.0)
                curr_res = get_unique_colidx_ratio(headgrp_idx[h_idx], n_shared_chans, omit_last_iter=True)
                curr_inst_res_redidx_count += curr_res["head_common_ratio_list"]
                curr_inst_res_total_idx_count += curr_res["head_total_idx_count_list"]
                curr_inst_iarea_util += curr_res["head_inner_area_util"]
                curr_inst_tarea_util += curr_res["head_total_area_util"]
                curr_inst_mean_route_ratio += curr_res["head_mean_route_ratio"]
                curr_hidx += [h_idx] * len(curr_res["head_total_area_util"])
                curr_seq_id += [inst] * len(curr_res["head_total_area_util"])
                curr_inst_raw_density += [float(len(headgrp_idx[h_idx])) / float(total_nblks)] * len(curr_res["head_total_area_util"])
                curr_effec_density += [sum(curr_res["head_effec_area"]) / float(total_nblks)] * len(curr_res["head_total_area_util"])

        # store all results of an inst to the entire result 
        all_inst_res_redidx_count.append(curr_inst_res_redidx_count)
        all_inst_res_total_idx_count.append(curr_inst_res_total_idx_count)
        all_inst_res_iarea_util.append(curr_inst_iarea_util)
        all_inst_res_tarea_util.append(curr_inst_tarea_util)
        all_inst_mean_route_ratio.append(curr_inst_mean_route_ratio)
        all_inst_raw_density.append(curr_inst_raw_density)
        all_inst_effec_density.append(curr_effec_density)
        all_inst_hidx.append(curr_hidx)
        all_inst_seq_ids.append(curr_seq_id)

    func_ret = {
        "task_res_redidx_count": all_inst_res_redidx_count, 
        "task_res_total_idx_count": all_inst_res_total_idx_count, 
        "task_res_iarea_util": all_inst_res_iarea_util, 
        "task_res_tarea_util": all_inst_res_tarea_util,
        "task_res_mean_route_ratio": all_inst_mean_route_ratio,
        "task_res_raw_density": all_inst_raw_density,
        "task_res_effec_density": all_inst_effec_density,
        "task_res_hidx": all_inst_hidx,
        "task_res_seq_ids": all_inst_seq_ids,
    }

    if aggre_method is not None:
        for k in func_ret.keys():
            func_ret[k] = aggre_method(func_ret[k])
    
    return func_ret

def compute_unique_colidx_ratio(task_list: list[str], model_names: list[str], swindow_list: list[int]):
    records = {
        "tasks": [], 
        "inst id": [],
        "head idx": [],
        "redundant index count": [], 
        "total index count": [], 
        "inner area util": [],
        "total area util": [],
        "raw head density": [],
        "effec head density": [],
        "mean_route_ratio": [],
        "model": [], 
        "swindow": [],
    }
    
    for swindow in swindow_list:
        for model_name in model_names:
            for task_name in (task_list):
                print(f"working on {model_name}-{task_name}")
                base_attn_path = f"/compas-old/projects/sparse-attention/{model_name}-attn-bfp20-{task_name}/"
                inst_rlist = util.get_pts_under_dir(base_attn_path, postfix="npy", datatype="ridx", fname_filter="iiSeqInst")
                inst_clist = util.get_pts_under_dir(base_attn_path, postfix="npy", datatype="cidx", fname_filter="iiSeqInst") 
                insts = [i.split("/")[-1].split("_ridx.npy")[0] for i in inst_rlist]

                head_ids = {}
                seq_lens = {}
                for inst in insts:
                    hwconfig_path = f"/compas-old/projects/sparse-attention/onchip-5hbm/{model_name}-attn-bfp20-{task_name}/{inst}/"
                    # get json files start with "hwconfig" under hwconfig_path
                    hwconfig_list = util.get_pts_under_dir(hwconfig_path, postfix="json")
                    hwconfig_list = [os.path.basename(pa) for pa in hwconfig_list if os.path.basename(pa).startswith("hwconfig")]
                    # extract head id from each member of hwconfig_list following the pattern "hwconfig_h<head_id>.json"
                    hwconfig_head_ids = [int(os.path.basename(pa).split("_h")[1].split(".")[0]) for pa in hwconfig_list]
                    head_ids[inst] = hwconfig_head_ids
                    print(f"head list: {hwconfig_head_ids} for inst {inst}")
                
                # compute all metric for all insts
                ret = get_unique_colidx_ratio_inst(inst_rlist, inst_clist, int(swindow), lambda x: list(chain(*x)), head_ids)
                # make sure the length of task_res_redidx_count and task_res_total_idx_count are the same
                assert len(ret["task_res_redidx_count"]) == len(ret["task_res_total_idx_count"])
                
                records["tasks"] += [task_name] * len(ret["task_res_redidx_count"])
                records["inst id"] += ret["task_res_seq_ids"]
                records["head idx"] += ret["task_res_hidx"]
                records["redundant index count"] += ret["task_res_redidx_count"]
                records["total index count"] += ret["task_res_total_idx_count"]
                records["inner area util"] += ret["task_res_iarea_util"]
                records["total area util"] += ret["task_res_tarea_util"]
                records["raw head density"] += ret["task_res_raw_density"]
                records["effec head density"] += ret["task_res_effec_density"]
                records["mean_route_ratio"] += ret["task_res_mean_route_ratio"]
                records["model"] += [model_name] * len(ret["task_res_redidx_count"])
                records["swindow"] += [swindow] * len(ret["task_res_redidx_count"])

    records = pd.DataFrame(records)
    # store records as pandas dataframe
    records.to_csv(f"/compas-old/projects/sparse-attention/onchip-5hbm/spars-analysis-onchip-related.csv", index=False, mode="w")

def plot_redunt_colidx_ratio_distribution():
    # read records from csv
    records = pd.read_csv(f"/compas-old/projects/sparse-attention/redidx-areautil-all.csv")
    # select records that has swindow = 12
    records = records[records['swindow'] == 12]
    # add a new column "removable redundant index ratio"
    records["removable redundant index ratio"] = records["redundant index count"] / records["total index count"]

    legend_color = {"llama2-7b-chat-4k": "C1", "mixtral-8x7b": "C2", "chatglm2-6b-32k": "C0"}
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.set_theme(font_scale=1.0)
    records = records[records["tasks"]=="lcc"]

    sns_plot = sns.histplot(
        data=records[records["model"]=="llama2-7b-chat-4k"], 
        x="removable redundant index ratio", 
        bins=30, 
        color=legend_color["llama2-7b-chat-4k"],
        kde=False)
    sns_plot.set_xticks(np.arange(0, 1.2, 0.2))
    
    plt.tight_layout()
    plt.savefig(f"./res_fig/block_prune/redidx-ratio-dist-omit-last-iter.pdf")


def plot_unique_colidx_ratio_boxplot_by_task(records: pd.DataFrame):
    # select records that has swindow = 12
    selected_records = records.loc[records['swindow'] == 12, :].copy()
    # add a new column "removable redundant index ratio"
    selected_records["removable redundant index ratio"] = \
        selected_records["redundant index count"] / selected_records["total index count"]
    # selected_records["effective sparsity"] = 1. - selected_records["total area util"]
    selected_records["effective sparsity"] = 1. - selected_records["effec head density"]
    selected_records["original sparsity"] = 1. - selected_records["raw head density"]
    selected_records["tensor block utilization"] = selected_records["inner area util"]

    # print(selected_records[selected_records["effec head density"] > 0.98])
    # with open("./res_fig/block_prune/very_dense_heads.txt", "w+") as f:
    #     f.writelines(selected_records[selected_records["effec head density"] > 0.98][["inst id", "head idx", "tasks", "model"]].drop_duplicates().to_string())

    global MODEL_PALETTE

    # boxplot with errorbars 
    sns.set(rc={'figure.figsize': (10, 6)}, font_scale=1.3)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.2)

    lm = sns.boxplot(
        data=selected_records, 
        x = "tasks", 
        y = "effective sparsity",
        hue = "model",
        legend=True,
        palette={k:v[1] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
        gap=.1,
        width=0.9,
        flierprops={"alpha": 0.3},
        whis=[1,99], 
        ax=axes[0]
    )

    errorbar_ax = axes[0].twiny()
    mean_sparsity = selected_records[["model", "tasks", "effective sparsity"]] \
                        .groupby(["model", "tasks"]).mean()["effective sparsity"]
    print(f"{mean_sparsity}")
    # print(f"mean of effective sparsity: {mean_sparsity.min()}, {mean_sparsity.max()}")
    sns.pointplot(
        data=selected_records, 
        x="tasks", 
        y="effective sparsity", 
        errorbar="sd", 
        hue="model", 
        err_kws={'linewidth': 5},
        ax=errorbar_ax,
        legend=False,
        dodge=0.6,
        linestyle="none",
        palette={k:v[0] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
    )
    axes[0].set(ylim=(-0.01, 1.01))
    axes[0].legend(title=None)
    axes[0].set_xticklabels([])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Effective sparsity")
    axes[0].tick_params(bottom=False)
    errorbar_ax.set_xticklabels([])
    errorbar_ax.set_xlabel("")
    errorbar_ax.tick_params(top=False) 

    lm = sns.boxplot(
        data=selected_records, 
        x = "tasks", 
        y = "original sparsity",
        hue = "model",
        legend=False,
        palette={k:v[1] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
        width=0.9,
        gap=.1,
        flierprops={"alpha": 0.5},
        whis=[1,99], 
        ax=axes[1]
    )

    errorbar_ax = axes[1].twiny()
    sns.pointplot(
        data=selected_records, 
        x="tasks", 
        y="original sparsity", 
        errorbar="sd", 
        hue="model", 
        err_kws={'linewidth': 5},
        ax=errorbar_ax,
        legend=False,
        dodge=0.6,
        linestyle="none",
        palette={k:v[0] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
    )

    axes[1].set(ylim=(-0.01, 1.01))
    axes[1].set_ylabel("Pre-aggregation sparsity")
    axes[1].set_xlabel("Tasks")
    errorbar_ax.set_xticklabels([])
    errorbar_ax.set_xlabel("")
    errorbar_ax.tick_params(top=False, bottom=False)
    labels = [label.get_text() for label in axes[1].get_xticklabels()]
    original_ticks = axes[1].get_xticks()
    axes[1].set_xticklabels(labels, rotation=30, ha='right')
    axes[1].set_xticks([ox + 0.2 for ox in original_ticks])
    axes[1].tick_params(top=False, bottom=False)
    sns.move_legend(axes[0], "upper center", ncol=3, bbox_to_anchor=(0.5, 1.25))
    plt.savefig(f"./res_fig/block_prune/effective_sparsity_boxplot.png", bbox_inches='tight')

    # vertical version of the boxplot
    sns.set(rc={'figure.figsize': (10, 11)}, font_scale=1.3)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(1,2)
    plt.subplots_adjust(wspace=0.1)

    lm = sns.boxplot(
        data=selected_records, 
        x = "effective sparsity",
        y = "tasks",
        hue = "model",
        legend=True,
        palette={k:v[1] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
        gap=.1,
        flierprops={"alpha": 0.5},
        whis=[1,99], 
        ax=axes[0]
    )

    errorbar_ax = axes[0].twiny()
    sns.pointplot(
        data=selected_records, 
        x= "effective sparsity",
        y= "tasks",
        errorbar="sd", 
        hue="model", 
        err_kws={'linewidth': 5},
        ax=errorbar_ax,
        legend=False,
        dodge=0.53,
        linestyle="none",
        palette={k:v[0] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
    )
    
    labels = [label.get_text() for label in axes[0].get_yticklabels()]
    axes[0].set_yticklabels(labels, rotation=40)
    axes[0].set(xlim=(-0.01, 1.01))
    axes[0].set_xlabel("Effective sparsity")
    axes[0].set_ylabel("Tasks")
    axes[0].tick_params(bottom=False)
    axes[0].legend(title=None)
    errorbar_ax.grid(visible=False)
    errorbar_ax.set_xticklabels([])
    errorbar_ax.set_xlabel("")
    errorbar_ax.tick_params(top=False) 
    sns.move_legend(axes[0], "upper center", ncol=3, bbox_to_anchor=(1.06, 1.07))
    # plt.savefig(f"./res_fig/block_prune/effective_sparsity_boxplot_vertical_upper.png", bbox_inches='tight', dpi=400)

    # fig, axes = plt.subplots(1,1)
    lm = sns.boxplot(
        data=selected_records, 
        x = "original sparsity",
        y = "tasks",
        hue = "model",
        legend=False,
        palette={k:v[1] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
        gap=.1,
        flierprops={"alpha": 0.5},
        whis=[1,99], 
        ax=axes[1]
    )

    errorbar_ax = axes[1].twiny()
    sns.pointplot(
        data=selected_records, 
        x="original sparsity", 
        y="tasks",
        errorbar="sd", 
        hue="model", 
        err_kws={'linewidth': 5},
        ax=errorbar_ax,
        legend=False,
        dodge=0.53,
        linestyle="none",
        palette={k:v[0] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
    )
    axes[1].set(xlim=(-0.01, 1.01))
    axes[1].set_xlabel("Pre-aggregation sparsity")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    errorbar_ax.grid(visible=False)
    errorbar_ax.set_xticklabels([])
    errorbar_ax.set_xlabel("")
    errorbar_ax.tick_params(top=False) 
    # sns.move_legend(axes, "upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    plt.savefig(f"./res_fig/block_prune/effective_sparsity_boxplot_vertical_2col.png", bbox_inches='tight', dpi=400)


def plot_route_ratio_by_task(records: pd.DataFrame):
    # select records that has swindow = 12
    selected_records = records.loc[records['swindow'] == 12, :].copy()

    global MODEL_PALETTE

    sns.set_theme(rc={'figure.figsize': (18, 5)}, font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    lm = sns.boxplot(
        data=selected_records, 
        x = "tasks", 
        y = "mean_route_ratio",
        hue = "model",
        legend=True,
        palette={k:v[1] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
        gap=.1,
        flierprops={"alpha": 0.5},
        whis=[1,99]
    )
    ax = lm.axes
    errorbar_ax = ax.twiny()

    sns.pointplot(
        data=selected_records, 
        x="tasks", 
        y="mean_route_ratio", 
        errorbar="sd", 
        hue="model", 
        err_kws={'linewidth': 5},
        ax=errorbar_ax,
        legend=False,
        dodge=0.53,
        linestyle="none",
        palette={k:v[0] for k, v in zip(MODEL_PALETTE.keys(), MODEL_PALETTE.values())},
    )
    # lm.set(ylim=(0.0, 1.0))
    # lm.get_legend().set_title(None)
    
    ax.legend(title=None)
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        # labels.append(textwrap.fill(text, width=10, break_on_hyphens=True))
        labels.append(text)
    ax.set_xticklabels(labels, rotation=10)
    errorbar_ax.set_xticklabels([])
    errorbar_ax.set_xlabel("")

    sns.move_legend(ax, "upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))
    plt.savefig(f"./res_fig/block_prune/route_ratio_boxplot.pdf", bbox_inches='tight')

def plot_unique_colidx_ratio_by_swindow(records: pd.DataFrame, fixed_dim_size=8, budget=720):
    '''
    fixed_dim_size is the size of the fixed hardware dimension in hw shape sweeping
    '''
    models = records['model'].unique()
    tasks = records['tasks'].unique()
    swindow_list = sorted(records["swindow"].unique())
    # group the records by tasks, model and swindow
    selected_records = records.loc[records['swindow'] <= 128, :].copy()
    selected_records["effective sparsity"] = 1. - selected_records["total area util"]
    grouped_records = selected_records.groupby(['swindow'])['effective sparsity'].mean().reset_index()

    # get latency records of different configs
    # structure: {modelname: {fix_r:[], fix_cl:[]}}
    lat_dat = {"fix_r": {}, "fix_cl": {}}
    all_recs = {}
    for mname, taskname in product(models, tasks):
        fpath = f"./res_fig/block_prune/idxmerge_window_experi/{mname}-attn-bfp20-{taskname}/"
        lat_files = util.get_pts_under_dir(fpath, "json")
        for lat_file in lat_files:
            hwconfig_in_fname = os.path.basename(lat_file).split(".")[0].split("_")[-3:]
            r, c, cl = int(hwconfig_in_fname[0][1:]), int(hwconfig_in_fname[1][1:]), int(hwconfig_in_fname[2][2:])
            print(f"find config {os.path.basename(lat_file)}: {(r, c, cl)}")
            assert r*c*(cl+2) == budget
            with open(lat_file, "r") as f:
                sparse_lat_profile = json.load(f)
                sparse_tops = np.mean([int(sparse_lat_profile[k]["avg_tops"]) for k in sparse_lat_profile.keys()])
                avg_load_lat = np.mean([int(sparse_lat_profile[k]["avg_tbc_load_lat"]) for k in sparse_lat_profile.keys()])
                avg_compute_lat = np.mean([int(sparse_lat_profile[k]["avg_tbc_compute_lat"]) for k in sparse_lat_profile.keys()])
                all_recs[f"{c}, {r}, {cl}"] = all_recs.get(f"{c}, {r}, {cl}", []) + \
                                                [{"tp": sparse_tops, "load": avg_load_lat, "compute": avg_compute_lat}]
                if cl == fixed_dim_size:
                    lat_dat["fix_cl"][c] = \
                        lat_dat["fix_cl"].get(c, []) + \
                        [{"tp": sparse_tops, "load": avg_load_lat, "compute": avg_compute_lat}]
                if r == fixed_dim_size:
                    lat_dat["fix_r"][c] = \
                        lat_dat["fix_r"].get(c, []) + \
                        [{"tp": sparse_tops, "load": avg_load_lat, "compute": avg_compute_lat}]

    # combine and average all configs 
    avg_all_recs = {}
    for config in all_recs.keys():
        avg_tp = np.mean([i["tp"] for i in all_recs[config]])
        avg_load = np.mean([i["load"] for i in all_recs[config]])
        avg_compute = np.mean([i["compute"] for i in all_recs[config]])
        avg_all_recs[config] = {"tp": float(avg_tp), "load": float(avg_load), "compute": float(avg_compute)} 

    with open("res_fig/block_prune/dse.all.json", "w") as f:
        json.dump(dict(sorted(avg_all_recs.items())), f)
    exit()
    # plot the grouped records as scatter plot with line connecting the points,
    # separate lines for each model and tasks, for different models use different colors
    # use different line style for different tasks
    # use swindow on x-axis with log scale, unique index ratio on y-axis
    global MODEL_PALETTE
    plt.rcParams.update({'font.size': 22})
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    ln1=ax1.plot(grouped_records['swindow'], 
                grouped_records['effective sparsity'], 
                marker='s', markersize=20, 
                label=f"effective sparsity", linewidth=5, color="C2")
    # ln2=ax2.plot(sorted(lat_dat["fix_r"].keys()), 
    #             [np.mean(d[1]) for d in sorted(lat_dat["fix_r"].items())], 
    #             marker='o', markersize=20, linestyle="-.",
    #             label=f"fixed #row", linewidth=5, color="C4")
    # ln3=ax2.plot(sorted(lat_dat["fix_cl"].keys()), 
    #             [np.mean(d[1]) for d in sorted(lat_dat["fix_cl"].items())], 
    #             marker='v', markersize=20, linestyle="--",
    #             label=f"fixed chain length", linewidth=5, color="C4")
    
    selected_max_tps, selected_ld_lat, selected_cp_lat = [], [], []
    for d0, d1 in zip(sorted(lat_dat["fix_cl"].items()), sorted(lat_dat["fix_r"].items())):
        mean_tp_d0 = np.mean([record["tp"] for record in d0[1]]) 
        mean_tp_d1 = np.mean([record["tp"] for record in d1[1]])
        selected_max_tps.append(max(mean_tp_d0, mean_tp_d1))
        if mean_tp_d0 > mean_tp_d1:
            selected_rec = lambda x: np.mean([record[x] for record in d0[1]])
            selected_ld_lat.append(selected_rec("load"))
            selected_cp_lat.append(selected_rec("compute"))
        else:
            selected_rec = lambda x: np.mean([record[x] for record in d1[1]])
            selected_ld_lat.append(selected_rec("load"))
            selected_cp_lat.append(selected_rec("compute"))

    ln2=ax2.plot(sorted(lat_dat["fix_cl"].keys()), 
                selected_max_tps, 
                marker='v', markersize=20, linestyle="--",
                label=f"fixed chain length", linewidth=5, color="C4")
    
    # lns = ln1 + ln2
    # labs = [l.get_label() for l in lns]
    # ax2.legend(lns, labs, loc="lower right")

    print(f"list of swindows: {grouped_records['swindow']}")
    print(f"list of effec spar: {grouped_records['effective sparsity']}")
    print(f"list of tops: {selected_max_tps}")
    print(f"list of ld lats: {selected_ld_lat}")
    print(f"list of cp lats: {selected_cp_lat}")

    plt.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    ax1.set_xlim(xmin=1)
    ax1.set_ylim(ymin=0, ymax=1)
    ax1.set_xlabel("#Index merging & sorting tree entries")
    ax1.set_ylabel("Average effective sparsity", color="C2")
    ax1.spines['left'].set_color('C2')
    ax1.tick_params(axis='y', colors='C2')

    ax2.spines['right'].set_color('C4')
    ax2.tick_params(axis='y', colors='C4')
    ax2.set_ylim(ymin=0)
    ax2.set_ylabel("TOPS", color="C4")

    plt.savefig("./res_fig/block_prune/effective-density-by-swindow.pdf", bbox_inches='tight')
    plt.clf()

    return

def plot_hw_perf(json_path_lists: list[str], label_list: list[str], res_path: str):
    dat_delay = {c:[] for c in json_path_lists}
    dat_tp = {c:[] for c in json_path_lists}
    for json_path in json_path_lists:
        num_layers = 28
        for l in range(num_layers):
            with open(os.path.join(json_path, f"src_data_layer_{l}.json"), "r") as fp:
                lat_dat = json.load(fp)
                dat_delay[json_path].append(lat_dat["compress blue"]["attxv"]["blocked prune sparse"][0])
                dat_tp[json_path].append(lat_dat["compress blue"]["attxv"]["blocked prune sparse"][1])

    # plot latency vs layers
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    fsize = 9
    bar_width = 0.15
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    x_labels = np.arange(1, num_layers+1, 1)
    for i, p in enumerate(json_path_lists):
        #compute bar idx
        if len(json_path_lists) % 2 == 0:
            x_pos = x_labels + (i - len(json_path_lists) / 2) * bar_width + bar_width / 2.0
        else:
            x_pos = x_labels + (i - (len(json_path_lists) - 1) / 2.0) * bar_width

        ax.bar(x_pos, dat_delay[p], 
                width=bar_width, color=f"C{i}", linewidth=1, label=label_list[i])

    ax.set_ylabel('Latency (sec)')
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xticks(x_labels)
    ax.set_xticklabels(x_labels)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    ax.set_xlabel('Layer')
    ax.legend()
    fig.tight_layout()
    fig.savefig(res_path + "/res_delay_s.pdf")
    fig.clf()


    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    x_labels = np.arange(1, num_layers+1, 1)
    for i, p in enumerate(json_path_lists):
        #compute bar idx
        if len(json_path_lists) % 2 == 0:
            x_pos = x_labels + (i - len(json_path_lists) / 2) * bar_width + bar_width / 0.2
        else:
            x_pos = x_labels + (i - (len(json_path_lists) - 1) / 2.0) * bar_width

        ax.bar(x_pos, dat_tp[p], 
                width=bar_width, color=f"C{i}", linewidth=1, label=label_list[i])

    ax.set_ylabel('Throughput (TOPs)')
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    ax.set_xticks(x_labels)
    ax.set_xticklabels(x_labels)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    ax.set_xlabel('Layer')
    ax.legend()
    fig.tight_layout()
    fig.savefig(res_path + "/res_tp_s.pdf")
    fig.clf()


def plot_rrspan_sweep(dat_file, n_layers, n_heads):
    '''
    plot how redundancy removal span affects the number of blocks after pruning 
    '''
    with open(dat_file, "r") as f:
        dat = f.readlines()
    
    res = {}
    for h_raw in dat:
        h_name = h_raw.split(":")[0]
        h_dat = h_raw.split(":")[1].strip("[]\n").split(", ")
        h_dat = [float(i) for i in h_dat]
        res[h_name] = h_dat

    fig, ax = plt.subplots(4, 7, figsize=(16, 10))
    for l in range(n_layers):
        for h in range(n_heads):
            ax[l // 7][l % 7].plot([0,1,2,4,6,8,10], res[f"l{l}h{h}"], color=f"C{h}", marker='.')
            ax[l // 7][l % 7].set_xlim(xmin=0)
            ax[l // 7][l % 7].set_ylim(ymin=0, ymax=1.0)
    
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    fig.supxlabel("redundancy removal window")
    fig.supylabel("#blocks remained/#blocks w/o removal")
    fig.savefig("res_fig/block_prune/sweep_rrspan.pdf")


def plot_rr_spmm_latdiff(dat_file, n_layers, n_heads, res_file):
    with open(dat_file, "r") as f:
        dat = json.load(f)

    latdiff_profile = {}
    for l in range(n_layers):
        fig, ax = plt.subplots(4, 8, figsize=(20, 10))
        for h in range(n_heads):
            lat_diff_per_head = []
            for inst_idx, inst in enumerate(dat[f"l{l}h{h}"]["lat_diff"]):
                lat_diff_per_head += inst
                ax[h // 8][h % 8].plot(
                    list(range(len(inst))), 
                    inst, 
                    color=f"C{inst_idx}", 
                    alpha=0.3,
                    marker='.', 
                    label=f"i{inst_idx}:{np.mean(inst):.1f}")
                
            ax[h // 8][h % 8].set_xlim(xmin=0)
            ax[h // 8][h % 8].legend(loc="upper left")
            latdiff_profile[f"l{l}h{h}"] = {"min": np.amin(lat_diff_per_head), "mean": np.mean(lat_diff_per_head)}
    
        fig.tight_layout(rect=(0.03, 0.03, 1, 1))
        fig.supxlabel("iteration")
        fig.supylabel("SpMM lat - redundancy removal lat (ns)")
        fig.savefig(f"res_fig/block_prune/spmm_rr_lat_diff_wfifo/l{l}.pdf")
        fig.clf()
    
    with open(res_file, "w") as f:
        json.dump(latdiff_profile, f)

def plot_rr_spmm_lat(dat_files, n_layers, n_heads, res_file):
    dats = []
    for pa in dat_files:
        with open(pa, "r") as f:
            dats.append(json.load(f))
    
    res = {}
    for fname, dat in zip(dat_files, dats):
        latdiff_profile, lat_profile, tops_profile = [], [], []
        for l, h in product(range(n_layers), range(n_heads)):
            lat_diff_per_head = []

            if dat[f"l{l}h{h}"]["lat_diff"]:
                for inst in dat[f"l{l}h{h}"]["lat_diff"]:
                    lat_diff_per_head += inst[1:]
                
                if lat_diff_per_head:
                    latdiff_profile += lat_diff_per_head

            lat_profile += dat[f"l{l}h{h}"]["total_lat"]
            tops_profile += dat[f"l{l}h{h}"]["total_tops"]
        
        res[f"{os.path.basename(fname)}"] = {}
        if latdiff_profile:
            res[f"{os.path.basename(fname)}"]["latdiff min"] = np.amin(latdiff_profile)
            res[f"{os.path.basename(fname)}"]["latdiff mean"] = np.mean(latdiff_profile)

        res[f"{os.path.basename(fname)}"]["lat mean"] = np.mean(lat_profile)
        res[f"{os.path.basename(fname)}"]["tops mean"] = np.mean(tops_profile)

    with open(res_file, "w") as f:
        json.dump(res, f, indent=2)


def get_onchip_res(dat_path, models_name, tasks_name, spmm_freq=300.0, tc_core_shape=(6, 12, 8), threshold_postfix=None):
    if dat_path[-1] != "/":
        dat_path += "/"

    spmm_cycle_delay = 1./spmm_freq * 1000.
    mat_b_load_delay_factor = 3.0

    # create pandas dataframe with columns: model,  task, onchip_lat, onchip_tp, seq_len
    df = pd.DataFrame(
        columns=[
            "model", 
            "task", 
            "inst_id", 
            "head_id", 
            "onchip_total_lat", 
            "onchip_total_tp", 
            "onchip_comp_lat",
            "onchip_comp_tp",
            "onchip_mat_b_load_lat",
            "seq_len", 
            "dense_lat", 
            "dense_tp"
        ]
    )

    # get the onchip res from the dat_path
    for model, task in product(models_name, tasks_name):
        if threshold_postfix:
            model_task_path = os.path.join(dat_path + f"{model}-attn-bfp20-{task}-{threshold_postfix}/")
        else:
            model_task_path = os.path.join(dat_path + f"{model}-attn-bfp20-{task}/")

        # get subdirs under model_task_path
        subdirs = [os.path.join(model_task_path, d) for d in os.listdir(model_task_path) if os.path.isdir(os.path.join(model_task_path, d))]
        for subdir in subdirs:
            # get the inst_id from the subdir name
            inst_id = subdir.split("/")[-1]
            # temporarily skip the random records
            if inst_id[0:9] != "iiSeqInst":
                continue 
            # get the seq_len from the subdir's "inst_profile.json" file
            with open(os.path.join(subdir, "inst_profile.json"), "r") as f:
                inst_profile = json.load(f)
                seq_len = inst_profile["seq_len"]

            # get dense res based on seq_len
            fake_dense_data = np.ones((seq_len, seq_len))
            fake_dense_data = np.tril(fake_dense_data)
            hw_array_shape = (tc_core_shape[0], tc_core_shape[1])
            dense_model = hw_modeling.StratixDpuModel(seq_len, 
                                                        seq_len, 
                                                        seq_len, 
                                                        128,
                                                        exp_dat=fake_dense_data, 
                                                        freq=spmm_freq, 
                                                        num_tcs=3960, 
                                                        tcc_array_shape=hw_array_shape, 
                                                        tcc_chainlen=tc_core_shape[2])
            dense_model.set_tccore_size(20)
            dense_flops, dense_lat, dense_util = \
                dense_model.tensor_fpga21_mat_sparse_flops(fake_dense_data, sparse_block_size=20)
            
            dense_res = PerfData()
            dense_res.total_lat += dense_lat
            total_ops = seq_len * seq_len * 2 * 128
            dense_res.add_data(dense_util, "util")
            dense_res.set_flops(total_ops)

            # search for all the files starting with "hwconfig_h" and ending with ".json" under subdir
            hwconfig_files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.startswith("hwconfig_h") and f.endswith(".json")]
            for hwconfig_file in hwconfig_files:
                hwconfig = None
                # get the head id from the hwconfig_file with the format "hwconfig_h{head_id}.json"
                hid = int(hwconfig_file.split("_")[-1].split(".")[0][1:])
                with open(hwconfig_file, "r") as f:
                    hwconfig = json.load(f)

                if hwconfig.get("total_lat_counter_res", None) is not None:
                    # get the onchip_lat and onchip_tp from the hwconfig
                    total_latency_res = hwconfig["total_lat_counter_res"] * spmm_cycle_delay
                    total_spmm_rr_flops = float(total_ops) / (float(total_latency_res) * 1e-9) / 1e12
                    componly_latency_res = hwconfig["compute_lat_counter_res"] * spmm_cycle_delay
                    comp_spmm_rr_flops = float(total_ops) / (float(componly_latency_res) * 1e-9) / 1e12
                    mat_b_load_delay = hwconfig["mat b size"] * spmm_cycle_delay / mat_b_load_delay_factor

                    # append the onchip_lat, onchip_tp, seq_len to the df
                    df.loc[len(df)] = {
                                        "model": model, 
                                        "task": task, 
                                        "inst_id": inst_id, 
                                        "head_id": hid,
                                        "onchip_total_lat": total_latency_res, 
                                        "onchip_total_tp": total_spmm_rr_flops, 
                                        "onchip_comp_lat": componly_latency_res, 
                                        "onchip_comp_tp": comp_spmm_rr_flops, 
                                        "onchip_mat_b_load_lat": mat_b_load_delay,
                                        "seq_len": seq_len, 
                                        "dense_lat": dense_res.total_lat, 
                                        "dense_tp": dense_res.total_flops
                    }

    if threshold_postfix:
        outpath = pathlib.Path(dat_path) / pathlib.Path(f"onchip_res_t{threshold_postfix}_{int(spmm_freq)}mhz.csv")
    else:
        outpath = pathlib.Path(dat_path) / pathlib.Path(f"onchip_res_{int(spmm_freq)}mhz.csv")
    
    df.to_csv(str(outpath.absolute()))
    return df

def get_gemm_emulated_res(
        dat_paths: list[pathlib.Path], 
        tc_core_shape: tuple[int, int, int],
        gemm_freq,
        csv_out_path: pathlib.Path
    ):
    # create pandas dataframe with columns: model,  task, onchip_lat, onchip_tp, seq_len
    df = pd.DataFrame(
        columns=[
            "seq_len", 
            "q_lat",
            "kv_lat",
            "qkT_lat",
            "aV_lat",
            "o_lat",
            "q_tops",
            "kv_tops",
            "o_tops",
            "aV_tops",
            "qkT_tops",
            "q_gemm_tops",
            "kv_gemm_tops",
            "o_gemm_tops",
            "aV_gemm_tops"
        ]
    )

    # get the emulated res from the dat_path
    workload_types = ["qkv", "qkT", "aV"]
    rec = {}
    for dat_path in dat_paths:
        for workload_type in workload_types:
            # get the seq_len from the subdir's "inst_profile.json" file
            with (dat_path / f"{workload_type}/inst_profile.json").open("r") as f:
                inst_profile = json.load(f)
                seq_len = inst_profile["seq_len"]

            workload_rp = dat_path / f"{workload_type}_ridx.npy"
            workload_cp = dat_path / f"{workload_type}_cidx.npy"
            print(f"loading {str(workload_rp)} and {str(workload_cp)}...")
            # preprocess index inputs
            ridx_dat = np.load(workload_rp)
            cidx_dat = np.load(workload_cp)
            # extract one head
            headgrp_ridx, headgrp_cidx = [], []
            curr_head_ridx, curr_head_cidx = [], []
            for ridx, cidx in zip(ridx_dat, cidx_dat):
                if ridx == -1 and cidx == -1:
                    headgrp_ridx.append(curr_head_ridx.copy())
                    headgrp_cidx.append(curr_head_cidx.copy())
                    curr_head_ridx, curr_head_cidx = [], []
                else:
                    curr_head_ridx.append(ridx)
                    curr_head_cidx.append(cidx)

            inst_dat = []
            src_ridx, src_cidx = headgrp_ridx[0], headgrp_cidx[0] 
            inst_dat.append([(r, c) for r, c in zip(src_ridx, src_cidx)])


            if workload_type == "qkv":
                res_q = rr2spmm_fifo_latency_overlap_analysis(
                                                            {"SeqInst0000": inst_dat}, 
                                                            {"SeqInst0000": seq_len}, 
                                                            tc_core_shape[1], tc_core_shape, 
                                                            512, 
                                                            out_buff_depth=1024,
                                                            spmm_freq = gemm_freq,
                                                            rremover_freq = gemm_freq,
                                                            workload_type="q")

                res_kv = rr2spmm_fifo_latency_overlap_analysis(
                                                            {"SeqInst0000": inst_dat}, 
                                                            {"SeqInst0000": seq_len}, 
                                                            tc_core_shape[1], tc_core_shape, 
                                                            512, 
                                                            out_buff_depth=1024,
                                                            spmm_freq = gemm_freq,
                                                            rremover_freq = gemm_freq,
                                                            workload_type="kv")
                
                rec["q_lat"] = res_q["SeqInst0000"]["avg_total_lat"]
                rec["kv_lat"] = res_kv["SeqInst0000"]["avg_total_lat"]
                rec["o_lat"] = res_q["SeqInst0000"]["avg_total_lat"]
                rec["q_tops"] = res_q["SeqInst0000"]["avg_tops"]
                rec["kv_tops"] = res_kv["SeqInst0000"]["avg_tops"]
                rec["o_tops"] = res_q["SeqInst0000"]["avg_tops"]
                rec["q_gemm_tops"] = res_q["SeqInst0000"]["dense_avg_tops"]
                rec["kv_gemm_tops"] = res_kv["SeqInst0000"]["dense_avg_tops"]
                rec["o_gemm_tops"] = res_q["SeqInst0000"]["dense_avg_tops"]

            else:
                res = rr2spmm_fifo_latency_overlap_analysis(
                                                            {"SeqInst0000": inst_dat}, 
                                                            {"SeqInst0000": seq_len}, 
                                                            tc_core_shape[1], tc_core_shape, 
                                                            512, 
                                                            out_buff_depth=1024,
                                                            spmm_freq = gemm_freq,
                                                            rremover_freq = gemm_freq,
                                                            workload_type=workload_type)
                rec[f"{workload_type}_lat"] = res["SeqInst0000"]["avg_total_lat"]
                rec[f"{workload_type}_tops"] = res["SeqInst0000"]["avg_tops"]
                rec[f"{workload_type}_gemm_tops"] = res["SeqInst0000"]["dense_avg_tops"]


        rec["seq_len"] = seq_len
        df.loc[len(df)] = rec

    if csv_out_path.exists():
        existing_df = pd.read_csv(str(csv_out_path.absolute()))
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(str(csv_out_path.absolute()))
    return df

def get_gemm_onchip_res(
        dat_path: pathlib.Path, 
        models_name, 
        tasks_name, 
        gemm_freq=300.0
    ):
    gemm_cycle_delay = 1./gemm_freq * 1000.
    mat_b_load_delay_factor = 3.0

    def get_gemm_synth_lat(p: pathlib.Path, force_n_b_blks=-1):
        res = 0
        hw_res_path = p / "hwconfig_h0.json"

        if hw_res_path.exists():
            with hw_res_path.open("r") as f:
                hw_res = json.load(f)

            total_lat = hw_res["total_lat_counter_res"] * gemm_cycle_delay
            mat_b_load_lat = hw_res["mat_b_load_counter_res"] * gemm_cycle_delay / mat_b_load_delay_factor
            n_mat_b_blks = hw_res["mat b col blks"]
        else:
            raise FileNotFoundError(f"{str(hw_res_path)} not found!")
        
        # pipelining mat b blk loading and computation
        res = 0
        # for GQA: force k,v to have only one head
        if force_n_b_blks > 0:
            n_mat_b_blks = force_n_b_blks
        for mat_b_blk_idx in range(n_mat_b_blks):
            if mat_b_blk_idx == 0:
                res += mat_b_load_lat
            else:
                res += max(mat_b_load_lat, total_lat)

        res += total_lat
        return res

    # create pandas dataframe with columns: model,  task, onchip_lat, onchip_tp, seq_len
    df = pd.DataFrame(
        columns=[
            "seq_len", 
            "q_lat",
            "kv_lat",
            "qkT_lat",
            "aV_lat",
            "o_lat"
        ]
    )

    # get the onchip res from the dat_path

    subdirs = [dat_path / d for d in os.listdir(dat_path) if (dat_path / d).is_dir()]
    for subdir in subdirs:
        # get the inst_id from the subdir name
        matmul_type = [(subdir / d).name for d in os.listdir(subdir) if (subdir / d).is_dir()]
        for t in matmul_type:
            assert t in ["qkv", "qkT", "aV"], f"Unexpected matmul type: {t}"

        # get the seq_len from the subdir's "inst_profile.json" file
        with (subdir / "inst_profile.json").open("r") as f:
            inst_profile = json.load(f)
            seq_len = inst_profile["seq_len"]

        q_lat = get_gemm_synth_lat(subdir / "qkv")
        kv_lat = get_gemm_synth_lat(subdir / "qkv", force_n_b_blks=1)
        qkT_lat = get_gemm_synth_lat(subdir / "qkT")
        aV_lat = get_gemm_synth_lat(subdir / "aV")

        df.loc[len(df)] = {
                            "seq_len": seq_len,
                            "q_lat": q_lat,
                            "kv_lat": kv_lat,
                            "qkT_lat": qkT_lat,
                            "aV_lat": aV_lat,
                            "o_lat": q_lat
        }

    outpath = pathlib.Path(dat_path) / pathlib.Path(f"onchip_res_gemms_{int(gemm_freq)}mhz.csv")
    df.to_csv(str(outpath.absolute()))
    return df

def plot_stacked_selfattn_ops_latency(
        gemm_dat: pd.DataFrame,
        n_heads = 32,
        out_fig_path = pathlib.Path("./res_fig/dense_lat_ops_seqlen.pdf")
        ):
    # drop all columns in gemm_dat with string values
    gemm_dat = gemm_dat.select_dtypes(include=["number"])
    # analyze latency breakdown of a single self-attn
    gemm_dat["qkv_lat"] = gemm_dat["q_lat"] + gemm_dat["kv_lat"] * 2
    gemm_dat["qkT_lat"] = gemm_dat["qkT_lat"] * n_heads
    gemm_dat["aV_lat"] = gemm_dat["aV_lat"] * n_heads
    gemm_dat["linear_lat"] = gemm_dat["qkv_lat"] + gemm_dat["o_lat"]
    gemm_dat["total_lat"] = gemm_dat["linear_lat"] + gemm_dat["qkT_lat"] + gemm_dat["aV_lat"]
    gemm_dat["linear_lat_prop"] = gemm_dat["linear_lat"] / gemm_dat["total_lat"]
    gemm_dat["qkT_lat_prop"] = gemm_dat["qkT_lat"] / gemm_dat["total_lat"]
    gemm_dat["aV_lat_prop"] = gemm_dat["aV_lat"] / gemm_dat["total_lat"]

    # --- Color and Hatch Mappings ---
    # Colors for models
    global MODEL_PALETTE

    # Hatches for latency types
    # average "q_lat", "kv_lat", "qkT_lat", "aV_lat", "o_lat" for different range of "seq_len"
    bins = [1024, 2048, 4096, 8192, 16384, 32768]
    gemm_dat["seq_len_bin"] = pd.cut(gemm_dat["seq_len"], bins=bins)
    gemm_dat_binned_mean = gemm_dat.groupby("seq_len_bin").mean().reset_index()
    print(gemm_dat_binned_mean.to_markdown())

    # plot the binned mean of "qkT_lat", "aV_lat", "o_lat" vs "seq_len_bin" using seaborn and label the bars using latency_hatches
    # for each mean, plot it to be stacked with previous type
    sns.set(rc={'figure.figsize':(8, 4.5)}, font_scale=1.2)
    fig, axes = plt.subplots(1, 1)
    bar_width = 0.7
    sns.barplot(gemm_dat, x="seq_len", y="linear_lat_prop", bottom=0,
                label="Linear", ax=axes, color="C4", width=bar_width)
    sns.barplot(gemm_dat, x="seq_len", y="aV_lat_prop", 
                bottom=gemm_dat["linear_lat_prop"],
                label="Attention × V", ax=axes, color="C5", width=bar_width)
    sns.barplot(gemm_dat, x="seq_len", y="qkT_lat_prop", 
                bottom=gemm_dat["linear_lat_prop"] + gemm_dat["aV_lat_prop"],
                label="Q × Kᵀ", ax=axes, color="C3", width=bar_width)

    # set x axis only show numbers in integer
    # axes.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    axes.set_xlabel("Context length")
    axes.set_ylabel("Normalized GEMM runtime \nin self-attention prefill stage")
    axes.set_ylim(ymin=0, ymax=1)
    axes.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.18), fancybox=False, columnspacing=0.4)
    fig.tight_layout()
    fig.savefig(out_fig_path, bbox_inches='tight')
    fig.clf()

def plot_onchip_res(dat: pd.DataFrame):
    dat["total_speedup"] = dat["onchip_total_tp"] / dat["onchip_comp_tp"]
    dat["comp_speedup"] = dat["onchip_comp_tp"] / dat["dense_tp"]
    # delete the "inst_id" column
    dat_ori = dat.copy()
    dat_inst_mean = dat.groupby(["model", "task", "inst_id"]).mean()
    dat_numbers_only = dat.drop(columns=["model", "task", "inst_id"])

    dat = dat.drop(columns=["inst_id"])
    # calculate the mean of onchip_tp of different inst_id for the same model and task
    dat_mean = dat.groupby(["model", "task"]).mean()
    # print(f"speedup by tasks: {dat_mean.to_string()}")
    dat_overall_mean = dat.drop(columns=["task"]).groupby("model").mean().reset_index()
    print(f"over all speedup by models: \n{dat_overall_mean[['model', 'onchip_comp_tp']].to_markdown()}")

    print(f"single core sparse throughput with onchip bd: {dat_numbers_only['onchip_comp_tp'].mean()} TOPS")
    print(f"single core dense throughput with onchip bd: {dat_numbers_only['dense_tp'].mean()} TOPS")

    global MODEL_PALETTE
    model_seq = ["chatglm2-6b-32k", "llama2-7b-chat-4k", "mixtral-8x7b"]
    task_seq = ["lcc", "multifieldqa_en", "multifieldqa_zh", "passage_retrieval_zh", "qasper", "samsum", "trec", "vcsum"]

    # plot normalized speed up
    # set the plot size to be 16,  6 for seaborn barplot
    sns.set(rc={'figure.figsize':(8, 4.5)}, font_scale=1.3)
    fig, axes = plt.subplots(1, 1)
    curr_model_palette = {k:MODEL_PALETTE[k][0] for k in MODEL_PALETTE.keys()}
    sns.barplot(dat_mean, x="task", y="comp_speedup", hue="model", width=0.65, 
                        palette=curr_model_palette, ax=axes)    
    labels = [label.get_text() for label in axes.get_xticklabels()]
    original_ticks = axes.get_xticks()
    axes.set_xticklabels(labels, rotation=30, ha='right')
    axes.set_xticks([ox + 0.2 for ox in original_ticks])
    axes.set_ylabel("Average normalized\n throughput of heads")
    axes.set_xlabel("Tasks")
    axes.set_ylim(ymin=0, ymax=5.5)
    axes.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))
    fig.savefig("./res_fig/block_prune/onchip_res_total_speedup.pdf", bbox_inches='tight')
    fig.clf()

    # vertical version of the same plot as above
    sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3)
    fig, axes = plt.subplots(1, 1)
    
    sns.barplot(dat_mean, x="comp_speedup", y="task", hue="model", width=0.8, 
                        palette=curr_model_palette, ax=axes, legend=True)
    axes.set_ylabel("Tasks")
    axes.set_xlabel("Average normalized throughput of heads")
    axes.set_xlim(xmin=0, xmax=4.5)
    axes.legend(title=None)
    sns.move_legend(axes, "upper center", ncol=3, bbox_to_anchor=(0.5, 1.09))
    fig.savefig("./res_fig/block_prune/onchip_res_total_speedup_vertical.pdf", bbox_inches='tight')
    fig.clf()

    # calculate onchip and offchip bandwidth
    # extract throughput without offchip loading
    onchip_thr_list = {m:{t: [] for t in task_seq} for m in model_seq}
    all_thr = []
    all_dense_thr = []
    n_queus = 2
    for m, t in product(model_seq, task_seq):
        selected = dat_ori.query(f"`model` == '{m}' and `task` == '{t}'")
        inst_ids = selected["inst_id"].unique()
        for i in inst_ids:
            inst_df = selected.query(f"inst_id == '{i}'").sort_values("head_id")
            inst_total_lats = list(inst_df["onchip_comp_lat"])
            inst_total_lats_dense = list(inst_df["dense_lat"])
            seq_len = inst_df["seq_len"].mean()
            total_ops = seq_len * seq_len * 2 * 128 * len(inst_total_lats)
            # balance all loads
            total_lats, _ = suboptimal_sublist_creator(inst_total_lats, n_queus)
            total_lats_dense, _ = suboptimal_sublist_creator(inst_total_lats_dense, n_queus)

            effec_total_lat = max([sum(l) for l in total_lats])
            effec_total_lat_dense = max([sum(l) for l in total_lats_dense])
            curr_ops = float(total_ops) / (float(effec_total_lat) * 1e-9) / 1e12
            # dense latency has a different time unit
            curr_ops_dense = float(total_ops) / float(effec_total_lat_dense) / 1e12
            onchip_thr_list[m][t].append(curr_ops) 
            all_thr.append(curr_ops)
            all_dense_thr.append(curr_ops_dense)

    print(f"dual core sparse throughput with onchip bd: {np.mean(all_thr)} TOPS")
    print(f"dual core dense throughput with onchip bd: {np.mean(all_dense_thr)} TOPS")

    # extract throughput with offchip loading assuming a single core
    offchip_thr_list = {m:{t: [] for t in task_seq} for m in model_seq}
    n_queus = 1
    all_thr = []
    for m, t in product(model_seq, task_seq):
        selected = dat_ori.query(f"`model` == '{m}' and `task` == '{t}'")
        inst_ids = selected["inst_id"].unique()
        for i in inst_ids:
            inst_df = selected.query(f"inst_id == '{i}'").sort_values("head_id")
            inst_total_lats = list(inst_df["onchip_total_lat"])
            inst_load_lats = list(inst_df["onchip_mat_b_load_lat"])
            seq_len = inst_df["seq_len"].mean()
            total_ops = seq_len * seq_len * 2 * 128 * len(inst_total_lats)
            total_lats = [[] for q in range(n_queus)]
            for q in range(n_queus):
                hlist = list(range(len(inst_total_lats)))[q:len(inst_total_lats):n_queus]
                for hidx in range(len(hlist) + 1):
                    if hidx == 0:
                        total_lats[q].append(inst_load_lats[hlist[hidx]])
                    elif hidx == (len(hlist)):
                        total_lats[q].append(inst_total_lats[hlist[-1]])
                    else:
                        total_lats[q].append(max(inst_load_lats[hlist[hidx-1]], inst_total_lats[hlist[hidx]]))

            effec_total_lat = max([sum(l) for l in total_lats])
            curr_ops = float(total_ops) / (float(effec_total_lat) * 1e-9) / 1e12
            offchip_thr_list[m][t].append(curr_ops) 
            all_thr.append(curr_ops)

    print(f"single core sparse throughput with offchip bd: {np.mean(all_thr)} TOPS")

    # extract throughput with offchip loading assuming a dual core
    offchip_thr_list = {m:{t: [] for t in task_seq} for m in model_seq}
    n_queus = 2
    all_thr = []
    for m, t in product(model_seq, task_seq):
        selected = dat_ori.query(f"`model` == '{m}' and `task` == '{t}'")
        inst_ids = selected["inst_id"].unique()
        for i in inst_ids:
            inst_df = selected.query(f"inst_id == '{i}'").sort_values("head_id")
            inst_total_lats = list(inst_df["onchip_total_lat"])
            inst_load_lats = list(inst_df["onchip_mat_b_load_lat"])
            seq_len = inst_df["seq_len"].mean()
            total_ops = seq_len * seq_len * 2 * 128 * len(inst_total_lats)
            total_lats = [[] for q in range(n_queus)]
            lats_wo_bload, sched_indices = suboptimal_sublist_creator(inst_total_lats, n_queus, True)
            for q in range(n_queus):
                for hidx in range(len(lats_wo_bload[q]) + 1):
                    if hidx == 0:
                        total_lats[q].append(inst_load_lats[sched_indices[q][hidx]])
                    elif hidx == (len(lats_wo_bload[q])):
                        total_lats[q].append(inst_total_lats[sched_indices[q][-1]])
                    else:
                        total_lats[q].append(
                            max(inst_load_lats[sched_indices[q][hidx-1]], inst_total_lats[sched_indices[q][hidx]]))

            effec_total_lat = max([sum(l) for l in total_lats])
            curr_ops = float(total_ops) / (float(effec_total_lat) * 1e-9) / 1e12
            offchip_thr_list[m][t].append(curr_ops) 
            all_thr.append(curr_ops)

    print(f"dual core sparse throughput with offchip bd: {np.mean(all_thr)} TOPS")

def plot_speedup_vs_sparsity(hw_perf_df: pd.DataFrame, density_df: pd.DataFrame):
    hw_perf_df["total_speedup"] = hw_perf_df["onchip_total_tp"] / hw_perf_df["onchip_comp_tp"]
    hw_perf_df["comp_speedup"] = hw_perf_df["onchip_comp_tp"] / hw_perf_df["dense_tp"]

    selected_hw_df = hw_perf_df[["model", "task", "inst_id", "head_id", "seq_len", "comp_speedup"]]
    selected_density_df = density_df[["model", "tasks", "inst id", "head idx", "effec head density", "raw head density"]]
    selected_density_df = selected_density_df.groupby(["model", "tasks", "inst id", "head idx"]).mean().reset_index()
    selected_density_df.rename(columns={"tasks": "task", "inst id" : "inst_id", "head idx": "head_id"}, inplace=True)
    dat = selected_hw_df.merge(selected_density_df, how="left", on=["model", "task", "inst_id", "head_id"])

    dat["raw head sparsity"] = 1. - dat["raw head density"]
    dat["effec head sparsity"] = 1. - dat["effec head density"]

    global MODEL_PALETTE
    curr_model_palette = {k:MODEL_PALETTE[k][0] for k in MODEL_PALETTE.keys()}
    
    # plot scatter figure
    models = ["llama2-7b-chat-4k", "chatglm2-6b-32k", "mixtral-8x7b"]
    tasks = ["lcc", "multifieldqa_en", "multifieldqa_zh", "passage_retrieval_zh", "qasper", "samsum", "trec", "vcsum"]
    
    sns.set(rc={'figure.figsize':(15, 15)}, font_scale=3.4)
    fig, axes = plt.subplots(1, 1)
    dat_sampled = dat.groupby('model').apply(lambda x: x.sample(n=200, random_state=42)).reset_index(drop=True)
    sns.scatterplot(
        dat_sampled, 
        x="effec head sparsity", 
        y="comp_speedup", 
        s=400, 
        ax=axes,
        alpha=0.65,
        hue="model",
        palette=curr_model_palette,
        style="model",
        legend=False,
    )
    
    axes.set_ylabel("Normalized throughput per head")
    axes.set_xlabel(f"Effective sparsity per head")
    axes.set_ylim(ymin=-0.01)
    axes.set_xlim(xmin=-0.01, xmax=1)

    fig.savefig("./res_fig/block_prune/speedup_vs_sparsity.pdf", dpi=1200, bbox_inches='tight')
    fig.clf()

    # plot speedup vs seq len
    dat_inst_mean = hw_perf_df.groupby(["model", "task", "inst_id"]).mean()
    sns.set(rc={'figure.figsize':(15, 15)}, font_scale=3.5)
    fig, axes = plt.subplots(1, 1)

    sns.scatterplot(dat_inst_mean, x="seq_len", y="comp_speedup",
                    s=300, alpha=0.8, hue="model", style="model", 
                    palette=curr_model_palette, ax=axes, legend=True)
        
    axes.set_ylabel("Average normalized throughput\n per instance")
    axes.set_xlabel("Context length")
    axes.set_ylim(ymin=-0.1, ymax=5.5)
    axes.set_xlim(xmin=-0.1)
    axes.legend(title=None)
    sns.move_legend(axes, "upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))

    fig.savefig("./res_fig/block_prune/onchip_res_speedup_vs_seqlen.pdf", bbox_inches='tight')
    fig.clf()

    sns.set(rc={'figure.figsize':(15, 15)}, font_scale=3.4)
    fig, axes = plt.subplots(1, 1)
    dat_inst_level = dat.groupby(["model", "task", "inst_id"]).mean().reset_index()
    sns.scatterplot(
        dat_sampled, 
        x="seq_len", 
        y="effec head sparsity", 
        s=400, 
        ax=axes,
        alpha=0.65,
        hue="model",
        palette=curr_model_palette,
        style="model",
        legend=False
    )
    
    axes.set_ylabel("Effective sparsity per head")
    axes.set_xlabel("Context length")
    axes.set_ylim(ymin=-0.01, ymax=1)
    axes.set_xlim(xmin=-0.01)
    
    fig.savefig("./res_fig/block_prune/sparsity_vs_seqlen.pdf", dpi=1200, bbox_inches='tight')
    fig.clf()

    # check correlation
    # Function to compute Pearson r for each group
    def compute_pearson(group):
        r_effc, p_effc = pearsonr(group['seq_len'], group['effec head sparsity'])
        r_raw, p_raw = pearsonr(group['seq_len'], group['raw head sparsity'])
        return pd.Series({
            'r_effc_spars': r_effc,
            'p_effc_spars': p_effc,
            'r_raw_spars': r_raw,
            'p_raw_spars': p_raw
        })

    # Group by 'model' and apply the function
    correlations = dat.groupby('model').apply(compute_pearson).reset_index()
    print(correlations)


def rr2spmm_wrap(all_inst_dat, seq_lens, hw_shape, freq, folding_factor, out_path): 
    rr2spmm_fifo_latency_overlap_analysis(
        all_inst_dat, seq_lens, 
        hw_shape[1], hw_shape, 
        512, 
        out_buff_depth=1024,
        spmm_freq = freq,
        rremover_freq = freq,
        out_json_basepath=out_path)

def sigma_wrap(all_inst_dat, seq_lens, hw_shape, freq, folding_factor, out_path): 
    sigma_latency_analysis(
        inst_list=all_inst_dat, 
        seqlen_list=seq_lens, 
        dpu_shape=hw_shape,
        n_matb_cols=128,
        transpose_folding_factor=folding_factor,
        spmm_freq=freq,
        out_json_basepath=out_path
    )

def naive_wrap(all_inst_dat, seq_lens, hw_shape, freq, folding_factor, out_path): 
    naive_roundrobin_latency_analysis(
        inst_list=all_inst_dat, 
        seqlen_list=seq_lens, 
        dpu_shape=hw_shape,
        n_matb_cols=128,
        spmm_freq=freq,
        out_json_basepath=out_path
    )

def dense_wrap(all_inst_dat, seq_lens, hw_shape, freq, folding_factor, out_path):
    tc_row, tc_col, tc_chain_len = hw_shape
    res = {}
    for inst_id in all_inst_dat.keys():
        seq_len = seq_lens[inst_id]
        fake_dense_data = np.ones((seq_len, seq_len))
        fake_dense_data = np.tril(fake_dense_data)
        total_ops = seq_len * seq_len * 128 * 2
        dense_model = hw_modeling.StratixDpuModel(seq_len, seq_len, seq_len, 128,
                                                    exp_dat=fake_dense_data, 
                                                    freq=freq, num_tcs=tc_col * tc_row * (tc_chain_len + 2), 
                                                    tcc_array_shape=(tc_row, tc_col), tcc_chainlen=tc_chain_len)
        dense_model.set_tccore_size(20)
        dense_flops, dense_lat, dense_util = \
            dense_model.tensor_fpga21_mat_sparse_flops(fake_dense_data, sparse_block_size=20)
        
        dense_res = PerfData()
        dense_res.total_lat += dense_lat
        dense_res.add_data(dense_util, "util")
        dense_res.set_flops(total_ops)

        res[inst_id] = {"total_lat": [dense_res.total_lat], "total_tops": [dense_res.total_flops]}
        res[inst_id]["avg_tops"] = dense_res.total_flops

    fpath = f"{out_path}/base_lat_r{tc_row}_c{tc_col}_cl{tc_chain_len}.json"
    if os.path.exists(fpath):
        with open(fpath, 'r') as file:
            try:
                existing_data = json.load(file)
                if not isinstance(existing_data, dict):
                    raise ValueError("The file does not contain a valid JSON object.")
            except json.JSONDecodeError:
                existing_data = {}
        
        existing_data.update(res)
    else:
        existing_data = res

    with open(fpath, 'w') as file:
        json.dump(existing_data, file, indent=2)

def sweep_rr_swindow_get_tops(model_name, task_name, tccore_budget, emulator, out_path):
    base_attn_path = f"/compas-old/projects/sparse-attention/{model_name}-attn-bfp20-{task_name}/"
    inst_rlist = sorted(util.get_pts_under_dir(base_attn_path, postfix="npy", datatype="ridx", fname_filter="iiSeqInst"))
    inst_clist = sorted(util.get_pts_under_dir(base_attn_path, postfix="npy", datatype="cidx", fname_filter="iiSeqInst"))

    insts = [i.split("/")[-1].split("_ridx.npy")[0] for i in inst_rlist]

    head_ids, seq_lens = {}, {}
    for inst in insts:
        hwconfig_path = f"/compas-old/projects/sparse-attention/onchip-5hbm/{model_name}-attn-bfp20-{task_name}/{inst}/"
        inst_profile_path = f"/compas-old/projects/sparse-attention/onchip-5hbm/{model_name}-attn-bfp20-{task_name}/{inst}/inst_profile.json"
        # get json files start with "hwconfig" under hwconfig_path
        hwconfig_list = util.get_pts_under_dir(hwconfig_path, postfix="json")
        hwconfig_list = [os.path.basename(pa) for pa in hwconfig_list if os.path.basename(pa).startswith("hwconfig")]
        # extract head id from each member of hwconfig_list following the pattern "hwconfig_h<head_id>.json"
        hwconfig_head_ids = [int(os.path.basename(pa).split("_h")[1].split(".")[0]) for pa in hwconfig_list]
        head_ids[inst] = hwconfig_head_ids
        # MARK: select only 5 heads temporary
        # head_ids[inst] = random.sample(hwconfig_head_ids, 5)
        with open(inst_profile_path, "r") as inst_pf:
            inst_pf_dict = json.load(inst_pf)
            seq_lens[inst] = int(inst_pf_dict["seq_len"])

    all_inst_dat = {}
    for inst_id, inst_rp, inst_cp in zip(insts, inst_rlist, inst_clist):
        print(f"loading {inst_rp} and {inst_cp}...")
        # preprocess index inputs
        ridx_dat = np.load(inst_rp)
        cidx_dat = np.load(inst_cp)
        # extract one head
        headgrp_ridx, headgrp_cidx = [], []
        curr_head_ridx, curr_head_cidx = [], []
        for ridx, cidx in zip(ridx_dat, cidx_dat):
            if ridx == -1 and cidx == -1:
                headgrp_ridx.append(curr_head_ridx.copy())
                headgrp_cidx.append(curr_head_cidx.copy())
                curr_head_ridx, curr_head_cidx = [], []
            else:
                curr_head_ridx.append(ridx)
                curr_head_cidx.append(cidx)

        # fetch a head
        idx_dat = []
        print(f"get {len(headgrp_ridx)} heads in total, using {len(head_ids[inst_id])} heads")
        for hidx in head_ids[inst_id]:
            src_ridx, src_cidx = headgrp_ridx[hidx], headgrp_cidx[hidx] 
            idx_dat.append([(r, c) for r, c in zip(src_ridx, src_cidx)])
        
        all_inst_dat[inst_id] = idx_dat

    # run the experiment
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if tccore_budget > 0.0:
        if emulator is rr2spmm_wrap:
            # get hw shapes for spmm core
            c_list = [4, 8, 12, 24, 36]
            hw_shapes = []
            prereq = lambda x1,x2: (x2 >= 8) and (x2 <= 32) and ((math.ceil(128./x1) >= 3*x2) or abs(math.ceil(128./x1) - 3*x2) < 40)
            for c in c_list:
                r_and_cl_pairs = find_positive_integer_pairs(int(tccore_budget // c), prereq)
                hw_shape = [(r_and_cl[0], c, r_and_cl[1]) for r_and_cl in r_and_cl_pairs]
                hw_shapes += hw_shape
            print(hw_shapes)

        if (emulator is naive_wrap) or (emulator is dense_wrap):
            # get hw shapes for naive and baseline core
            c_list = [4, 8, 12, 24, 36]
            hw_shapes = []
            prereq = lambda x1,x2: (x2 >= 8) and (x2 <= 32) and (128.0/x1 > 2)
            for c in c_list:
                r_and_cl_pairs = find_positive_integer_pairs(int(tccore_budget // c), prereq)
                hw_shape = [(r_and_cl[0], c, r_and_cl[1]) for r_and_cl in r_and_cl_pairs]
                hw_shapes += hw_shape
            print(hw_shapes)
        
        if emulator is sigma_wrap:
            # get hw shapes for SIGMA core
            c_list = [4, 8, 16, 32]
            hw_shapes = [(c, int(math.floor(tccore_budget / (c+2)))) for c in c_list]
            print(hw_shapes)

        freqs = [300.0] * len(hw_shapes)
        folding_factor = [2] * len(hw_shapes)
    else:
        ## or specify a shape
        if emulator is naive_wrap:
            # shapes for naive
            hw_shapes = [(1,4,15), (3,4,15), (3,8,12), (3,12,13), (4,12,11), (6, 12, 8),
                        (4,24,8), (4,24,11), (6,24,8), (5,36,8), (6,36,8), (7,36,8), (12,24,8)]
            freqs = [330.0, 320.0, 300.0, 280.0, 280.0, 300.0, 260.0, 240.0, 220.0, 200.0, 130.0, 130.0, 120.0]
            folding_factor = [2] * len(hw_shapes)

        if emulator is dense_wrap:
            # shapes for baseline
            hw_shapes = [(1,4,15), (3,4,15), (3,8,12), (3,12,13), (4,12,11), (2,24,13), (1,36,21),
                        (2,24,18), (3,24,13), (2,36,15), (2,36,18), (2,36,23), (6,36,8), (7,36,8), (8,36,8)]
            freqs = [370, 320, 310, 300, 300, 260, 250, 250, 230, 240, 220, 200, 150, 140, 120]
            folding_factor = [2] * len(hw_shapes)
            # this config is for sigma validation
            # hw_shapes = [(4, 4, 15)]
            # freqs = [300]
            # folding_factor = [1]

        if emulator is rr2spmm_wrap:
            # shapes for spmm
            hw_shapes = [(1,4,15), (3,4,15), (3,8,12), (3,12,13), (4,12,11), (6, 12, 8),
                        (4,24,8), (4,24,11), (6,24,8), (5,36,8), (6,36,8), (7,36,8), (12,24,8)]
            freqs = [330, 300, 300, 280, 270, 300, 260, 240, 220, 200, 160, 120, 120]
            folding_factor = [2] * len(hw_shapes)

        if emulator is sigma_wrap:
            # shapes for SIGMA
            hw_shapes = [(16, 70), (16, 60), (16, 54), (16, 44), (16, 36), (16, 30), (32, 10), (32, 6), (32, 2)]
            freqs = [100, 120, 150, 190, 220, 240, 210, 260, 330]
            folding_factor = [4, 4, 8, 8, 8, 8, 4, 4, 4]
            # this config is for sigma validation
            # hw_shapes = [(32, 8)]
            # freqs = [300]
            # folding_factor = [1]
    
    args = [(all_inst_dat, seq_lens, i, freq, ff, out_path) for i, ff, freq in zip(hw_shapes, folding_factor, freqs)]
    with multiprocessing.Pool(processes=7) as pool:
        pool.starmap(emulator, args)

def eval_emulator(dat: pd.DataFrame, models, tasks):
    # get latency records of required config:
    r, c, cl = 6, 12, 8
    lat_dat = {m: {t: {} for t in tasks} for m in model_names}
    for mname, taskname in product(models, tasks):
        fpath = \
            f"./res_fig/block_prune/idxmerge_window_experi/{mname}-attn-bfp20-{taskname}/spmm_rr_lat_diff_wfifo_512_r{r}_c{c}_cl{cl}.json"
        with open(fpath, "r") as f:
            sparse_lat_profile = json.load(f)
            inst_list = sparse_lat_profile.keys()
            for inst in sparse_lat_profile.keys():
                lat_dat[mname][taskname][inst] = sparse_lat_profile[inst]["avg_tops"]

    # get latency records of onchip res
    dat["total_speedup"] = dat["onchip_total_tp"] / dat["onchip_comp_tp"]
    dat["comp_speedup"] = dat["onchip_comp_tp"] / dat["dense_tp"]
    # delete the "inst_id" column
    dat_inst_mean = dat.groupby(["model", "task", "inst_id"]).mean()

    errs = []
    for m, t in product(models, tasks):
        for i in lat_dat[m][t].keys():
            onchip_res = dat_inst_mean.query(f"`model` == '{m}' and `task` == '{t}' and `inst_id` == '{i}' ")["onchip_comp_tp"]
            emu_res = lat_dat[m][t][i]
            err = abs(onchip_res-emu_res) / float(onchip_res)
            errs.append(err)

    print(f"average err: {np.mean(errs):.4f}")


def plot_roofline(hw_perf_df: pd.DataFrame, density_df: pd.DataFrame, hw_size: dict):
    # calculate hw arithmetic capability and input bandwidth for ideal roofline
    n_input_bits = (hw_size["r"] * hw_size["l"] + hw_size["c"]) * 88 / 8.0 / (1024.0 ** 3)
    max_bandwidth = n_input_bits / (1. / hw_size["freq"]* 1e-6)
    tops = hw_size["r"] * hw_size["l"] * hw_size["c"] * (20 * 3 * 2) / 1e12
    tops = tops / (1. / hw_size["freq"]* 1e-6)
    print(f"max tops: {tops:.2f}, max bandwidth: {max_bandwidth:.2f}")

    # sample some data from hw results
    hw_perf_df["total_speedup"] = hw_perf_df["onchip_total_tp"] / hw_perf_df["onchip_comp_tp"]
    hw_perf_df["comp_speedup"] = hw_perf_df["onchip_comp_tp"] / hw_perf_df["dense_tp"]

    selected_density_df = density_df[density_df["swindow"] == 12].copy()
    selected_density_df = selected_density_df.groupby(["model", "tasks", "inst id", "head idx"]).mean().reset_index()
    selected_density_df.rename(columns={"tasks": "task", "inst id" : "inst_id", "head idx": "head_id"}, inplace=True)
    dat = hw_perf_df.merge(selected_density_df, how="left", on=["model", "task", "inst_id", "head_id"])
    sampled_dat = dat.sample(500)

    # iterate through and log data points
    data_points_raw_density_sparse, data_points_effect_density_sparse = [], []
    data_points_raw_density_mild, data_points_effect_density_mild = [], []
    data_points_raw_density_dense, data_points_effect_density_dense = [], []
    for _, record in sampled_dat.iterrows():
        # devided by 2 because the density excluded the upper right matrix
        print(record)
        n_dense_blks = \
            math.ceil(math.ceil(record["seq_len"] / 20.) * math.ceil(record["seq_len"] / 3.) / 2 * record["raw head density"])
        n_ops = n_dense_blks * 128 * (20 * 3 * 2) / (1000 ** 4)
        latency = record["onchip_comp_lat"] * 1e-9
        compute_tops = n_ops / latency
        mem_usage = (n_dense_blks * 88 * 3 + n_dense_blks * 128 * 88) / 8.0 / (1024.0 ** 3)
        arith_intensity = n_ops / mem_usage
        if record["raw head density"] > 0.5:
            data_points_raw_density_dense.append((compute_tops, arith_intensity))
        elif 0.2 < record["raw head density"] < 0.5:
            data_points_raw_density_mild.append((compute_tops, arith_intensity))
        elif record["raw head density"] < 0.2:
            data_points_raw_density_sparse.append((compute_tops, arith_intensity))
        
        n_dense_blks = \
            math.ceil(math.ceil(record["seq_len"] / 20.) * math.ceil(record["seq_len"] / 3.) / 2 * record["effec head density"])
        n_ops = n_dense_blks * 128 * (20 * 3 * 2) / (1000 ** 4)
        latency = record["onchip_comp_lat"] * 1e-9
        compute_tops = n_ops / latency
        mem_usage = (n_dense_blks * 88 * 3 + n_dense_blks * 128 * 88) / 8.0 / (1024.0 ** 3)
        arith_intensity = n_ops / mem_usage
        if record["effec head density"] > 0.5:
            data_points_effect_density_dense.append((compute_tops, arith_intensity))
        elif 0.2 < record["effec head density"] < 0.5:
            data_points_effect_density_mild.append((compute_tops, arith_intensity))
        elif record["effec head density"] < 0.2:
            data_points_effect_density_sparse.append((compute_tops, arith_intensity))

    # Initialize plotter
    rl = Roofline()

    # Configure units and performance limits
    rl.set_units("TOPs", "GB")
    rl.set_ideal(max_arith=tops, max_bandwidth=max_bandwidth)  # Your hardware limits

    raw_dps = [
        ("s<50%", data_points_raw_density_dense), 
        ("50%<s<80%", data_points_raw_density_mild), 
        ("80%<s", data_points_raw_density_sparse)]
    effective_dps = [
        ("s<50%", data_points_effect_density_dense), 
        ("50%<s<80%", data_points_effect_density_mild), 
        ("80%<s", data_points_effect_density_sparse)]
    # Add data points
    # for cate in raw_dps:
    #     for dp in cate[1]:
    #         rl.add_point(dp[0], dp[1], category=f"w/o aggregation, {cate[0]}")

    for cate in effective_dps:
        for dp in cate[1]:
            rl.add_point(dp[0], dp[1], category=f"w/ aggregation, {cate[0]}")


    rl.set_palette({
        "w/o aggregation, s<50%": "#001F3F",
        "w/o aggregation, 50%<s<80%,": "#87CEEB",
        "w/o aggregation, 80%<s": "#0074D9",
        "w/ aggregation, s<50%": "#CC5500",
        "w/ aggregation, 50%<s<80%": "#FF7518",
        "w/ aggregation, 80%<s": "#FF4500",
    })
    # Generate plot
    rl.plot(pathlib.Path("./res_fig/block_prune/roofline_plot_diff_s_agg.pdf"))

def plot_thres_tops_score(onchip_res_list: dict[pd.DataFrame], scores_list: dict):
    # onchip_res_list = dict(sorted(onchip_res_list.items()))
    # get list of thresholds:
    thres_list = onchip_res_list.keys()
    # format thres_list to scores_list supported keys
    thres_list_for_scores = [f"block prune thres (x{i.split('x')[0]}/seq_lenx20)" for i in thres_list]

    # get model list and task list by onchip res
    models, tasks = [], []
    for pd_onchip_res in onchip_res_list.values():
        if models and tasks:
            models = list(set(pd_onchip_res["model"].unique().tolist()) & set(models))
            tasks = list(set(pd_onchip_res["task"].unique().tolist()) & set(tasks))
        else:
            models = pd_onchip_res["model"].unique().tolist()
            tasks = pd_onchip_res["task"].unique().tolist()

    print(f"plotter gets models:{models} and tasks:{tasks}")

    # get scores for thres, by models (scores from 1x to nx)
    avg_scores_by_models = {m:[0]*len(thres_list) for m in models}
    avg_spars_by_models = {m:[0]*len(thres_list) for m in models}
    for m in models:
        for t in tasks:
            curr_scores = \
                [scores_list[m][t][thres]["score"] for thres in thres_list_for_scores]
            avg_scores_by_models[m] = \
                (np.array(curr_scores) + np.array(avg_scores_by_models[m])).tolist()
            curr_spars = \
                [scores_list[m][t][thres]["sparsity"] for thres in thres_list_for_scores]
            avg_spars_by_models[m] = \
                (np.array(curr_spars) + np.array(avg_spars_by_models[m])).tolist()
            
        avg_scores_by_models[m] = \
            (np.array(avg_scores_by_models[m]) / len(tasks)).tolist()
        avg_spars_by_models[m] = \
            (np.array(avg_spars_by_models[m]) / len(tasks)).tolist()
    
    print(avg_scores_by_models)
    # get list of average tops over tasks, for each model, each threshold
    avg_tps_by_models = {m:[] for m in models}
    for thres, onchip_dat in onchip_res_list.items():
        model_only_dat = onchip_dat.drop(columns=["task", "inst_id"])
        model_only_dat["comp_speedup"] = \
            model_only_dat["onchip_comp_tp"] / model_only_dat["dense_tp"]        
        avg_tp_models = model_only_dat.groupby(["model"]).mean()["comp_speedup"]
        for m, tp in dict(avg_tp_models).items():
            avg_tps_by_models[m].append(tp)

    print(avg_tps_by_models)

    # plot 2d tps, score and thres
    global MODEL_PALETTE
    sns.set_theme()
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.subplots(1, 1)
    ax2 = ax.twinx()
    for m in models:
        x_data = avg_spars_by_models[m]
        y_data = avg_scores_by_models[m]
        z_data = avg_tps_by_models[m]
        s1 = ax.plot(x_data, y_data,
                            c=MODEL_PALETTE[m][0],
                            marker='s',
                            markersize = 5,
                            alpha=0.7,
                            label=m + " (score)",
                            linestyle="-",
                            linewidth=2,
                            )
        s2 = ax2.plot(x_data, z_data,
                            c=MODEL_PALETTE[m][0],
                            marker='o',
                            markersize = 5,
                            alpha=0.7,
                            linestyle="dotted",
                            linewidth=2,
                            label=m + " (speedup)")

    ax.set_xlabel('sparsity')
    ax.set_ylabel('scores\n(solid line)')
    ax.set_ylim(bottom=0)
    ax2.set_ylabel('Normalized Throughput Speedup\n(dotted line)')
    ax2.set_ylim(bottom=1, top=5.5)
    # ax2.set_ylim(bottom=1, top=10)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax2.tick_params(axis='y', which='both', left=False, right=False)

    # --- Combine legends ---
    # Get handles and labels from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # all_handles = handles1 + handles2
    # all_labels = labels1 + labels2
    # hl = sorted(zip(all_labels, all_handles), key=operator.itemgetter(0))
    all_labels = labels1 + labels2
    all_handles = handles1 + handles2
    # all_labels, all_handles = zip(*hl)

    # Create a single legend for the entire figure
    fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.512, 0.1),  fontsize=9, ncol=2)
    # fig.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(0.12, 0.5), fontsize=9)
    ax.grid(True, axis='y', linestyle='-', alpha=0.7)
    ax.grid(True, axis='x', linestyle='-', alpha=0.7)
    ax2.grid(False) 

    output_filename = './res_fig/block_prune/thres_accu_tops.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def plot_scaling(dat: pathlib.Path, designs: list[str]):
    base_out_path = pathlib.Path("./res_fig/block_prune/scaling_analysis/")

    def get_existing_color_by_label_name(ax, label_name):
        # Get the current axes and legend handles
        handles, labels = ax.get_legend_handles_labels()
        hue_colors = {}
        for handle, label in zip(handles, labels):
            color = handle.get_color()
            hue_colors[label] = color

        return hue_colors.get(label_name, None)

    raw_dat = pd.read_csv(dat / "scaling_test_all.csv")
    raw_dat_freqonly = pd.read_csv(dat / "scaling_test_all.csv")
    selected_dat = raw_dat[raw_dat["type"].isin(designs)]
    selected_dat_freqonly = raw_dat_freqonly[raw_dat_freqonly["type"].isin(designs)]

    # find out failing pnr points
    failed_pnr_points = selected_dat[selected_dat["pass P\&R"] == False]
    succeeded_pnr_points = selected_dat[selected_dat["pass P\&R"] == True]
    failed_pnr_points_freqonly = selected_dat_freqonly[selected_dat_freqonly["pass P\&R"] == False]
    succeeded_pnr_points_freqonly = selected_dat_freqonly[selected_dat_freqonly["pass P\&R"] == True]

    type_palette = palette_dict = \
        {type_name: f"C{idx+4}" for idx, type_name in enumerate(succeeded_pnr_points["type"].unique().tolist())}

    sns.set_theme()
    sns.set(font_scale=1.3)
    fig = plt.figure(figsize=(10, 6.5))
    plt.subplots_adjust(wspace=0.25)
    ax = fig.subplots(1, 2)
    markersize = 9.5
    sns.lineplot(data=succeeded_pnr_points, x="tensor block util", y="avg. throughput all", 
                 ax=ax[0], hue="type", marker="s", alpha=0.8, palette=type_palette, markersize=markersize)
    # mark failed pnr points for different designs
    for d in designs:
        failed_pnr_points_d = failed_pnr_points[failed_pnr_points["type"] == d]
        if not failed_pnr_points_d.empty:
            x_coord, y_coord = [], []
            x_coord += [succeeded_pnr_points[succeeded_pnr_points["type"] == d].iloc[0]["tensor block util"]]
            y_coord += [succeeded_pnr_points[succeeded_pnr_points["type"] == d].iloc[0]["avg. throughput all"]]
            x_coord += [failed_pnr_points_d.iloc[-1]["tensor block util"]]
            y_coord += [failed_pnr_points_d.iloc[-1]["avg. throughput all"]]

            color = get_existing_color_by_label_name(ax[0], d)
            ax[0].plot(x_coord, y_coord, marker='None', linestyle='--', color=color, alpha=0.8,
                    markeredgewidth=0, markersize=markersize+3, zorder=10, clip_on=False)
            ax[0].plot(x_coord[-1:], y_coord[-1:], marker='X', linestyle='None', color=color, alpha=0.8,
                    markeredgewidth=0, markersize=markersize+3, zorder=10, label='Designs failed to route', clip_on=False)
        
    ax[0].set_xlabel('#Tensor Blocks')
    ax[0].set_ylabel('Average Throughput (TOPS)')
    ax[0].set_xlim(xmin=0)
    ax[0].set_ylim(ymin=0)
    plt.grid(True)

    sns.lineplot(data=succeeded_pnr_points_freqonly, x="tensor block util", y="freq", 
                 ax=ax[1], hue="type", marker="s", alpha=0.8, palette=type_palette, markersize=markersize)
    # mark failed pnr points for different designs
    for d in designs:
        failed_pnr_points_d = failed_pnr_points_freqonly[failed_pnr_points_freqonly["type"] == d]
        if not failed_pnr_points_d.empty:
            x_coord, y_coord = [], []
            x_coord += [succeeded_pnr_points_freqonly[succeeded_pnr_points_freqonly["type"] == d].iloc[0]["tensor block util"]]
            y_coord += [succeeded_pnr_points_freqonly[succeeded_pnr_points_freqonly["type"] == d].iloc[0]["freq"]]
            x_coord += [failed_pnr_points_d.iloc[-1]["tensor block util"]]
            y_coord += [failed_pnr_points_d.iloc[-1]["freq"]]

            color = get_existing_color_by_label_name(ax[1], d)
            ax[1].plot(x_coord, y_coord, marker='None', linestyle='--', color=color, alpha=0.8,
                    markeredgewidth=0, markersize=markersize+3, zorder=10, clip_on=False)
            ax[1].plot(x_coord[-1:], y_coord[-1:], marker='X', linestyle='None', color=color, alpha=0.8,
                    markeredgewidth=0, markersize=markersize+3, zorder=10, label='Designs failed to route', clip_on=False)
            
    ax[1].set_xlabel('#Tensor Blocks')
    ax[1].set_ylabel('Frequency (MHz)')
    ax[1].set_xlim(xmin=0)
    ax[1].set_ylim(ymin=0)
    plt.grid(True)

    ax[0].legend().set_visible(False)
    ax[1].legend().set_visible(False)
    # enable and put legend on the top of entire figure
    # first get legends from both subplots, and keep only one for each label
    # then put them together and create a new legend for the entire figure
    handles1, labels1 = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax[1].get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    hl = zip(all_labels, all_handles)
    all_labels, all_handles = zip(*hl)
    unique_labels, unique_handles = [], []
    for label, handle in zip(all_labels, all_handles):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    # Create a single legend for the entire figure, put it on the top center
    fig.legend(unique_handles, unique_labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=len(unique_labels))

    output_filename = base_out_path / 'scaling_analysis_tops_freq_new.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

if __name__ == "__main__":
    model_names = ["chatglm2-6b-32k", "llama2-7b-chat-4k", "mixtral-8x7b"]
    task_list = ["lcc", "multifieldqa_en", "multifieldqa_zh", "passage_retrieval_zh", "qasper", "samsum", "trec", "vcsum"]

    ## compute effective sparsity and save them to csv
    # compute_unique_colidx_ratio(task_list, model_names, swindow_list)
    ## processing onchip test results and save them to csv
    # for prune_thres in ["1x", "2x", "3x", "4x"]:
    #     onchip_df_res = get_onchip_res(
    #         "/compas-old/projects/sparse-attention/micro25/onchip", 
    #         model_names, task_list, 
    #         spmm_freq=300.0, 
    #         tc_core_shape=(6, 12, 8), 
    #         threshold_postfix=prune_thres
    #     )

    ## processing onchip test results for 3x threshold for chatglm2, and 1x for the other two
    # onchip_df_res = get_onchip_res(
    #     "/compas-old/projects/sparse-attention/onchip-5hbm", 
    #     model_names, task_list, 
    #     spmm_freq=270.0, 
    #     tc_core_shape=(6, 12, 8), 
    #     threshold_postfix=""
    # )

    # getting best config for different hw impl
    # for mname, task in product(model_names, task_list):
        # for tc_budget in [68, 204, 336, 540, 624, 720, 840, 960, 1080, 1248, 1440, 1800, 2160, 2520]:
            # # design space explore for spmm
            # sweep_rr_swindow_get_tops(mname, task, tc_budget, rr2spmm_wrap, 
            #                         f"./res_fig/block_prune/spmm/{tc_budget}/{mname}-attn-bfp20-{task}")
            # # design space explore for sigma
            # sweep_rr_swindow_get_tops(mname, task, tc_budget, sigma_wrap, 
            #                         f"./res_fig/block_prune/sigma/{tc_budget}/{mname}-attn-bfp20-{task}")
            # # design space explore for naive round robin
            # sweep_rr_swindow_get_tops(mname, task, tc_budget, naive_wrap, 
            #                         f"./res_fig/block_prune/naive/{tc_budget}/{mname}-attn-bfp20-{task}")
            # # design space explore for gemm
            # sweep_rr_swindow_get_tops(mname, task, tc_budget, dense_wrap, 
            #                         f"./res_fig/block_prune/dense/{tc_budget}/{mname}-attn-bfp20-{task}")

    ## getting throughput for different hw impl
    # for mname, task in product(model_names, task_list):
        # design space explore for spmm
        # sweep_rr_swindow_get_tops(mname, task, 0.0, rr2spmm_wrap, 
        #                           f"./res_fig/block_prune/spmm/{mname}-attn-bfp20-{task}")
        # design space explore for sigma
        # sweep_rr_swindow_get_tops(mname, task, 0.0, sigma_wrap, 
        #                           f"./res_fig/block_prune/sigma/{mname}-attn-bfp20-{task}")
        # design space explore for naive round robin
        # sweep_rr_swindow_get_tops(mname, task, 0.0, naive_wrap, 
        #                           f"./res_fig/block_prune/naive/{mname}-attn-bfp20-{task}")
        # design space explore for gemm
        # sweep_rr_swindow_get_tops(mname, task, 0.0, dense_wrap, 
        #                           f"./res_fig/block_prune/dense/{mname}-attn-bfp20-{task}")

    ## processing test for gemm and save them to csv
    # gemm_path_list = [
    #     pathlib.Path(f"/compas-old/projects/sparse-attention/onchip-5hbm/synth-gemm/seq_len_{i}")
    #     for i in [32768]
    # ]
    # get_gemm_onchip_res(
    #     pathlib.Path("/compas-old/projects/sparse-attention/onchip-5hbm/synth-gemm"),
    #     model_names,
    #     task_list,
    #     300
    # )
    # get_gemm_emulated_res(
    #     gemm_path_list, 
    #     (6, 12, 8), 300, 
    #     pathlib.Path(f"/compas-old/projects/sparse-attention/onchip-5hbm/synth-gemm/emulated_res_gemms_300mhz.csv")
    # )

    ## read processed onchip results
    onchip_df_res = pd.read_csv("/compas-old/projects/sparse-attention/onchip-5hbm/onchip_res_300mhz.csv")
    # gemm_df_res = pd.read_csv("/compas-old/projects/sparse-attention/onchip-5hbm/onchip_res_gemms_300mhz.csv")
    # gemm_emulated_df_res = pd.read_csv("/compas-old/projects/sparse-attention/onchip-5hbm/synth-gemm/emulated_res_gemms_300mhz.csv")
    ## read effective sparsity data
    density_df_res = pd.read_csv("/compas-old/projects/sparse-attention/onchip-5hbm/spars-analysis-onchip-related.csv")

    # print(gemm_emulated_df_res)
    ## plotting figures
    # plot_onchip_res(onchip_df_res)
    # plot_stacked_selfattn_ops_latency(gemm_df_res, out_fig_path=pathlib.Path("./res_fig/dense_lat_ops_seqlen.pdf"))
    # plot_stacked_selfattn_ops_latency(gemm_emulated_df_res, out_fig_path=pathlib.Path("./res_fig/dense_lat_ops_seqlen_emulated.pdf"))
    # eval_emulator(onchip_df_res, model_names, task_list)
    # plot_unique_colidx_ratio_boxplot_by_task(density_df_res)
    # plot_route_ratio_by_task(density_df_res)
    # plot_unique_colidx_ratio_by_swindow(density_df_res, 8)
    # plot_redunt_colidx_ratio_distribution()
    plot_speedup_vs_sparsity(onchip_df_res, density_df_res)
    # plot_roofline(onchip_df_res, density_df_res, {"r": 6, "c": 12, "l": 8, "freq": 300})

    # # read onchip results for different thresholds
    # onchip_df_res_list = {}
    # for i in ["1x", "2x", "3x", "4x"]:
    #     onchip_df_res_list[i] = pd.read_csv(f"/compas-old/projects/sparse-attention/micro25/onchip/onchip_res_t{i}_300mhz.csv")
    # # # read longbench scores
    # with pathlib.Path("./res_fig/block_prune/formatted_data_longbench.json").open("r") as f:
    #     prune_score_eval_res = json.load(f)
    # # plot speedup vs score
    # plot_thres_tops_score(onchip_df_res_list, prune_score_eval_res)
    ## plot scaling results
    # plot_scaling(pathlib.Path("./res_fig/block_prune/scaling_analysis"), 
    #              ["Block-agg. SpMM", "SIGMA SpMM", "GEMM"])
    
    # attn_path_chatglm = inst_list[0] + ".pt"
    # src_attn = torch.load(attn_path_chatglm).numpy()
    # explore_row_features(src_attn, inst_idx=inst_idx)
    # dense_val_idx_histogram([attn_path_chatglm], 50)
    # for i in inst_list:
    #     plot_stacked_heatmap_inst([base_attn_path + i + ".pt"], \
    #                                 [base_attn_path + i + ".json"], \
    #                                 f"./res_fig/temp/hotpotqa/{i}")

    # res = block_prune_analysis(inst_list, (3, 20), 28, 32)
    # for f in inst_list:
    #     transfer_attn_to_bprune_dense_idx(f, (3, 20))

    # bprune_sweep_rrspan(inst_list, [0,1,2,4,6,8,10], 20)
    # plot_rrspan_sweep("res_fig/block_prune/sweep_rrspan.txt", 28, 32)

    # # preprocess index inputs
    # base_attn_ridx_path = "/var/services/homes/tianchu.ji/mackeson-home/HGO/proj/intel-tensor-core-matmul/sim/tb/sparse_matmul_data/midsize_ridx.npy"
    # base_attn_cidx_path = "/var/services/homes/tianchu.ji/mackeson-home/HGO/proj/intel-tensor-core-matmul/sim/tb/sparse_matmul_data/midsize_cidx.npy"
    # ridx_dat = np.load(base_attn_ridx_path)
    # cidx_dat = np.load(base_attn_cidx_path)


    # spmm_non_rr_latency_analysis(inst_list, 12, 28, 32, (nrows, ncols, chain_len))
        
    # profile_list = [f"res_fig/block_prune/spmm_rr_lat_diff_wfifo_{i}.json" for i in fdeps]
    # profile_list.append("res_fig/block_prune/spmm_worr_lat_diff.json")
    # plot_rr_spmm_lat(profile_list, 28, 32, 
    #                  f"res_fig/block_prune/spmm_rr_lat_profile_r{nrows}_c{ncols}_l{chain_len}.json")
 
    exit()
    