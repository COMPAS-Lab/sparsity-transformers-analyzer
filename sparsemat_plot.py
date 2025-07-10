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
    rr2spmm_fifo_latency_overlap_analysis)
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

model_palette = {
    "chatglm2-6b-32k": ("#1f77b4", "#84cdff"), 
    "llama2-7b-chat-4k": ("#ff7f0e", "#ffc85f"), 
    "mixtral-8x7b": ("#2ca02c", "#76ce55")
}

def sublist_creator(lst, n):
    lists = [[] for _ in range(n)]
    totals = [(0, i) for i in range(n)]
    heapq.heapify(totals)
    for value in lst:
        total, index = heapq.heappop(totals)
        lists[index].append(value)
        heapq.heappush(totals, (total + value, index))
    return lists

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

    global model_palette

    # boxplot with errorbars 
    sns.set(rc={'figure.figsize': (21, 6)}, font_scale=1.3)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.2)

    lm = sns.boxplot(
        data=selected_records, 
        x = "tasks", 
        y = "effective sparsity",
        hue = "model",
        legend=True,
        palette={k:v[1] for k, v in zip(model_palette.keys(), model_palette.values())},
        gap=.1,
        flierprops={"alpha": 0.5},
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
        dodge=0.53,
        linestyle="none",
        palette={k:v[0] for k, v in zip(model_palette.keys(), model_palette.values())},
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
        palette={k:v[1] for k, v in zip(model_palette.keys(), model_palette.values())},
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
        dodge=0.53,
        linestyle="none",
        palette={k:v[0] for k, v in zip(model_palette.keys(), model_palette.values())},
    )

    axes[1].set(ylim=(-0.01, 1.01))
    axes[1].set_ylabel("Pre-aggregation sparsity")
    axes[1].set_xlabel("Tasks")
    errorbar_ax.set_xticklabels([])
    errorbar_ax.set_xlabel("")
    errorbar_ax.tick_params(top=False) 
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
        palette={k:v[1] for k, v in zip(model_palette.keys(), model_palette.values())},
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
        palette={k:v[0] for k, v in zip(model_palette.keys(), model_palette.values())},
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
        palette={k:v[1] for k, v in zip(model_palette.keys(), model_palette.values())},
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
        palette={k:v[0] for k, v in zip(model_palette.keys(), model_palette.values())},
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

    global model_palette

    sns.set_theme(rc={'figure.figsize': (18, 5)}, font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    lm = sns.boxplot(
        data=selected_records, 
        x = "tasks", 
        y = "mean_route_ratio",
        hue = "model",
        legend=True,
        palette={k:v[1] for k, v in zip(model_palette.keys(), model_palette.values())},
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
        palette={k:v[0] for k, v in zip(model_palette.keys(), model_palette.values())},
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
    global model_palette
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
            "model", 
            "task", 
            "inst_id", 
            "seq_len", 
            "q_lat",
            "kv_lat",
            "qkT_lat",
            "aV_lat",
            "o_lat"
        ]
    )

    # get the onchip res from the dat_path
    for model, task in product(models_name, tasks_name):
        model_task_path = dat_path / f"{model}-attn-bfp20-{task}"

        # get subdirs under model_task_path
        subdirs = [model_task_path / d for d in os.listdir(model_task_path) if (model_task_path / d).is_dir()]
        for subdir in subdirs:
            # get the inst_id from the subdir name
            inst_id = subdir.name
            # temporarily skip the random records
            if inst_id[0:9] != "iiSeqInst":
                continue 
            # get the seq_len from the subdir's "inst_profile.json" file
            with (subdir / "inst_profile.json").open("r") as f:
                inst_profile = json.load(f)
                seq_len = inst_profile["seq_len"]

            q_lat = get_gemm_synth_lat(subdir / "qkv")
            kv_lat = get_gemm_synth_lat(subdir / "qkv", force_n_b_blks=1)
            qkT_lat = get_gemm_synth_lat(subdir / "qkT")
            aV_lat = get_gemm_synth_lat(subdir / "aV")

            df.loc[len(df)] = {
                                "model": model, 
                                "task": task, 
                                "inst_id": inst_id, 
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
        spmm_dat: pd.DataFrame,
        n_heads = 32,
        ):
    # analyze latency breakdown of a single self-attn
    gemm_dat["qkv_lat"] = gemm_dat["q_lat"] + gemm_dat["kv_lat"] * 2
    gemm_dat["qkT_lat"] = gemm_dat["qkT_lat"] * n_heads
    gemm_dat["aV_lat"] = gemm_dat["aV_lat"] * n_heads
    gemm_dat["linear_lat"] = gemm_dat["qkv_lat"] + gemm_dat["o_lat"]
    gemm_dat_inst_mean = gemm_dat.drop(columns=["inst_id"]).groupby(["model", "task"]).mean().reset_index()

    print(gemm_dat_inst_mean.to_markdown())

    # spmm_dat_inst_mean = spmm_dat.drop(columns=["inst_id"]).groupby(["model", "task"]).mean().reset_index()
    # spmm_dat["avg_aV"] = spmm_dat["onchip_total_lat"] 

    latency_types = ["qkT_lat", "aV_lat", "linear_lat"]
    # Melt the DataFrame to long format as before
    df_melted = gemm_dat_inst_mean.melt(
        id_vars=["model", "task"],
        value_vars=latency_types,
        var_name="latency_type",
        value_name="latency_value"
    )

    # Ensure consistent order for tasks and models for plotting
    unique_tasks = gemm_dat_inst_mean['task'].unique()
    unique_models = gemm_dat_inst_mean['model'].unique()

    # --- Define X-axis positions and labels ---
    bar_width = 0.8 # Width of each individual model's stacked bar
    task_margin = 1.0 # Margin between different tasks
    model_spacing = 0.0 # No margin between models within the same task

    x_positions = []
    x_labels = []
    task_x_centers = [] # For placing task labels
    current_x = 0

    for task in unique_tasks:
        models_in_task = df_melted[df_melted['task'] == task]['model'].unique()
        num_models_in_task = len(models_in_task)

        task_start_x = current_x
        for i, model in enumerate(models_in_task):
            x_positions.append(current_x)
            x_labels.append(model) # Label with model name
            current_x += bar_width + model_spacing # Move to the next model position
        
        # Calculate center for task label
        task_end_x = current_x - model_spacing # End of the last model bar
        task_x_centers.append((task_start_x + task_end_x - bar_width) / 2 + bar_width/2) # Center of the task group
        
        current_x += task_margin # Add margin after the last model of the current task

    # --- Color and Hatch Mappings ---
    # Colors for models
    global model_palette

    # Hatches for latency types
    latency_hatches = {
        "qkT_lat": "xx",
        "aV_lat": "o",
        "linear_lat": "//"
    }

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group data by (task, model) for plotting
    grouped_data = df_melted.groupby(['task', 'model'])

    # Iterate through each bar position to draw stacked segments
    for i, (x_pos, model_label) in enumerate(zip(x_positions, x_labels)):
        task = gemm_dat_inst_mean.loc[gemm_dat_inst_mean['model'] == model_label, 'task'].iloc[0]
        
        # Get the data for the current model within its task
        current_model_data = df_melted[(df_melted['task'] == task) & (df_melted['model'] == model_label)]
        
        bottom_value = 0
        for lat_type in latency_types:
            latency_val = current_model_data[current_model_data['latency_type'] == lat_type]['latency_value'].sum()
            if not pd.isna(latency_val) and latency_val > 0: # Only plot if there's a value
                ax.bar(
                    x_pos,
                    latency_val,
                    width=bar_width,
                    bottom=bottom_value,
                    color=model_palette[model_label],
                    hatch=latency_hatches[lat_type],
                    edgecolor='black', # Add black edge for better visibility of hatches
                    linewidth=0.5
                )
                bottom_value += latency_val

    # --- Customizing X-axis ---
    task_label_positions = []
    for t_id in range(len(unique_tasks)):
        task_label_positions.append(x_positions[t_id * len(model_names) + 1])
    ax.set_xticks(task_label_positions)
    ax.set_xticklabels(unique_tasks, rotation=15, ha='right')
    ax.set_xlabel("Tasks", fontsize=12)
    ax.set_ylabel("Latency (ns)", fontsize=12)

    # Add horizontal lines or text for task separation/labels
    # We'll use custom text labels for tasks
    # Get the unique tasks in order
    unique_tasks_df = df_melted[['task', 'model']].drop_duplicates().sort_values(by=['task', 'model'])

    task_group_boundaries = []
    current_task = None
    for i, (idx, row) in enumerate(unique_tasks_df.iterrows()):
        if row['task'] != current_task:
            if current_task is not None:
                task_group_boundaries.append(i - 0.5) # Mark end of previous group
            task_group_boundaries.append(i - 0.5) # Mark start of new group
            current_task = row['task']
    task_group_boundaries.append(len(x_positions) - 0.5) # End of the last group

    ax.tick_params(axis='x', which='minor', bottom=False) # Remove minor ticks

    # Adjust primary x-axis limits to accommodate for the last bar and potential margin
    ax.set_xlim(-bar_width/2, current_x - task_margin + bar_width/2) # Adjust limits to frame bars nicely

    # --- Create Custom Legends ---
    # Legend for Latency Types (Hatches)
    hatch_patches = [
        Patch(facecolor='white', edgecolor='black', hatch=latency_hatches[lt], label=lt.replace("_lat", ""))
        for lt in latency_types
    ]
    hatch_legend = ax.legend(handles=hatch_patches, title="component", ncol=len(hatch_patches),
                            bbox_to_anchor=(0.2, 1.12), loc='upper center', borderaxespad=0.)

    # Legend for Models (Colors)
    color_patches = [
        Patch(facecolor=model_palette[model][0], edgecolor='black', label=model)
        for model in unique_models
    ]
    color_legend = ax.legend(handles=color_patches, title="Model", ncol=len(color_patches),
                            bbox_to_anchor=(0.66, 1.12), loc='upper center', borderaxespad=0.)

    ax.add_artist(hatch_legend) # Add the first legend back

    plt.tight_layout() # Adjust layout to make space for legends
    plt.savefig("./res_fig/dense_lat_ops.pdf")


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

    global model_palette
    model_seq = ["chatglm2-6b-32k", "llama2-7b-chat-4k", "mixtral-8x7b"]
    task_seq = ["lcc", "multifieldqa_en", "multifieldqa_zh", "passage_retrieval_zh", "qasper", "samsum", "trec", "vcsum"]

    # plot normalized speed up
    # set the plot size to be 16,  6 for seaborn barplot
    sns.set(rc={'figure.figsize':(18, 4.5)}, font_scale=1.3)
    fig, axes = plt.subplots(1, 1)
    curr_model_palette = {k:model_palette[k][0] for k in model_palette.keys()}
    sns.barplot(dat_mean, x="task", y="comp_speedup", hue="model", width=0.4, 
                        palette=curr_model_palette, ax=axes)    
    labels = [label.get_text() for label in axes.get_xticklabels()]
    axes.set_xticklabels(labels, rotation=10)
    axes.set_ylabel("Average normalized throughput\n of heads")
    axes.set_xlabel("Tasks")
    axes.set_ylim(ymin=0, ymax=5.5)
    axes.legend(loc="upper center", ncol=3)
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

    # plot speedup vs seq len
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
            total_lats = sublist_creator(inst_total_lats, n_queus)
            total_lats_dense = sublist_creator(inst_total_lats_dense, n_queus)

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

    global model_palette
    curr_model_palette = {k:model_palette[k][0] for k in model_palette.keys()}
    
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


def rr2spmm_wrap(all_inst_dat, seq_lens, hw_shape, out_path): 
    rr2spmm_fifo_latency_overlap_analysis(
        all_inst_dat, seq_lens, 
        hw_shape[1], hw_shape, 
        512, 
        out_buff_depth=1024,
        out_json_basepath=out_path)

def sweep_rr_swindow_get_tops(model_name, task_name):
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
    out_path = f"./res_fig/block_prune/idxmerge_window_experi.new/{model_name}-attn-bfp20-{task_name}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    c_list = [4, 8, 12, 24, 36]
    hw_shapes = []
    prereq = lambda x1,x2: (x2 >= 8) and ((math.ceil(128./x1) >= 3*x2) or abs(math.ceil(128./x1) - 3*x2) < 20)
    for c in c_list:
        r_and_cl_pairs = find_positive_integer_pairs(720//c, prereq)
        hw_shape = [(r_and_cl[0], c, r_and_cl[1]) for r_and_cl in r_and_cl_pairs]
        hw_shapes += hw_shape

    # selected shapes:
    # [(18, 4, 8), (6, 8, 13), (9, 8, 8), (3, 12, 18), (4, 12, 13), (5, 12, 10), (6, 12, 8), (1, 24, 28), (2, 24, 13), (3, 24, 8), (1, 36, 18), (2, 36, 8)]
    # new shapes:
    # [(6, 8, 13), (3, 12, 18), (4, 12, 13), (5, 12, 10), (1, 24, 28), (2, 24, 13), (1, 36, 18)]
    finished_hw_shapes = [(18, 4, 8), (9, 8, 8), (6, 12, 8), (3, 24, 8), (2, 36, 8),
                 (8, 4, 18), (8, 8, 9), (8, 12, 6), (8, 24, 3), (8, 36, 2)]
    
    new_hw_shapes = [item for item in hw_shapes if item not in finished_hw_shapes]
    print(new_hw_shapes)
    args = [(all_inst_dat, seq_lens, i, out_path) for i in new_hw_shapes]    
    with multiprocessing.Pool(processes=10) as pool:
        pool.starmap(rr2spmm_wrap, args)

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
    thres_list = onchip_df_res_list.keys()
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
    global model_palette
    sns.set_theme()
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.subplots(1, 1)
    ax2 = ax.twinx()
    for m in models:
        x_data = avg_spars_by_models[m]
        y_data = avg_scores_by_models[m]
        z_data = avg_tps_by_models[m]
        s1 = ax.plot(x_data, y_data,
                            c=model_palette[m][0],
                            marker='s',
                            markersize = 5,
                            alpha=0.7,
                            label=m + " (score)",
                            linestyle="-",
                            linewidth=2,
                            )
        s2 = ax2.plot(x_data, z_data,
                            c=model_palette[m][0],
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

    output_filename = './res_fig/block_prune/thres_accu_tops_2d.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    plt.clf()
    plt.close(fig)
 

if __name__ == "__main__":
    # hardware config
    # hw_shapes = get_tccore_config((range(9, 18, 1)), 864, 11*16)

    # inst_idx = 4
    model_names = ["chatglm2-6b-32k", "llama2-7b-chat-4k", "mixtral-8x7b"]
    # model_names = ["chatglm2-6b-32k"]
    task_list = ["lcc", "multifieldqa_en", "multifieldqa_zh", "passage_retrieval_zh", "qasper", "samsum", "trec", "vcsum"]

    # compute_unique_colidx_ratio(task_list, model_names, swindow_list)
    # onchip_df_res = get_onchip_res(
    #     "/compas-old/projects/sparse-attention/micro25/onchip", 
    #     # "/compas-old/projects/sparse-attention/onchip-5hbm", 
    #     model_names, task_list, 
    #     spmm_freq=300.0, 
    #     tc_core_shape=(6, 12, 8), 
    #     threshold_postfix="4x"
    # )
    # for mname, task in product(model_names, task_list):
    #      sweep_rr_swindow_get_tops(mname, task)
    # get_gemm_onchip_res(
    #     pathlib.Path("/compas-old/projects/sparse-attention/onchip-5hbm"),
    #     model_names,
    #     task_list,
    #     300
    # )

    # onchip_df_res = pd.read_csv("/compas-old/projects/sparse-attention/onchip-5hbm/onchip_res_300mhz.csv.old")
    # density_df_res = pd.read_csv("/compas-old/projects/sparse-attention/onchip-5hbm/spars-analysis-onchip-related.csv")
    gemm_df_res = pd.read_csv("/compas-old/projects/sparse-attention/onchip-5hbm/onchip_res_gemms_300mhz.csv")

    # plot_onchip_res(onchip_df_res)
    plot_stacked_selfattn_ops_latency(gemm_df_res, None)
    # eval_emulator(onchip_df_res, model_names, task_list)
    # plot_unique_colidx_ratio_boxplot_by_task(density_df_res)
    # plot_route_ratio_by_task(density_df_res)
    # plot_unique_colidx_ratio_by_swindow(density_df_res, 8)
    # plot_redunt_colidx_ratio_distribution()
    # plot_speedup_vs_sparsity(onchip_df_res, density_df_res)
    # plot_roofline(onchip_df_res, density_df_res, {"r": 6, "c": 12, "l": 8, "freq": 300})

    # onchip_df_res_list = {}
    # for i in ["1x", "2x", "3x", "4x"]:
    #     onchip_df_res_list[i] = pd.read_csv(f"/compas-old/projects/sparse-attention/micro25/onchip/onchip_res_t{i}_300mhz.csv")
    # with pathlib.Path("./res_fig/block_prune/formatted_data_longbench.json").open("r") as f:
    #     prune_score_eval_res = json.load(f)
    # plot_thres_tops_score(onchip_df_res_list, prune_score_eval_res)
    
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
    
