import numpy as np
import random
import torch
from scipy import optimize
import hw_modeling
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
import os, json, util
from itertools import product
import pandas as pd
import seaborn as sns

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

    def get_unique_idx_rate(curr_blk_list: list) -> list[tuple]:
        # redundant index count computing method
        total_idx_count = len(list(chain(*curr_blk_list)))
        sorted_idx_res = list(np.unique(list(chain(*curr_blk_list))))
        n_removed_redidx = total_idx_count - len(sorted_idx_res)
        unique_idx_rate = len(sorted_idx_res)

        return n_removed_redidx, total_idx_count

    last_ridx = idx_list[0][0] if idx_list else None
    curr_idx_blk = [[] for i in range(n_shared_chans)]
    curr_rowblk_counter = 0
    # bfp conversion and idx attaching
    for rec_idx in range(len(idx_list)):
        curr_ridx = idx_list[rec_idx][0]
        if curr_ridx > last_ridx:
            last_ridx = curr_ridx
            curr_rowblk_counter += 1

        if curr_rowblk_counter == n_shared_chans:
            curr_redidx_count, curr_total_idx_count = get_unique_idx_rate(curr_idx_blk)
            res_redidx_count_list.append(curr_redidx_count)
            res_total_idx_count_list.append(curr_total_idx_count)
            curr_idx_blk = [[] for i in range(n_shared_chans)]
            curr_rowblk_counter = 0

        curr_idx_blk[curr_rowblk_counter].append(idx_list[rec_idx][1])

        # tail
        if (rec_idx == len(idx_list)-1) and (not omit_last_iter):
            curr_redidx_count, curr_total_idx_count = get_unique_idx_rate(curr_idx_blk)
            res_redidx_count_list.append(curr_redidx_count)
            res_total_idx_count_list.append(curr_total_idx_count)
            curr_idx_blk = [[] for i in range(n_shared_chans)]
            curr_rowblk_counter = 0

    return res_redidx_count_list, res_total_idx_count_list

def get_unique_colidx_ratio_inst(
        inst_ridx_list: list[str], 
        inst_cidx_list: list[str],
        n_shared_chans: int, 
        aggre_method = None):
    
    all_inst_res_redidx_count, all_inst_res_total_idx_count = [], []
    for rinst, cinst in zip(inst_ridx_list, inst_cidx_list):
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
        for h_idx in range(len(headgrp_idx)):
            head_common_ratio_list, head_total_idx_count_list = \
                get_unique_colidx_ratio(headgrp_idx[h_idx], n_shared_chans, omit_last_iter=True)
            curr_inst_res_redidx_count += head_common_ratio_list
            curr_inst_res_total_idx_count += head_total_idx_count_list

        all_inst_res_redidx_count.append(curr_inst_res_redidx_count)
        all_inst_res_total_idx_count.append(curr_inst_res_total_idx_count)

    if aggre_method is not None:
        return aggre_method(all_inst_res_redidx_count), aggre_method(all_inst_res_total_idx_count)
    
    return all_inst_res_redidx_count, all_inst_res_total_idx_count

def compute_unique_colidx_ratio(task_list: list[str], model_names: list[str], swindow_list: list[int], method: str):
    records = {"tasks": [], "redundant index count": [], "total index count": [], "model": [], "swindow": []}
    
    for swindow in swindow_list:
        for model_name in model_names:
            for task_name in (task_list):
                base_attn_path = f"/chronos_data/tji/.huggingface_cache/transformers/bprune-data/isca/{model_name}-attn-bfp20-{task_name}/"
                inst_rlist = util.get_pts_under_dir(base_attn_path, postfix="npy", datatype="ridx")
                inst_clist = util.get_pts_under_dir(base_attn_path, postfix="npy", datatype="cidx")
                task_res_redidx_count, task_res_total_idx_count = get_unique_colidx_ratio_inst(inst_rlist, inst_clist, int(swindow), lambda x: list(chain(*x)))
                # make sure the length of task_res_redidx_count and task_res_total_idx_count are the same
                assert len(task_res_redidx_count) == len(task_res_total_idx_count)
                
                records["tasks"] += [task_name] * len(task_res_redidx_count)
                records["redundant index count"] += task_res_redidx_count
                records["total index count"] += task_res_total_idx_count
                records["model"] += [model_name] * len(task_res_redidx_count)
                records["swindow"] += [swindow] * len(task_res_redidx_count)

    records = pd.DataFrame(records)
    # store records as pandas dataframe
    records.to_csv(f"/chronos_data/tji/.huggingface_cache/transformers/bprune-data/isca/redidx-count.csv", index=False)

def plot_unique_colidx_ratio_boxplot_by_task():
    # read records from csv
    records = pd.read_csv(f"/chronos_data/tji/.huggingface_cache/transformers/bprune-data/isca/redidx-count.csv")
    # select records that has swindow = 12
    records = records[records['swindow'] == 16]
    # add a new column "removable redundant index ratio"
    records["removable redundant index ratio"] = records["redundant index count"] / records["total index count"]

    sns.set_theme(rc={'figure.figsize': (18, 4.5)}, font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    lm = sns.violinplot(
        data=records, 
        x = "tasks", 
        y = "removable redundant index ratio", 
        inner_kws=dict(color=".8"),
        hue = "model",
        legend=True,
        palette={"llama2-7b-chat-4k": "C1", "mixtral-8x7b": "C2", "chatglm2-6b-32k": "C0"}
        )
    lm.set(ylim=(0.0, 1.0))
    # lm.get_legend().set_title(None)
    
    ax = lm.axes
    ax.legend(title=None)
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        # labels.append(textwrap.fill(text, width=10, break_on_hyphens=True))
        labels.append(text)
    ax.set_xticklabels(labels, rotation=10)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(f"./res_fig/block_prune/isca/redidx-ratio-bplot-omit-last-iter.pdf")

def plot_unique_colidx_ratio_by_swindow():
    records = pd.read_csv(f"/chronos_data/tji/.huggingface_cache/transformers/bprune-data/isca/redidx-count.csv")

    # group the records by tasks, model and swindow
    records = records[records['swindow'] < 64]
    grouped_records = records.groupby(['model', 'swindow'])['redundant index count'].mean().reset_index()
    
    # plot the grouped records as scatter plot with line connecting the points,
    # separate lines for each model and tasks, for different models use different colors
    # use different line style for different tasks
    # use swindow on x-axis with log scale, unique index ratio on y-axis
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(12, 8))
    lstyles = {"lcc": "-", "multifieldqa_en": "--", "multifieldqa_zh": "-.", "passage_retrieval_zh": ":", "qasper": "-", "samsum": "--", "trec": "-.", "vcsum": ":"}
    for model_idx, model in enumerate(grouped_records['model'].unique()):
        # plot 
        color_table = {"llama2-7b-chat-4k": "C1", "mixtral-8x7b": "C2", "chatglm2-6b-32k": "C0"}
        plt.plot(grouped_records[grouped_records['model'] == model]['swindow'], 
                    grouped_records[grouped_records['model'] == model]['redundant index count'], 
                    marker='s', markersize=10, 
                    label=f"{model}", linewidth=5, color=color_table[model])

    plt.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    plt.xscale('log')
    plt.xlim(xmin=2)
    plt.ylim(ymin=0)
    plt.legend()
    plt.xlabel("row group size (R)")
    plt.ylabel("average #removable redundant index")
    plt.tight_layout()
    plt.savefig("./res_fig/block_prune/isca/redidx-count-by-swindow-zoomin.pdf")
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

    ax.set_ylabel('latency (sec)')
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xticks(x_labels)
    ax.set_xticklabels(x_labels)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    ax.set_xlabel('layer')
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

    ax.set_ylabel('throughput (TOPs)')
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    ax.set_xticks(x_labels)
    ax.set_xticklabels(x_labels)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    ax.set_xlabel('layer')
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


def get_onchip_res(dat_path, models_name, tasks_name, spmm_freq=300.0, tc_core_shape=(6, 12, 8)):
    if dat_path[-1] != "/":
        dat_path += "/"

    spmm_cycle_delay = 1./spmm_freq * 1000.

    # create pandas dataframe with columns: model,  task, onchip_lat, onchip_tp, seq_len
    df = pd.DataFrame(columns=["model", "task", "inst_id", "head_id", "onchip_lat", "onchip_tp", "seq_len", "dense_lat", "dense_tp"])

    # get the onchip res from the dat_path
    for model, task in product(models_name, tasks_name):
        model_task_path = os.path.join(dat_path + f"{model}-attn-bfp20-{task}/")
        # get subdirs under model_task_path
        subdirs = [os.path.join(model_task_path, d) for d in os.listdir(model_task_path) if os.path.isdir(os.path.join(model_task_path, d))]
        for subdir in subdirs:
            # get the inst_id from the subdir name
            inst_id = subdir.split("/")[-1]
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

                if hwconfig.get("lat_counter_res", None) is not None:
                    # get the onchip_lat and onchip_tp from the hwconfig
                    latency_res = hwconfig["lat_counter_res"] * spmm_cycle_delay
                    spmm_rr_flops = float(total_ops) / (float(latency_res) * 1e-9) / 1e12

                    # append the onchip_lat, onchip_tp, seq_len to the df
                    df.loc[len(df)] = {
                                        "model": model, 
                                        "task": task, 
                                        "inst_id": inst_id, 
                                        "head_id": hid,
                                        "onchip_lat": latency_res, 
                                        "onchip_tp": spmm_rr_flops, 
                                        "seq_len": seq_len, 
                                        "dense_lat": dense_res.total_lat, 
                                        "dense_tp": dense_res.total_flops
                    }

    df.to_csv("./res_fig/block_prune/onchip_res.csv")
    return df

def plot_onchip_res(dat: pd.DataFrame):
    dat["speedup"] = dat["onchip_tp"] / dat["dense_tp"]
    # delete the "inst_id" column
    dat = dat.drop(columns=["inst_id"])
    # calculate the mean of onchip_tp of different inst_id for the same model and task
    dat_mean = dat.groupby(["model", "task"], group_keys=True).mean()

    # set the plot size to be 16,  6 for seaborn barplot
    sns.set(rc={'figure.figsize':(16, 6)}, font_scale=1.3)
    
    print(dat_mean)

    # plot the mean of onchip_tp and dense_tp in a same bar chart, on the x-axis group tasks 
    # with the same model without gaps, and add a gap between each model    
    model_seq = ["chatglm2-6b-32k", "llama2-7b-chat-4k", "mixtral-8x7b"]
    task_seq = ["lcc", "multifieldqa_en", "multifieldqa_zh", "passage_retrieval_zh", "qasper", "samsum", "trec", "vcsum"]
    bplot = sns.barplot(dat_mean, x="task", y="speedup", hue="model",
                        width=0.4, palette={"llama2-7b-chat-4k": "C1", "mixtral-8x7b": "C2", "chatglm2-6b-32k": "C0"})

    # for cont_idx, i in enumerate(bplot.containers):
    #     labels = [int(dat_mean.loc[model_seq[cont_idx], task_seq[task_idx]]["seq_len"]) for task_idx in range(len(task_seq))]
    #     bplot.bar_label(i, labels, fmt='%d')
        
    bplot.set_ylabel("throughput speedup")
    bplot.set_ylim(ymin=0, ymax=7)

    # set the legend to be outside the plot, and one line for each legend
    bplot.get_figure().axes[0].legend(loc="upper center", ncol=3)
    bplot.get_figure().tight_layout()
    bplot.get_figure().savefig("./res_fig/block_prune/onchip_res_speedup.pdf")

if __name__ == "__main__":
    # hardware config
    # hw_shapes = get_tccore_config((range(9, 18, 1)), 864, 11*16)
    # print(hw_shapes)
    hw_shapes = [
    ]

    # inst_idx = 4
    model_names = ["llama2-7b-chat-4k", "mixtral-8x7b", "chatglm2-6b-32k"]
    task_list = ["lcc", "multifieldqa_en", "multifieldqa_zh", "passage_retrieval_zh", "qasper", "samsum", "trec", "vcsum"]
    # task_list = ["lcc", "multifieldqa_en", "multifieldqa_zh", "passage_retrieval_zh", "qasper"]

    # plot_unique_colidx_ratio_boxplot_by_task(task_list, model_names)
    # generate a list with power of 2, from 4 to 5000
    # swindow_list = [2**i for i in range(2, 12)]
    # swindow_list = list(sorted(swindow_list + [12]))
    # compute_unique_colidx_ratio(task_list, model_names, swindow_list, method="unique")
    # plot_unique_colidx_ratio_boxplot_by_task()
    plot_unique_colidx_ratio_by_swindow()
    exit()
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
    
    # # extract one head
    # headgrp_ridx, headgrp_cidx = [], []
    # curr_head_ridx, curr_head_cidx = [], []
    # for ridx, cidx in zip(ridx_dat, cidx_dat):
    #     if ridx == -1 and cidx == -1:
    #         headgrp_ridx.append(curr_head_ridx.copy())
    #         headgrp_cidx.append(curr_head_cidx.copy())
    #         curr_head_ridx, curr_head_cidx = [], []
    #     else:
    #         curr_head_ridx.append(ridx)
    #         curr_head_cidx.append(cidx)

    # # fetch a head
    # idx_dat = []
    # print(f"get {len(headgrp_ridx)} heads in total")
    # for hidx in range(len(headgrp_ridx)):
    #     src_ridx, src_cidx = headgrp_ridx[hidx], headgrp_cidx[hidx] 
    #     idx_dat.append([(r, c) for r, c in zip(src_ridx, src_cidx)])

    # fdeps = [200]
    # def rr2spmm_wrap(hw_shape): 
    #     rr2spmm_fifo_latency_overlap_analysis(
    #         [idx_dat], [4480], hw_shape[1], hw_shape, 200, out_buff_depth=1024)
    
    # import multiprocessing
    # with multiprocessing.Pool() as pool:
    #     pool.map(rr2spmm_wrap, hw_shapes)

    # spmm_non_rr_latency_analysis(inst_list, 12, 28, 32, (nrows, ncols, chain_len))
        
    # profile_list = [f"res_fig/block_prune/spmm_rr_lat_diff_wfifo_{i}.json" for i in fdeps]
    # profile_list.append("res_fig/block_prune/spmm_worr_lat_diff.json")
    # plot_rr_spmm_lat(profile_list, 28, 32, 
    #                  f"res_fig/block_prune/spmm_rr_lat_profile_r{nrows}_c{ncols}_l{chain_len}.json")


    # onchip_df_res = get_onchip_res(
    #     "/compas-old/projects/sparse-attention/onchip", 
    #     ["chatglm2-6b-32k", "llama2-7b-chat-4k", "mixtral-8x7b"], 
    #     ["lcc", "multifieldqa_en", "multifieldqa_zh", "passage_retrieval_zh", "qasper", "samsum", "trec", "vcsum"], 
    #     spmm_freq=300.0, 
    #     tc_core_shape=(6, 12, 8)
    #     )
    onchip_df_res = pd.read_csv("./res_fig/block_prune/onchip_res.csv")
    plot_onchip_res(onchip_df_res)

    exit()
    
