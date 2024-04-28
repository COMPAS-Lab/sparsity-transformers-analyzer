import numpy as np
import random
import torch
from scipy import optimize
from sparsemat_hw_modeling import (
    distance_of_dense_vals_per_row, 
    transfer_attn_to_bprune_dense_idx,
    bprune_sweep_rrspan,
    rr2spmm_latency_overlap_analysis)
import matplotlib
from matplotlib import pyplot as plt
import os, json, util
from tqdm import tqdm

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


def plot_rr_spmm_latdiff(dat_file, n_layers, n_heads):
    with open(dat_file, "r") as f:
        dat = json.load(f)

    for l in range(n_layers):
        fig, ax = plt.subplots(4, 8, figsize=(20, 10))
        for h in range(n_heads):
            for inst_idx, inst in enumerate(dat[f"l{l}h{h}"]):
                ax[h // 8][h % 8].plot(
                    list(range(len(inst))), 
                    inst, 
                    color=f"C{inst_idx}", 
                    alpha=0.3,
                    marker='.', 
                    label=f"i{inst_idx}:{np.mean(inst):.1f}")
                
            ax[h // 8][h % 8].set_xlim(xmin=0)
            ax[h // 8][h % 8].legend(loc="upper left")
    
        fig.tight_layout(rect=(0.03, 0.03, 1, 1))
        fig.supxlabel("iteration")
        fig.supylabel("SpMM lat - redundancy removal lat (ns)")
        fig.savefig(f"res_fig/block_prune/spmm_rr_lat_diff/l{l}.pdf")
        fig.clf()
    

if __name__ == "__main__":
    # inst_idx = 4
    base_attn_path = f"/chronos_data/tji/.huggingface_cache/transformers/chatglm2-6b-32k-attn-bfp20-1e-3-hotpotqa-bprune-scaled/"
    # list all insts
    inst_list = util.get_pts_under_dir(base_attn_path, "npy")
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

    rr2spmm_latency_overlap_analysis(inst_list[0:4], 12, n_layers=28, n_heads=32, tc_core_shape=(4, 12, 12))
    plot_rr_spmm_latdiff("res_fig/block_prune/spmm_rr_lat_diff.json", 28, 32)
    exit()

    with open("./res_fig/block_prune/bprune_row_density_profile/bpruning_hotpotqa_bthres.json", "w") as f:
        json.dump(res, f)

    elem_spars = []
    for i in [base_attn_path + p + ".json" for p in inst_list]:
        with open(i) as fp:
            res = json.load(fp)
            elem_spars.append(res["spar_mean"])

    all_bsparse = None
    with open("./res_fig/block_prune/bprune_row_density_profile/bpruning_hotpotqa_bthres.json") as fp:
        res = json.load(fp)
        all_bsparse = list(res.values())

    print(f"element-wise sparsity: {np.mean(elem_spars)}")
    print(f"block sparsity: {np.mean(all_bsparse)}")
    
