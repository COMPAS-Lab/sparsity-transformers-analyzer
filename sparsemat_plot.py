import numpy as np
import random
import torch
from scipy import optimize
from sparsemat_hw_modeling import (
    distance_of_dense_vals_per_row, 
    compare_topk_focus_inst)
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
        num_layers = 24
        for l in range(num_layers):
            with open(os.path.join(json_path, f"src_data_layer_{l}.json"), "r") as fp:
                lat_dat = json.load(fp)
                dat_delay[json_path].append(lat_dat["compress blue"]["attxv"]["blocked prune sparse"][0])
                dat_tp[json_path].append(lat_dat["compress blue"]["attxv"]["blocked prune sparse"][1])

    # plot latency vs layers
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    x_labels = np.arange(1, num_layers+1, 1)
    for i, p in enumerate(json_path_lists):
        #compute bar idx
        if len(json_path_lists) % 2 == 0:
            x_pos = x_labels + (i - len(json_path_lists) / 2) * 0.2 + 0.1
        else:
            x_pos = x_labels + (i - len(json_path_lists - 1) / 2) * 0.2

        ax.bar(x_pos, dat_delay[p], 
                width=0.2, color=f"C{i}", linewidth=1, label=label_list[i])

    ax.set_ylabel('latency (sec)')
    ax.set_ylim(ymin=0)
    ax.set_xticks(x_labels)
    ax.set_xticklabels(x_labels)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    ax.set_xlabel('layer')
    ax.legend()
    fig.tight_layout()
    fig.savefig(res_path + "/res_delay.pdf")
    fig.clf()


    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    x_labels = np.arange(1, num_layers+1, 1)
    for i, p in enumerate(json_path_lists):
        #compute bar idx
        if len(json_path_lists) % 2 == 0:
            x_pos = x_labels + (i - len(json_path_lists) / 2) * 0.2 + 0.1
        else:
            x_pos = x_labels + (i - len(json_path_lists - 1) / 2) * 0.2

        ax.bar(x_pos, dat_tp[p], 
                width=0.2, color=f"C{i}", linewidth=1, label=label_list[i])

    ax.set_ylabel('throughput (TOPs)')
    ax.set_ylim(ymin=0)
    ax.set_xticks(x_labels)
    ax.set_xticklabels(x_labels)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    ax.set_xlabel('layer')
    ax.legend()
    fig.tight_layout()
    fig.savefig(res_path + "/res_tp.pdf")
    fig.clf()


if __name__ == "__main__":
    # plot_dat_list = [
    #     "./res_fig/block_prune/bprune_sweep_chainlen/10",
    #     "./res_fig/block_prune/hotpotqa_bprune_retain_replicated",
    # ]
    # label_list = ["reduce rep", "keep rep"]
    # plot_hw_perf(plot_dat_list, label_list, "./res_fig/block_prune/hotpotqa_bprune_retain_replicated")
    # exit()

    # inst_idx = 4
    base_attn_path = f"/chronos_data/tji/.huggingface_cache/transformers/chatglm2-6b-32k-attn-bfp20-hotpotqa-bprune-topkmax/"
    # list all insts
    inst_list = util.get_pts_under_dir(base_attn_path)
    inst_list = inst_list[0:2]
    # attn_path_chatglm = inst_list[0] + ".pt"
    # src_attn = torch.load(attn_path_chatglm).numpy()
    # explore_row_features(src_attn, inst_idx=inst_idx)
    # dense_val_idx_histogram([attn_path_chatglm], 50)
    # for i in inst_list:
    #     plot_stacked_heatmap_inst([base_attn_path + i + ".pt"], \
    #                                 [base_attn_path + i + ".json"], \
    #                                 f"./res_fig/temp/hotpotqa/{i}")

    # res = block_prune_analysis([base_attn_path + p + ".pt" for p in inst_list], (3, 20), 28, 32)

    topksum_path = f"/chronos_data/tji/.huggingface_cache/transformers/chatglm2-6b-32k-attn-bfp20-hotpotqa-bprune-topksum/"
    topkmax_path = f"/chronos_data/tji/.huggingface_cache/transformers/chatglm2-6b-32k-attn-bfp20-hotpotqa-bprune-topkmax/"
    # map same seq length instances
    topksum_pts = util.get_pts_under_dir(topksum_path)
    topkmax_pts = util.get_pts_under_dir(topkmax_path)
    mapped_same_insts = []
    if len(topksum_pts) == len(topkmax_pts):
        while topksum_pts:
            tsum_size = os.stat(topksum_pts[0]).st_size
            for curr_tmax_idx in range(len(topkmax_pts)):
                tmax_size = os.stat(topkmax_pts[curr_tmax_idx]).st_size
                if tsum_size == tmax_size:
                    mapped_same_insts.append((topksum_pts[0], topkmax_pts[curr_tmax_idx]))
                    del topksum_pts[0], topkmax_pts[curr_tmax_idx]
                    break
    
    print(f"mapped insts: {mapped_same_insts}")
    for same_inst_pair in tqdm(mapped_same_insts):
        compare_topk_focus_inst(same_inst_pair[0], same_inst_pair[1], (3, 20))
    exit()
    # import jsonw
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
    
