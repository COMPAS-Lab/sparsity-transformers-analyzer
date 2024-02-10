import torch
import numpy as np
import random
from tqdm import tqdm
from scipy import optimize
from math import ceil, floor
from sparsemat_hw_modeling import distance_of_dense_vals_per_row, dense_val_idx_histogram, plot_stacked_heatmap_inst, block_prune_analysis
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile
import json

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


if __name__ == "__main__":
    inst_idx = 4
    base_attn_path = f"/chronos_data/tji/.huggingface_cache/transformers/chatglm2-6b-32k-attn-bfp20-1e-3-hotpotqa-bprune-scaled/"
    # list all insts
    inst_list = [f.split(".")[0] \
                 for f in listdir(base_attn_path) \
                    if isfile(base_attn_path + f) and f[0] == "i" and f.endswith(".pt")]
    
    # inst_list = random.sample(inst_list, 1)
    # inst_list = inst_list[:5]
    # src_attn = torch.load(attn_path_chatglm).numpy()
    # explore_row_features(src_attn, inst_idx=inst_idx)
    # dense_val_idx_histogram([attn_path_chatglm], 50)
    # for i in inst_list:
    #     plot_stacked_heatmap_inst([base_attn_path + i + ".pt"], \
    #                                 [base_attn_path + i + ".json"], \
    #                                 f"./res_fig/temp/hotpotqa/{i}")

    res = block_prune_analysis([base_attn_path + p + ".pt" for p in inst_list], (3, 20))
    # import jsonw
    with open("./res_fig/block_prune/hotpotqa_bprune/bpruning_hotpotqa_bthres_3p0.json", "w+") as f:
        json.dump(res, f)

    elem_spars = []
    for i in [base_attn_path + p + ".json" for p in inst_list]:
        with open(i) as fp:
            res = json.load(fp)
            elem_spars.append(res["spar_mean"])

    all_bsparse = None
    with open("./res_fig/block_prune/hotpotqa_bprune/bpruning_hotpotqa_bthres_3p0.json") as fp:
        res = json.load(fp)
        all_bsparse = list(res.values())

    print(f"element-wise sparsity: {np.mean(elem_spars)}")
    print(f"block sparsity: {np.mean(all_bsparse)}")
    exit()
    target_len = 2048

    for i in range(1):
        ref_path = (f"/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/"
                    f"llama-7b-hf-attsample/attn_s{i}b0.pt")
        tar_path = (f"/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/"
                    f"seqlen_{target_len}_interp_llama7bhf/attn_s{i}b0.pt")
        src_attn = torch.load(ref_path).numpy()

        explore_row_features(src_attn, inst_idx=i)
        det_attn = np.zeros((src_attn.shape[0], src_attn.shape[1], target_len, target_len))
        for layer in range(src_attn.shape[0]):
            for head in range(src_attn.shape[1]):
                temp_src = src_attn[layer][head]
                temp_tar = gen_spmat_by_sparsity(temp_src, target_len, is_plot_figure=True)
                det_attn[layer][head] = temp_tar

        print(f"src shape: {src_attn.shape}, gen shape: {det_attn.shape}")
        det_attn = torch.tensor(det_attn)
        torch.save(det_attn, tar_path)
