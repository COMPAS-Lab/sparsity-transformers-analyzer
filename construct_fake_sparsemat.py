import torch
import numpy as np
import random
from math import ceil
from sparse_tensor_analyzer import distance_of_dense_vals_per_row

def gen_spmat_by_sparsity(ref_mat: np.array, 
                          target_seqlen: int, 
                          fake_dense_val = 0.4, 
                          is_plot_figure = False) -> np.array:
    row_spar = np.count_nonzero(ref_mat, axis=-1)
    row_dist, _ = distance_of_dense_vals_per_row(ref_mat, row_size=1)
    row_first_dval = np.argmax(ref_mat > 0, axis=-1)

    ref_mat_len = float(ref_mat.shape[-1])
    scaled_row_dist = np.ceil(row_dist / ref_mat_len * target_seqlen)
    scaled_row_first_dval = np.ceil(row_first_dval / ref_mat_len * target_seqlen)
    num_dvals_row = np.ceil(target_seqlen * row_spar / ref_mat_len)

    #scale each row distribution attribute to target len:
    scaled_x_coord = np.arange(0, ref_mat_len, 1)
    scaled_x_coord = np.floor(scaled_x_coord / ref_mat_len * target_seqlen)
    scaled_row_dist = np.ceil(np.interp(list(range(target_seqlen)), 
                                scaled_x_coord, scaled_row_dist)).astype(np.int)
    scaled_row_first_dval = np.floor(np.interp(list(range(target_seqlen)), 
                                scaled_x_coord, scaled_row_first_dval)).astype(np.int)
    num_dvals_row = np.floor(np.interp(list(range(target_seqlen)), 
                                scaled_x_coord, num_dvals_row)).astype(np.int)
    
    if is_plot_figure:
        from matplotlib import pyplot as plt
        plt.plot(row_dist, color="r")
        plt.plot(row_first_dval, color="g")
        plt.plot(row_spar, color="b")
        plt.savefig("./res_fig/temp/row_spar_feature.pdf")
        plt.clf()
    
        plt.plot(scaled_row_dist, color="r")
        plt.plot(scaled_row_first_dval, color="g")
        plt.plot(num_dvals_row, color="b")
        plt.savefig("./res_fig/temp/row_spar_feature_scaled.pdf")
        plt.clf()

    res_mat = np.zeros((target_seqlen, target_seqlen))
    for r in range(int(ref_mat_len)):
        cand_cols = np.arange(
            scaled_row_first_dval[r], 
            scaled_row_dist[r] + scaled_row_first_dval[r] + 1, 1).astype(np.int)
        selected_num_cols = min(len(cand_cols), num_dvals_row[r])
        if len(cand_cols) < num_dvals_row[r]:
            print("#candidates not enough!")
        dval_cols = random.sample(list(cand_cols), selected_num_cols)
        for c in dval_cols:
            res_mat[r][c] = fake_dense_val

    # confirm sparsity
    ori_spar = np.count_nonzero(ref_mat) / ref_mat.size
    new_spar = np.count_nonzero(res_mat) / res_mat.size
    print(f"ori: {ori_spar:.3f}, new: {new_spar:.3f}")
    return res_mat

if __name__ == "__main__":
    target_len = 2048
    for i in [1]:
        ref_path = f"/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/llama-7b-hf-attsample/attn_s{i}b0.pt"
        tar_path = f"/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/seqlen_2048_interp_llama7bhf/attn_s{i}b0.pt"
        src_attn = torch.load(ref_path).numpy()
        det_attn = np.zeros((src_attn.shape[0], src_attn.shape[1], target_len, target_len))
        for layer in range(src_attn.shape[0]):
            for head in range(src_attn.shape[1]):
                temp_src = src_attn[layer][head]
                temp_tar = gen_spmat_by_sparsity(temp_src, target_len, is_plot_figure=True)
                det_attn[layer][head] = temp_tar
                exit()

        print(f"src shape: {src_attn.shape}, gen shape: {det_attn.shape}")
        det_attn = torch.tensor(det_attn)
        torch.save(det_attn, tar_path)