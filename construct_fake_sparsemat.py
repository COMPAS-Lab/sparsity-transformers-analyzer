import torch
import numpy as np
import random
from scipy import optimize
from math import ceil
from sparse_tensor_analyzer import distance_of_dense_vals_per_row

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
        from matplotlib import pyplot as plt
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
    from matplotlib import pyplot as plt

    row_spar_list, row_dist_list, row_first_dval_list = [], [], []

    for ref_mat_l in mats:
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


def plot_congested_inst(attn, attn_flatten_size: tuple, is_plot_figure = False):
    '''
    check the total number of congestions in a given flattened attention array
    '''
    def analyze_multiport_balance(mat, num_ports: int):
        assert(mat.ndim == 2), "incorrect mat size"

        def row_congest(rows: list, mem_segsize: int):
            num_ports = len(rows)
            ports_accessing_list = [0] * num_ports
            mem_segment_idx = np.array([p * mem_segsize for p in range(num_ports)])
            for r in rows:
                # find out row access idx
                r_access = np.where(r > 0.0)[0]
                # figure out the congestions
                for r_access_idx in r_access:
                    curr_access_idx = np.argmax(mem_segment_idx > r_access_idx)
                    ports_accessing_list[curr_access_idx] += 1

            # compute ratio of the congestion
            # congestion ratio: cycles actually take / cycles take with perfect balance
            ideal_cycles = ceil(np.sum(ports_accessing_list) / num_ports)
            real_cycles = np.amax(ports_accessing_list)
            return real_cycles, ideal_cycles

        if mat.shape[0] % num_ports != 0:
            mat = np.pad(mat, ((0, mat.shape[0] % num_ports), (0, 0)),
                        "constant", constant_values=0)
        
        mem_segsize = mat.shape[0] // num_ports
        total_real_cycles, total_ideal_cycles = 0, 0
        for i in range(mem_segsize):
            # method 1: parallizing neighboring rows
            row_access_list = [mat[i * num_ports + j] for j in range(num_ports)]
            # method 2: interleaving the rows
            # row_access_list = [mat[j * mem_segsize + i] for j in range(num_ports)]
            real_cycles, ideal_cycles = row_congest(row_access_list, mem_segsize)
            total_real_cycles += real_cycles
            total_ideal_cycles += ideal_cycles

        return total_real_cycles, total_ideal_cycles
        
    flatten_attn = np.resize(attn, attn_flatten_size)
    all_overhead = []
    for a in flatten_attn:
        r_cycles, a_cycles = analyze_multiport_balance(a, 16)
        all_overhead += [r_cycles / a_cycles]

    if is_plot_figure:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(16, 9))
        # plt.scatter(list(range(len(total_num_access))), total_num_access, 
        #         color="b", alpha=0.4)
        # plt.scatter(list(range(len(total_num_cong))), total_num_cong, 
        #         color="b", alpha = 1.0)
        plt.scatter(list(range(len(all_overhead))), all_overhead, 
                color="b", alpha=0.4)
        all_overhead = np.mean(all_overhead)
        plt.title(f"ratio of congestion: {all_overhead:.3f}")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0, ymax=10)
        plt.savefig("./res_fig/temp/congestion_multiway.png")

    return all_overhead

if __name__ == "__main__":
    target_len = 2048
    for i in range(1):
        ref_path = (f"/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/"
                    f"llama-7b-hf-attsample/attn_s{i}b0.pt")
        tar_path = (f"/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/"
                    f"seqlen_{target_len}_interp_llama7bhf/attn_s{i}b0.pt")
        src_attn = torch.load(ref_path).numpy()

        # explore_row_features(src_attn, inst_idx=i)
        # det_attn = np.zeros((src_attn.shape[0], src_attn.shape[1], target_len, target_len))
        # for layer in range(src_attn.shape[0]):
        #     for head in range(src_attn.shape[1]):
        #         temp_src = src_attn[layer][head]
        #         temp_tar = gen_spmat_by_sparsity(temp_src, target_len, is_plot_figure=True)
        #         det_attn[layer][head] = temp_tar

        # print(f"src shape: {src_attn.shape}, gen shape: {det_attn.shape}")
        # det_attn = torch.tensor(det_attn)
        # torch.save(det_attn, tar_path)

        # analyze congestions
        res = plot_congested_inst(src_attn, (-1, src_attn.shape[-2], src_attn.shape[-1]), is_plot_figure=True)