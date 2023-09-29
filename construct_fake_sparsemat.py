import torch
import numpy as np
import random
from tqdm import tqdm
from scipy import optimize
from math import ceil
from sparse_tensor_analyzer import distance_of_dense_vals_per_row
from matplotlib import pyplot as plt

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


def multiport_congestion_analysis(attn, 
                        attn_flatten_size: tuple, 
                        num_ports: int, 
                        num_tc_access: int,
                        cached_cols: tuple[int, int], 
                        is_plot_figure = False):
    '''
    check the total number of congestions in a given flattened attention array
    '''
    def analyze_multiport_balance(mat):
        assert(mat.ndim == 2), "incorrect mat size"

        def get_loaded_access(rows: list, replicated_range: tuple):
            # init results
            loaded_access = [[] for p in range(num_ports)]
            ori_loaded_access = [[] for p in range(num_ports)]
            # segmented mem sections based on the number of SPRAMs 
            mem_segsize = ceil(rows[0].shape[0] / num_ports)
            mem_segment_idx = np.array([p * mem_segsize for p in range(num_ports)])

            r_access_list = [np.where(r > 0.0)[0].tolist() for r in rows]
            while sum([len(rlist) for rlist in r_access_list]) > 0:
                curr_loaded_access = [[] for p in range(num_ports)]
                for r in range(len(rows)):
                    if r_access_list[r]:
                        mem_addr = np.argmax(mem_segment_idx > r_access_list[r][0]) - 1
                        curr_loaded_access[mem_addr] += [r_access_list[r][0]]
                        r_access_list[r].pop(0)

                # maintain the original load here
                curr_ori_loaded_access = curr_loaded_access[:]
                # check replicated mem access in the section 0
                sec0_uncacheable_access = np.sum(np.array(curr_loaded_access[0]) >= replicated_range[1])
                if sec0_uncacheable_access > 0:
                    all_access_np = np.array(curr_loaded_access[0])
                    uncacheable_access_idx = np.where(all_access_np >= replicated_range[1])
                    curr_loaded_access[0] = all_access_np[uncacheable_access_idx].tolist()
                    if len(curr_ori_loaded_access[0]) > len(curr_loaded_access[0]):
                        curr_loaded_access[0] = [-1] + all_access_np[uncacheable_access_idx].tolist()
                elif curr_loaded_access[0]:
                    curr_loaded_access[0] = [-1]
                
                lensum = 0
                for a in curr_loaded_access:
                    lensum += len(a)

                if lensum == 0:
                    print("void loaded access!")
                # merge current res to all loaded_access
                for p in range(num_ports):
                    loaded_access[p] += curr_loaded_access[p]
                    ori_loaded_access[p] += curr_ori_loaded_access[p]

            return loaded_access, ori_loaded_access

        if mat.shape[0] % num_tc_access != 0:
            mat = np.pad(mat, ((0, mat.shape[0] % num_tc_access), (0, 0)),
                        "constant", constant_values=0)
        
        tc_access_grpsize = mat.shape[0] // num_tc_access
        total_loaded_access = [[] for p in range(num_ports)]
        total_ori_loaded_access = [[] for p in range(num_ports)]
        for i in range(tc_access_grpsize):
            # method 2: interleaving
            row_access_list = [mat[j * tc_access_grpsize + i] for j in range(num_tc_access)]
            loaded_access, ori_loaded_access = get_loaded_access(row_access_list, cached_cols)
            for p in range(num_ports):
                total_loaded_access[p] += loaded_access[p]
                total_ori_loaded_access[p] += ori_loaded_access[p]

        return total_loaded_access, total_ori_loaded_access
        
    flatten_attn = np.resize(attn, attn_flatten_size)
    all_loaded_ratio = []
    num_all_actual_cycles, num_all_ideal_cycles = [], []
    for a in tqdm(flatten_attn, unit=" mat"):
        mat_loaded_access, mat_ori_loaded_access = analyze_multiport_balance(a)
        num_ideal_cycles = float(sum([len(r) for r in mat_ori_loaded_access])) / num_ports
        num_actual_cycles = max([len(r) for r in mat_loaded_access])
        all_loaded_ratio += [float(num_actual_cycles) / num_ideal_cycles]
        num_all_actual_cycles += [num_actual_cycles]
        num_all_ideal_cycles += [num_ideal_cycles]

    if is_plot_figure:
        plt.figure(figsize=(16, 9))
        plt.scatter(list(range(len(num_all_actual_cycles))), num_all_actual_cycles, 
                color="b", alpha=0.4, label="actual")
        plt.scatter(list(range(len(num_all_ideal_cycles))), num_all_ideal_cycles, 
                color="r", alpha=0.4, label="ideal")
        all_overhead = np.mean(all_loaded_ratio)
        plt.title(f"avgerage ratio of congestion: {all_overhead:.3f}")
        plt.xlabel("mat index")
        plt.ylabel("#mem access")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.legend()
        plt.savefig(f"./res_fig/temp/congestion_{num_ports}p_{num_tc_access}tc" + \
                    f"_with_cache{cached_cols[0]}to{cached_cols[1]}.png")
        plt.close()

    return sum(num_all_actual_cycles), sum(num_all_ideal_cycles)


def multiway_congestion_dist_analysis(
        attn, 
        attn_flatten_size: tuple, 
        num_ports: int,
        is_plot_figure: bool):
    '''
    check the distribution of the congestions that causes 
    the extra cycles
    '''
    def analyze_multiport_balance(mat):
        assert(mat.ndim == 2), "incorrect mat size"

        def get_loaded_access(rows: list, mem_segsize: int):
            loaded_access = [[] for p in range(num_ports)]
            mem_segment_idx = np.array([p * mem_segsize for p in range(num_ports)])
            r_access_list = [np.where(r > 0.0)[0].tolist() for r in rows]
            while sum([len(rlist) for rlist in r_access_list]) > 0:
                for r in range(len(rows)):
                    if r_access_list[r]:
                        mem_addr = np.argmax(mem_segment_idx > r_access_list[r][0]) - 1
                        loaded_access[mem_addr] += [r_access_list[r][0]]
                        r_access_list[r].pop(0)

            return loaded_access

        if mat.shape[0] % num_ports != 0:
            mat = np.pad(mat, ((0, mat.shape[0] % num_ports), (0, 0)),
                        "constant", constant_values=0)
        
        mem_segsize = mat.shape[0] // num_ports
        total_loaded_access = [[] for p in range(num_ports)]
        for i in range(mem_segsize):
            # method 1: parallizing neighboring rows
            row_access_list = [mat[i * num_ports + j] for j in range(num_ports)]
            # method 2: interleaving the rows
            # row_access_list = [mat[j * mem_segsize + i] for j in range(num_ports)]
            curr_loaded_access = get_loaded_access(row_access_list, mem_segsize)
            for p in range(num_ports):
                total_loaded_access[p] += curr_loaded_access[p]

        return total_loaded_access
        
    flatten_attn = np.resize(attn, attn_flatten_size)
    all_loaded_access = [[] for p in range(num_ports)]
    for a in flatten_attn:
        mat_loaded_access = analyze_multiport_balance(a)
        for p in range(num_ports):
            all_loaded_access[p] += mat_loaded_access[p]

    # plot distribution
    if is_plot_figure:
        mem_segsize = ceil(flatten_attn.shape[-1] / num_ports)
        plt.figure()
        for p in range(num_ports):
            plt.hist(all_loaded_access[p], 20, (p * mem_segsize, (p + 1) * mem_segsize))
            
        plt.title(f"congestion distribution")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.savefig(f"./res_fig/temp/congestion_dist.png")
    
    return

def mem_acc_cycle_diff_numports(num_ports: tuple[int, int, int], 
                                num_tc_access: int,
                                ref_att_path_list: list[str]):
    '''
    plot access cycle changes as the number of SPRAM changes
    '''
    avg_access = []
    for curr_num_ports in np.arange(*num_ports):
        all_access_cycles_list, all_ideal_cycles_list = [], []
        for inst in ref_att_path_list:
            src_attn = torch.load(inst).numpy()
            res = multiport_congestion_analysis(src_attn, 
                                    (-1, src_attn.shape[-2], src_attn.shape[-1]), 
                                    num_ports=curr_num_ports, num_tc_access=num_tc_access,
                                    cached_cols=(0, 5), is_plot_figure=True)
            all_access_cycles_list += [res[0]]
            all_ideal_cycles_list += [res[1]]
        avg_access += [(np.mean(all_access_cycles_list), np.mean(all_ideal_cycles_list))]


    plt.figure()
    plt.plot(np.arange(*num_ports), 
             [a[0] for a in avg_access], 
             marker="s", color="b", label="actual RAM read")
    plt.plot(np.arange(*num_ports), 
             [a[1] for a in avg_access], 
             marker="s", color="r", label="ideal RAM read") 
    plt.xlabel("#SPRAM")
    plt.ylabel("#cycles to read RAM")
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig(f"./res_fig/temp/mem_acc_vs_numports.png")
    plt.close()


def mem_acc_cycle_diff_cached_cols(num_cached_cols: tuple[int, int, int], 
                                   num_ports: int, num_tc_access: int,
                                   ref_att_path_list: list[str]):
    '''
    plot access cycle changes as the number of cached columns in the mat B
    changes
    '''
    avg_access = []
    for curr_cached_cols in np.arange(*num_cached_cols):
        all_access_cycles_list, all_ideal_cycles_list = [], []
        for inst in ref_att_path_list:
            src_attn = torch.load(inst).numpy()
            res = multiport_congestion_analysis(src_attn, 
                                    (-1, src_attn.shape[-2], src_attn.shape[-1]), 
                                    num_ports=num_ports, num_tc_access=num_tc_access,
                                    cached_cols=(0, curr_cached_cols), is_plot_figure=True)
            all_access_cycles_list += [res[0]]
            all_ideal_cycles_list += [res[1]]
        avg_access += [(np.mean(all_access_cycles_list), np.mean(all_ideal_cycles_list))]

    plt.figure()
    plt.plot(np.arange(*num_cached_cols), 
             [a[0] for a in avg_access], 
             marker="s", color="b", label="actual RAM read")
    plt.plot(np.arange(*num_cached_cols), 
             [a[1] for a in avg_access], 
             marker="s", color="r", label="ideal RAM read") 
    plt.xlabel("cached cols (0 to x)")
    plt.ylabel("#cycles to read RAM")
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig(f"./res_fig/temp/mem_acc_vs_cached_cols.png")
    plt.close()


if __name__ == "__main__":
    target_len = 2048
    selected_idx = random.sample(list(range(100)), 10)
    ref_attn_list = [f"/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/" + \
                    f"llama-7b-hf-attsample/attn_s{i}b0.pt" for i in selected_idx]
    
    # mem_acc_cycle_diff_numports((2, 24, 2), 123, ref_attn_list)
    mem_acc_cycle_diff_cached_cols((20, 100, 10), 20, 123, ref_attn_list)
    exit()

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
