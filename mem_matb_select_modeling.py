import torch
import numpy as np
import random
from tqdm import tqdm
from scipy import optimize
from math import ceil, floor
from sparse_tensor_analyzer import distance_of_dense_vals_per_row
from matplotlib import pyplot as plt

def multiport_congestion_analysis(attn, 
                        attn_flatten_size: tuple, 
                        num_ports: int, 
                        num_tc_access: int,
                        num_replication: int,
                        cached_range_in_col: tuple[int, int], 
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
            # segmented mem sections based on the number of SPRAMs, each 
            # SPRAM stores the access within [0, (p + 1) * mem_segsize-1] 
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
                # check replicated mem access in every section
                for sec_idx in range(num_ports):
                    if replicated_range[1] > mem_segment_idx[sec_idx]:
                        # first filter out uncached ones
                        sec_uncacheable_access = np.sum(np.array(curr_loaded_access[sec_idx]) > replicated_range[1])
                        sec_cacheable_access = float(len(curr_loaded_access[sec_idx]) - sec_uncacheable_access)
                        cacheable_access_iter = ceil(sec_cacheable_access / num_replication)
                        if sec_uncacheable_access > 0:
                            all_access_np = np.array(curr_loaded_access[sec_idx])
                            uncacheable_access_idx = np.where(all_access_np >= replicated_range[1])
                            curr_loaded_access[sec_idx] = all_access_np[uncacheable_access_idx].tolist()
                            if len(curr_ori_loaded_access[sec_idx]) > len(curr_loaded_access[sec_idx]):
                                # deal with the situation that there are cached contents being accessed
                                curr_loaded_access[sec_idx] = [-1] * cacheable_access_iter + all_access_np[uncacheable_access_idx].tolist()
                        elif curr_loaded_access[sec_idx]:
                            # deal with the situation that only cached contents are accessed
                            curr_loaded_access[sec_idx] = [-1] * cacheable_access_iter
                    else:
                        break

                # merge current res to all loaded_access
                for p in range(num_ports):
                    loaded_access[p] += curr_loaded_access[p]
                    ori_loaded_access[p] += curr_ori_loaded_access[p]

                ideal_cycles = float(sum([len(r) for r in ori_loaded_access])) / num_ports
                actual_cycles = max([len(r) for r in loaded_access])
                all_loaded_ratio = float(actual_cycles) / ideal_cycles

            return loaded_access, ori_loaded_access

        if mat.shape[0] % num_tc_access != 0:
            num_padded_row_iters = ceil(mat.shape[0] / num_tc_access) * num_tc_access - mat.shape[0]
            mat = np.pad(mat, ((0, num_padded_row_iters), (0, 0)),
                        "constant", constant_values=0)
        
        tc_access_grpsize = mat.shape[0] // num_tc_access
        total_loaded_access = [[] for p in range(num_ports)]
        total_ori_loaded_access = [[] for p in range(num_ports)]
        for i in range(tc_access_grpsize):
            # method 2: interleaving
            row_access_list = [mat[j * tc_access_grpsize + i] for j in range(num_tc_access)]
            loaded_access, ori_loaded_access = get_loaded_access(row_access_list, cached_range_in_col)
            for p in range(num_ports):
                total_loaded_access[p] += loaded_access[p]
                total_ori_loaded_access[p] += ori_loaded_access[p]

        ideal_cycles = float(sum([len(r) for r in total_ori_loaded_access])) / num_ports
        actual_cycles = max([len(r) for r in total_loaded_access])
        all_loaded_ratio = float(actual_cycles) / ideal_cycles
        
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
                    f"_with_cache{cached_range_in_col[0]}to{cached_range_in_col[1]}.png")
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


def col_rep_2_m20k(rep_time, col_range, max_len_per_memseg, num_memseg, bcol_per_buffer, num_tc_cols):
    ALL_M20K = 6847.0
    M20K_SIZE = 20480.0
    base_single_buffer_size = max_len_per_memseg * 16.0 * bcol_per_buffer

    num_full_rep_memseg = floor(col_range / max_len_per_memseg)
    partial_rep_col_range = col_range - max_len_per_memseg * num_full_rep_memseg

    num_replicated_memseg = ceil(float(col_range) / max_len_per_memseg)
    num_unreplicated_memseg = float(num_memseg - num_replicated_memseg)
    unreplicated_m20k = ceil(base_single_buffer_size / M20K_SIZE) * num_unreplicated_memseg
    replicated_m20k = ceil(base_single_buffer_size / M20K_SIZE) * (rep_time + 1)  * num_full_rep_memseg + \
                        ceil(base_single_buffer_size / M20K_SIZE) + \
                        ceil((partial_rep_col_range * 16 * bcol_per_buffer) / M20K_SIZE) * rep_time
    
    print(f"#unreplicated and #replicated: {unreplicated_m20k} and {replicated_m20k}")
    print(f"base: {base_single_buffer_size/M20K_SIZE}")
    m20k_ratio = (unreplicated_m20k + replicated_m20k) * num_tc_cols / ALL_M20K

    return m20k_ratio


def mem_acc_cycle_diff_numports(num_ports: tuple[int, int, int], 
                                num_tc_access: int,
                                ref_att_path_list: list[str]):
    '''
    plot access cycle changes as the number of SPRAM changes
    '''
    avg_access = []
    avg_seq_len = 0.0
    for curr_num_ports in np.arange(*num_ports):
        all_access_cycles_list, all_ideal_cycles_list = [], []
        all_seq_len = []
        for inst in ref_att_path_list:
            src_attn = torch.load(inst).numpy()
            all_seq_len.append(src_attn.shape[-1])
            res = multiport_congestion_analysis(src_attn, 
                                    (-1, src_attn.shape[-2], src_attn.shape[-1]), 
                                    num_ports=curr_num_ports, num_tc_access=num_tc_access, num_replication=10, 
                                    cached_range_in_col=(0, 5), is_plot_figure=True)
            all_access_cycles_list += [res[0]]
            all_ideal_cycles_list += [res[1]]
        avg_access += [(np.mean(all_access_cycles_list), np.mean(all_ideal_cycles_list))]
        avg_seq_len = np.mean(all_seq_len)

    # calculate reference dense mat mul reading
    # MARK: hard-coded number of mat B cols to be 128
    # num_loada_iters = seq_len / (20 * chain_len)
    num_loada_iters = avg_seq_len / (20 * 14)
    dense_parallel_matb_access = 128/2 * num_loada_iters * 32 * 32

    plt.figure()
    plt.plot(np.arange(*num_ports), 
             [a[0] for a in avg_access], 
             marker="s", color="b", label="actual RAM read")
    plt.plot(np.arange(*num_ports), 
             [a[1] for a in avg_access], 
             marker="s", color="r", label="ideal RAM read")
    plt.hlines(dense_parallel_matb_access, num_ports[0], num_ports[1], 
               colors="black", linestyles="dashed", label="dense parallel read")
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
    avg_seq_len = 0.0
    for curr_cached_cols in np.arange(*num_cached_cols):
        all_access_cycles_list, all_ideal_cycles_list = [], []
        all_seq_len = []
        for inst in ref_att_path_list:
            src_attn = torch.load(inst).numpy()
            all_seq_len.append(src_attn.shape[-1])
            res = multiport_congestion_analysis(src_attn, 
                                    (-1, src_attn.shape[-2], src_attn.shape[-1]), 
                                    num_ports=num_ports, num_tc_access=num_tc_access, num_replication=num_tc_access,
                                    cached_range_in_col=(0, curr_cached_cols), is_plot_figure=True)
            all_access_cycles_list += [res[0]]
            all_ideal_cycles_list += [res[1]]
        avg_access += [(np.mean(all_access_cycles_list), np.mean(all_ideal_cycles_list))]
        avg_seq_len = np.mean(all_seq_len)

    # calculate reference dense mat mul reading
    # MARK: hard-coded number of mat B cols to be 128
    # num_loada_iters = seq_len / (20 * chain_len)
    num_loada_iters = avg_seq_len / (20 * 14)
    dense_parallel_matb_access = 128/2 * num_loada_iters * 32 * 32

    plt.figure()
    plt.plot(np.arange(*num_cached_cols), 
             [a[0] for a in avg_access], 
             marker="s", color="b", label="w/ replication")
    plt.plot(np.arange(*num_cached_cols), 
             [a[1] for a in avg_access], 
             marker="s", color="r", label="w/o replication") 
    plt.hlines(dense_parallel_matb_access, num_cached_cols[0], num_cached_cols[1], 
               colors="black", linestyles="dashed", label="dense parallel read")
    plt.xlabel("replicated range in each col (0->x)")
    plt.ylabel("#cycles to read RAM")
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig(f"./res_fig/temp/mem_acc_vs_cached_cols.png")
    plt.close()


def mem_acc_cycle_diff_rep_times(num_rep_times: tuple[int, int, int], 
                                   num_ports: int, num_tc_access: int,
                                   cached_range: int,
                                   ref_att_path_list: list[str]):
    '''
    plot access cycle changes as the number of cached columns in the mat B
    changes
    '''
    avg_access = []
    avg_seq_len = 0.0
    for curr_rep_time in np.arange(*num_rep_times):
        all_access_cycles_list, all_ideal_cycles_list = [], []
        all_seq_len = []
        for inst in ref_att_path_list:
            src_attn = torch.load(inst).numpy()
            all_seq_len.append(src_attn.shape[-1])
            res = multiport_congestion_analysis(src_attn, 
                                    (-1, src_attn.shape[-2], src_attn.shape[-1]), 
                                    num_ports=num_ports, num_tc_access=num_tc_access, num_replication=curr_rep_time,
                                    cached_range_in_col=(0, cached_range), is_plot_figure=True)
            all_access_cycles_list += [res[0]]
            all_ideal_cycles_list += [res[1]]
        avg_access += [(np.mean(all_access_cycles_list), np.mean(all_ideal_cycles_list))]
        avg_seq_len = np.mean(all_seq_len)

    # calculate reference dense mat mul reading
    # MARK: hard-coded number of mat B cols to be 128
    # num_loada_iters = seq_len / (20 * chain_len)
    num_loada_iters = avg_seq_len / (20 * 14)
    dense_parallel_matb_access = 128/2 * num_loada_iters * 32 * 32

    plt.figure()
    plt.plot(np.arange(*num_rep_times), 
             [a[0] for a in avg_access], 
             marker="s", color="b", label="w/ replication")
    plt.plot(np.arange(*num_rep_times), 
             [a[1] for a in avg_access], 
             marker="s", color="r", label="w/o replication") 
    plt.hlines(dense_parallel_matb_access, num_rep_times[0], num_rep_times[1], 
               colors="black", linestyles="dashed", label="dense parallel read")
    plt.xlabel("#replication")
    plt.ylabel("#cycles to read RAM")
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig(f"./res_fig/temp/mem_acc_vs_rep_times.png")
    plt.close()


def analyze_rep_cols_vs_m20k(rep_time, col_range):
    max_len = 500
    num_memseg = 20
    res = [col_rep_2_m20k(rep_time, c, max_len / num_memseg, num_memseg, 64, 2) for c in col_range]

    plt.figure()
    plt.plot(col_range, res, marker="s", color="b")
    plt.xlabel("replicated cols")
    plt.ylabel("M20k util")
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    # plt.legend()
    plt.savefig(f"./res_fig/temp/cols_vs_m20k.png")
    plt.close()

if __name__ == "__main__":
    target_len = 2048
    selected_idx = random.sample(list(range(100)), 50)
    ref_attn_list = [f"/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/" + \
                    f"llama-7b-hf-attsample/attn_s{i}b0.pt" for i in selected_idx]
    
    mem_acc_cycle_diff_numports((2, 24, 2), 123, ref_attn_list)
    mem_acc_cycle_diff_cached_cols((1, 100, 5), 20, 123, ref_attn_list)
    mem_acc_cycle_diff_rep_times((1, 124, 1), 20, 123, 30, ref_attn_list)
    analyze_rep_cols_vs_m20k(123, np.arange(5, 100, 5))
