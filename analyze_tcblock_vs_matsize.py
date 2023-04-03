
import torch
import matplotlib
import numpy as np
import hw_modeling
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
from math import ceil, floor, sqrt
from hw_modeling import StratixDpuModel
import pickle

def closest_factors_to_target(chain, target):
    true_chainlen = float(chain+2)
    rest_fact = target/true_chainlen

    closest_multres = 0
    factor_1, factor_2 = 0, 0

    for i in range(2, ceil(sqrt(rest_fact)), 1):
        mult = floor(rest_fact/i)
        curr_multres = mult*i*true_chainlen
        if (curr_multres <= target) and \
            (target-curr_multres <= (target-closest_multres)):
            closest_multres = curr_multres
            factor_1, factor_2 = i, mult
    
    return factor_1, factor_2

def list_closest_factors_to_target(chain, target):
    true_chainlen = int(chain)
    rest_fact = target//true_chainlen

    factor_list = []

    for i in range(2, ceil(sqrt(rest_fact)), 1):
        mult = rest_fact//i
        curr_multres = mult*i*true_chainlen
        if curr_multres == target:
            factor_list.append((i, mult))
    
    return factor_list

def get_util(workload: dict, hw_size: tuple, chain_len: int):
    '''
    get_util: compute unpadded_matsize / actual_matsize after padding being computed
    '''
    mat_a_row, mat_a_col, _, mat_b_col = workload["size"]
    row_spar = workload["row_sparsity"]
    mat_a_col = ceil(mat_a_col*(1.-row_spar))
    hw_row, hw_col = hw_size

    hw_vec_actualsize = chain_len*20*ceil(mat_a_col/(chain_len*20.0))
    hw_matb_actual_colsize = hw_row*ceil(mat_b_col/hw_row)
    hw_mata_actual_rowsize = 3*hw_col*ceil(mat_a_row/(3.0*hw_col))

    original_workload_size = mat_a_row * mat_a_col * mat_b_col
    actual_size = hw_vec_actualsize * hw_matb_actual_colsize * hw_mata_actual_rowsize
    return original_workload_size/actual_size

def tcchain_r_c_sweep():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    sizes = (3456, 180, 180, 3456)
    # sizes = (3456, 2160, 2160, 3456)

    flops_vs_chain_len = {"4": [], "9": [], "12":[], "18": []}
    util_vs_chain_len = {"4": [], "9": [], "12":[], "18": []}
    xaxis_per_chainlen = {"4": [], "9": [], "12":[], "18": []}
    best_shape = {"4": [], "9": [], "12":[], "18": []}
    
    tcc_array_shapes = {l: list_closest_factors_to_target(int(l), 2304) for l in flops_vs_chain_len.keys()}
    print(tcc_array_shapes)

    for chain_len_str in flops_vs_chain_len.keys():
        chain_len = int(chain_len_str)

        #sweep across hw shape
        flops, util, xaxis = [], [], []
        curr_best_flops = float("-inf")
        for hw_shape in tcc_array_shapes[chain_len_str]:
            mat_a_row, mat_a_col, mat_b_row, mat_b_col = sizes
            total_util = get_util(sizes, hw_shape, chain_len)

            fake_dense_data = np.ones((mat_a_row, mat_a_col))
            curr_model = StratixDpuModel(mat_a_row, mat_a_col, mat_b_row, mat_b_col, 
                                        exp_dat=fake_dense_data, 
                                        freq=300, 
                                        num_tcs=3960, 
                                        tcc_array_shape=hw_shape, 
                                        tcc_chainlen=chain_len)

            curr_model.set_tccore_size(20)
            curr_flops, curr_lat = curr_model.tensor_fpga21_mat_sparse_flops(
                                                        fake_dense_data, \
                                                        True, False, \
                                                        using_single_column=False,
                                                        sparse_block_size=1)
            flops.append(curr_flops)
            util.append(total_util)
            xaxis.append(hw_shape[0])

            if curr_flops > curr_best_flops:
                curr_best_flops = curr_flops
                best_shape[chain_len_str] = hw_shape

        
        flops_vs_chain_len[chain_len_str] = flops
        util_vs_chain_len[chain_len_str] = util
        xaxis_per_chainlen[chain_len_str] = xaxis

    print("best shapes: ", best_shape)

    fig, axs=plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    for idx, clen in enumerate(flops_vs_chain_len.keys()):
        axs[0].plot(xaxis_per_chainlen[clen], flops_vs_chain_len[clen], linestyle='-', 
                marker='s', linewidth=1, alpha=0.8, label=f"chain len={clen}", color=f"C{idx}")
        axs[1].plot(xaxis_per_chainlen[clen], util_vs_chain_len[clen], linestyle='-', 
                marker='s', linewidth=1, alpha=0.8, label=f"chain len={clen}", color=f"C{idx}")
        
    axs[0].set_ylabel('TOPs')
    axs[0].set_xlim(xmin=0)
    axs[0].set_xlabel('tensor core array rows')
    axs[0].set_ylim(ymin=0)
    axs[0].set_ylim(ymax=100)
    axs[0].grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    axs[0].legend()
    axs[1].set_xlabel('tensor core array rows')
    axs[1].set_ylabel('utils')
    axs[1].set_xlim(xmin=0)
    axs[1].set_ylim(ymax=1.0)
    axs[1].grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig("res_fig/rowsize_sweep_chainlen_small.pdf")
    plt.cla()

def tcchain_len_sweep(mats):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3
    
    mats = [{"size": (1024, 1024, 1024, 768), "row_sparsity": frac, "label": "sparse_attn_h"} \
            for frac in np.arange(0.1, 1.0, 0.1)
            ]
    
    mats += [
            #13b
            {"size": (1024, 5120, 5120, 5120), "row_sparsity": 0.0, "label": "qkv proj"}, 
            {"size": (1024, 1024, 1024, 5120), "row_sparsity": 0.0, "label": "attxv proj"}, 
            {"size": (1024, 5120, 5120, 20480), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            {"size": (1024, 20480, 20480, 5120), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
            # 6.7b
            # {"size": (1024, 4096, 4096, 4096), "row_sparsity": 0.0, "label": "qkv proj"}, 
            # {"size": (1024, 1024, 1024, 4096), "row_sparsity": 0.0, "label": "attxv proj"}, 
            # {"size": (1024, 4096, 4096, 16384), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            # {"size": (1024, 16384, 16384, 4096), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
            # 125m-small
            {"size": (512, 768, 768, 768), "row_sparsity": 0.0, "label": "qkv proj"}, 
            {"size": (512, 512, 512, 768), "row_sparsity": 0.0, "label": "attxv proj"}, 
            {"size": (512, 768, 768, 3072), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            {"size": (512, 3072, 3072, 768), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
            # 125m
            # {"size": (1024, 768, 768, 768), "row_sparsity": 0.0, "label": "qkv proj"}, 
            # {"size": (1024, 1024, 1024, 768), "row_sparsity": 0.0, "label": "attxv proj"}, 
            # {"size": (1024, 768, 768, 3072), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            # {"size": (1024, 3072, 3072, 768), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
            # 350m
            # {"size": (1024, 1024, 1024, 1024), "row_sparsity": 0.0, "label": "qkv proj"}, 
            # {"size": (1024, 1024, 1024, 1024), "row_sparsity": 0.0, "label": "attxv proj"}, 
            # {"size": (1024, 1024, 1024, 4096), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            # {"size": (1024, 4096, 4096, 1024), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
            #1.3b
            # {"size": (1024, 2048, 2048, 2048), "row_sparsity": 0.0, "label": "qkv proj"}, 
            # {"size": (1024, 1024, 1024, 2048), "row_sparsity": 0.0, "label": "attxv proj"}, 
            # {"size": (1024, 2048, 2048, 8192), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            # {"size": (1024, 8192, 8192, 2048), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
    ]

    mats.sort(key=lambda s: s["size"][0] * s["size"][1] * (1. - s["row_sparsity"]) * s["size"][3])
    mats.sort(key=lambda s: 1. - s["row_sparsity"])

    # chain_len_list = {str(i): [] for i in range(2, 34, 4)}
    chain_len_list = {str(i): [] for i in [2, 5, 7, 11, 17, 29, 31]}
    # chain_len_list = {"5": [], "5-10": [], "10": [], "6":[], "6-12": [], "12": [], "8":[], "8-16": [], "16": []}
    flops_vs_chain_len = chain_len_list.copy()
    util_vs_chain_len = chain_len_list.copy()
    dens_vs_chain_len = chain_len_list.copy()
    # construct shapes
    tcc_array_shapes = {}
    for len in flops_vs_chain_len.keys():
        if len.isnumeric():
            tcc_array_shapes[len] = closest_factors_to_target(int(len), 3960)
        else:
            short, long = len.split("-")
            long_shape = closest_factors_to_target(int(long), 3960)
            short_shape = (long_shape[0]*2, long_shape[1])
            tcc_array_shapes[len] = (short_shape, long_shape)
    # tcc_array_shapes = {l: closest_factors_to_target(int(l), 3960) for l in flops_vs_chain_len.keys()}
    # tcc_array_shapes = {"3": (48, 16), "6": (24, 16), "9":(16, 16), "18": (8, 16)}
    tcc_array_util_percentage = {k: (0., 0.) for k in chain_len_list.keys()}

    print(tcc_array_shapes)

    for chain_len_str in flops_vs_chain_len.keys():
        #sweep across mat size
        flops, util, density = [], [], []
        for m in mats:
            mat_a_row, mat_a_col, mat_b_row, mat_b_col = m["size"]

            # differentiate between stat/dyna chain lengths
            if chain_len_str.isnumeric():
                chain_len = int(chain_len_str)
                tcc_array_shape = tcc_array_shapes[chain_len_str]
                dsp_percentage = tcc_array_shape[0]*tcc_array_shape[1]*(chain_len+2)/3960.0
                dot_size = tcc_array_shape[0]*tcc_array_shape[1]*chain_len/3960.0
                tcc_array_util_percentage[chain_len_str] = (dot_size, dsp_percentage)
                print(f"actual size of len {chain_len_str}: {dsp_percentage:.3f}")
            else:
                short_chain_len, long_chain_len = chain_len_str.split("-")
                short_chain_len, long_chain_len = int(short_chain_len), int(long_chain_len)
                effective_a_col = mat_a_col * (1. - m["row_sparsity"])
                #select chain len based on mat a col:
                if effective_a_col <= short_chain_len * 20:
                    print(f"{chain_len_str} design selects short len")
                    tcc_array_shape = tcc_array_shapes[chain_len_str][0]
                    chain_len = short_chain_len
                else:
                    print(f"{chain_len_str} design selects long len")
                    tcc_array_shape = tcc_array_shapes[chain_len_str][1]
                    chain_len = long_chain_len

                dsp_percentage = tcc_array_shapes[chain_len_str][1][0] * \
                                    tcc_array_shapes[chain_len_str][1][1] * \
                                    (long_chain_len+2) / 3960.0
                dot_size = tcc_array_shape[0]*tcc_array_shape[1]*chain_len/3960.0
                tcc_array_util_percentage[chain_len_str] = (dot_size, dsp_percentage)
                print(f"actual size of len {chain_len_str}: {dsp_percentage:.3f}")
            
            total_util = get_util(m, tcc_array_shape, chain_len)
            # generate fake data to be sent to hw model:
            fake_data = None
            if m["row_sparsity"] > 0.0:
                dense_acol_size = ceil(mat_a_col * (1.-m["row_sparsity"]))
                fake_data = np.ones((mat_a_row, dense_acol_size))
                fake_data = np.pad(fake_data, ((0, 0), (0, mat_a_col-dense_acol_size)), "constant", constant_values = (0,))
                # print(f"padded mat size: {fake_data.shape}")
            else:
                fake_data = np.ones((mat_a_row, mat_a_col))

            curr_model = StratixDpuModel(mat_a_row, mat_a_col, mat_b_row, mat_b_col, 
                                        exp_dat=fake_data, 
                                        freq=300, 
                                        num_tcs=3960, 
                                        tcc_array_shape=tcc_array_shape, 
                                        tcc_chainlen=chain_len)

            curr_model.set_tccore_size(20)
            curr_flops, curr_lat = curr_model.tensor_fpga21_mat_sparse_flops(
                                                        fake_data, \
                                                        True, False, \
                                                        using_single_column=False,
                                                        sparse_block_size=1)
            flops.append(curr_flops)
            util.append(total_util)
            density.append(1. - m["row_sparsity"])
        
        flops_vs_chain_len[chain_len_str] = flops
        util_vs_chain_len[chain_len_str] = util
        dens_vs_chain_len[chain_len_str] = density

    matmul_size = [i["size"][0]*ceil(i["size"][1]*(1.-i["row_sparsity"]))*i["size"][3] for i in mats]

    fig, axs=plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    for idx, clen in enumerate(flops_vs_chain_len.keys()):
        # axs[0].plot(matmul_size, flops_vs_chain_len[clen], linestyle='-', 
        #         marker='s', linewidth=1, alpha=0.8, label=f"chain len={clen}", color=f"C{idx}")
        axs[0].scatter(dens_vs_chain_len[clen], flops_vs_chain_len[clen], 
                marker='o', s=24, alpha=0.6, label=f"chain len={clen}", color=f"C{idx}")
        axs[1].scatter(matmul_size, np.array(util_vs_chain_len[clen]) * tcc_array_util_percentage[clen][0], 
                marker='o', s=24, alpha=0.6, label=f"chain len={clen}", color=f"C{idx}")

    all_x = list(np.arange(0.1, 1.1, 0.1))
    all_y = []
    for idx_d in range(10):
        all_y.append(max([flops_vs_chain_len[clen][idx_d] for clen in flops_vs_chain_len.keys()]))

    coeffs = np.polyfit(all_x, all_y, 5)
    tdline = np.poly1d(coeffs)
    tdline_x = np.arange(0.1, 1.1, 0.1)
    axs[0].plot(tdline_x, tdline(tdline_x), linestyle = "--", color = "red", alpha = 0.6, linewidth=1)

    print(flops_vs_chain_len)
    peak_tops = 50.0 * (300./440.) * (3960./2387.) * (20./8.0)
    axs[0].plot(tdline_x, peak_tops * 1./tdline_x, linestyle = "--", color = "blue", alpha = 0.6, linewidth=1)
    # axs[0].axhline(y=peak_tops, linewidth=1, color='r', linestyle='--')
    axs[0].text(0.95, peak_tops+5, "theoretical", fontsize=8, va='center', ha='center')
    
    dsp_percentage_bar = axs[2].bar(tcc_array_util_percentage.keys(), 
                [v[1]*100.0 for v in tcc_array_util_percentage.values()], 
                width=0.3, 
                align="center", 
                color="C9")
    dot_percentage_bar = axs[2].bar(tcc_array_util_percentage.keys(), 
                [v[0]*100.0 for v in tcc_array_util_percentage.values()], 
                width=0.3, 
                align="center", 
                color=["C0", "C1", "C2"])
        
    axs[0].set_ylabel('TOPs')
    axs[0].set_xlim(xmin=0)
    # axs[0].set_xlim(xmax=2*1e10)
    axs[0].set_ylim(ymin=0)
    # axs[0].set_ylim(ymax=100)
    axs[0].grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    axs[0].set_xlabel('Density')
    axs[0].legend()

    axs[1].set_ylabel(r'actual MAC size/padded MAC size X TC% used for dot-product')
    axs[1].set_xlim(xmin=0)
    axs[1].set_ylim(ymin=0)
    axs[1].set_ylim(ymax=1.1)
    axs[1].grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    axs[1].set_xlabel('MAC Operations')
    axs[1].legend()

    axs[2].bar_label(dot_percentage_bar, labels=[f"{l:.1f}" for l in dot_percentage_bar.datavalues])
    axs[2].bar_label(dsp_percentage_bar, labels=[f"{l:.1f}" for l in dsp_percentage_bar.datavalues])
    axs[2].set_xlabel('Chain length')
    axs[2].set_ylabel(r'TC% used for dot-product')
    axs[2].set_ylim(ymin=0)
    # axs[1].set_ylim(ymax=105)
    axs[2].grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    fig.tight_layout()
    fig.savefig("res_fig/density_sweep_chainlen_fixed_totaldsp_prime.pdf")
    # pickle.dump(fig, open('res_fig/matsize_sweep_chainlen.fig.pickle', 'wb'))
    plt.cla()

def max_eff_util_chainlen(workload):
    '''
    find out the chain length with max util under a given workload
    '''
    chain_len_lst = list(range(1, 32, 1))

    curr_max_util, curr_max_util_clen = 0, 0
    for clen in chain_len_lst:
        hw_size = closest_factors_to_target(clen, 3960)
        frag_util = get_util(workload, hw_size, clen)
        stat_util = hw_size[0]*hw_size[1]*clen/3960.0
        curr_util = frag_util * stat_util
        if curr_util > curr_max_util:
            curr_max_util, curr_max_util_clen = curr_util, clen
    
    return curr_max_util_clen

def multi_mat_exec_tops(
        tc_len: int,
        tc_array_shape: tuple,
        mat_sizes: List[dict],
        freq: int = 300,
):
    lat, util, density, ops = [], [], [], []
    for m in mat_sizes:
        mat_a_row, mat_a_col, mat_b_row, mat_b_col = m["size"]

        total_util = get_util(m, tc_array_shape, tc_len)
        # generate fake data to be sent to hw model:
        fake_data = None
        if m["row_sparsity"] > 0.0: 
            dense_acol_size = ceil(mat_a_col * (1.-m["row_sparsity"]))
            fake_data = np.ones((mat_a_row, dense_acol_size))
            fake_data = np.pad(
                fake_data, ((0, 0), (0, mat_a_col-dense_acol_size)), "constant", constant_values=(0,))
            # print(f"padded mat size: {fake_data.shape}")
        else:
            fake_data = np.ones((mat_a_row, mat_a_col))

        curr_model = StratixDpuModel(mat_a_row, mat_a_col, mat_b_row, mat_b_col,
                                        exp_dat=fake_data,
                                        freq=300,
                                        num_tcs=3960,
                                        tcc_array_shape=tc_array_shape,
                                        tcc_chainlen=tc_len)

        curr_model.set_tccore_size(20)
        _, curr_lat = curr_model.tensor_fpga21_mat_sparse_flops(
            fake_data,
            True, False,
            using_single_column=False,
            sparse_block_size=1)
        
        curr_ops = m["size"][0] * m["size"][3] * m["size"][1] * ceil(1. - m["row_sparsity"])

        lat.append(curr_lat)
        ops.append(curr_ops)
        util.append(total_util)
        density.append(1. - m["row_sparsity"])

    total_ops = sum(ops) * 2
    time_latency = sum(lat) * 1./freq * 1e-6

    return time_latency, total_ops

def tcchain_len_multidualcore(small_mats, large_mats, dsp_split=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    def get_mats_ops(mats: list):
        print(f"computing mat list with size {len(mats)}")
        mat_ops_lst = []
        for m in mats:
            ops = m["size"][0] * m["size"][3] * m["size"][1] * ceil(1. - m["row_sparsity"])
            mat_ops_lst.append(ops)

        return sum(mat_ops_lst)

    small_mats.sort(key=lambda s: 1. - s["row_sparsity"])
    large_mats.sort(key=lambda s: 1. - s["row_sparsity"])
    small_mats_ops = get_mats_ops(small_mats)
    large_mats_ops = get_mats_ops(large_mats)
    print(f"small vs. large: {small_mats_ops/large_mats_ops}")

    short_chain_len, long_chain_len = 0, 0
    # find out short chain len with max util on att head
    for smat in small_mats:
        if smat["label"] == "att":
            short_chain_len = max_eff_util_chainlen(smat)
    # find out long chain len with max util on dense_ffn
    for lmat in large_mats:
        if lmat["label"] == "dense_ffn_fc2":
            long_chain_len = max_eff_util_chainlen(lmat)

    print(f"short len selected: {short_chain_len}, long len selected: {long_chain_len}")

    short_chain_lat, long_chain_lat = 0, 0
    short_chain_ops, long_chain_ops = 0, 0
    if dsp_split is None:
        # search best split that evenly distributes small and large latency
        small_split_l, small_split_r = 0.000, 1.000
        small_split = (small_split_l + small_split_r) / 2.0

        short_chain_lat, long_chain_lat = 0., 1.

        while(abs(long_chain_lat-short_chain_lat) > 0.001):
            short_chain_shape = closest_factors_to_target(short_chain_len, ceil(3960.0 * small_split))
            long_chain_shape = closest_factors_to_target(long_chain_len, 3960-ceil(3960.0 * small_split))
            short_chain_lat, short_chain_ops = \
                multi_mat_exec_tops(short_chain_len, short_chain_shape, small_mats)
            long_chain_lat, long_chain_ops = \
                multi_mat_exec_tops(long_chain_len, long_chain_shape, large_mats)
            
            if (short_chain_lat - long_chain_lat) > 0.0:
                small_split_l = small_split
            elif (short_chain_lat - long_chain_lat) < 0.0:
                small_split_r = small_split
            
            if abs(long_chain_lat-short_chain_lat) > 0.001:
                small_split = (small_split_l + small_split_r) / 2.0
            
            print(f"l: {small_split_l}, r: {small_split_r}, next sel: {small_split}")
            print(f"abs diff: {long_chain_lat-short_chain_lat}")

        dsp_split = (small_split, 1.- small_split)
    else:
        # construct shapes
        short_chain_shape = closest_factors_to_target(short_chain_len, ceil(3960.0 * dsp_split[0]))
        long_chain_shape = closest_factors_to_target(long_chain_len, 3960-ceil(3960.0 * dsp_split[0]))

        short_chain_lat, short_chain_ops = \
            multi_mat_exec_tops(short_chain_len, short_chain_shape, small_mats)
        long_chain_lat, long_chain_ops = \
            multi_mat_exec_tops(long_chain_len, long_chain_shape, large_mats)
    
    print(f"short lat: {short_chain_lat}, long lat: {long_chain_lat}")

    lat_res = max(short_chain_lat, long_chain_lat)
    total_ops = short_chain_ops + long_chain_ops
    flops = total_ops / lat_res / 1e12

    print(f"total flops: {flops}")

    return flops, (short_chain_lat, long_chain_lat)

def tcchain_len_dynacore(small_mats, large_mats):

    small_mats.sort(key=lambda s: 1. - s["row_sparsity"])
    large_mats.sort(key=lambda s: 1. - s["row_sparsity"])

    short_chain_len, long_chain_len = 7, 14
    # construct shapes
    long_chain_shape = closest_factors_to_target(long_chain_len, 3960)
    short_chain_shape = (long_chain_shape[0]*2, long_chain_shape[1])
    
    short_chain_lat, short_chain_ops = \
        multi_mat_exec_tops(short_chain_len, short_chain_shape, small_mats)
    long_chain_lat, long_chain_ops = \
        multi_mat_exec_tops(long_chain_len, long_chain_shape, large_mats)
    
    print(f"short lat: {short_chain_lat}, long lat: {long_chain_lat}")

    lat_res = short_chain_lat + long_chain_lat
    total_ops = short_chain_ops + long_chain_ops
    flops = total_ops / lat_res / 1e12

    print(f"total flops: {flops}")

    return flops, (short_chain_lat, long_chain_lat)

def explore_dualcore_lat_thrput(small_mats, large_mats):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    dsp_split_lst = [(s, 1.-s) for s in np.arange(0.05, 1.0, 0.05)]

    tops_res, short_lat_res, long_lat_res = [], [], []
    for s in dsp_split_lst:
        tops, (short_chain_lat, long_chain_lat) = tcchain_len_multidualcore(small_mats, large_mats, dsp_split=s)
        tops_res.append(tops)
        short_lat_res.append(short_chain_lat)
        long_lat_res.append(long_chain_lat)

    short_ratio = [i[0] for i in dsp_split_lst]

    ax.scatter(short_ratio, tops_res, 
                marker='o', s=24, alpha=0.6, color=f"C0")
    ax2 = ax.twinx()
    ax2.bar(short_ratio, 
            short_lat_res, 
            width=-0.02, 
            align="edge", 
            color="C1")
    ax2.bar(short_ratio, 
            long_lat_res, 
            width=0.02, 
            align="edge", 
            color="C2")
    
    ax.set_ylabel('TOPs')
    ax2.set_ylabel('latency/s')
    ax.set_xlim(xmin=0)
    ax2.set_xlim(xmin=0)
    # ax[0].set_xlim(xmax=2*1e10)
    ax.set_ylim(ymin=-0.02)
    # ax[0].set_ylim(ymax=100)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    ax.set_xlabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig("res_fig/explore_split_1.3b_bestutil.pdf")
    plt.cla()
    pass

def tops_diff_mats(small_mats_list, large_mats_list):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    dual_core_tops_list = {i: 0 for i in small_mats_list.keys()}
    dyna_core_tops_list = {i: 0 for i in small_mats_list.keys()}

    for model_name in small_mats_list.keys():
        dualcore_tops, _ = tcchain_len_multidualcore(small_mats_list[model_name], large_mats_list[model_name])
        dual_core_tops_list[model_name] = dualcore_tops
        dynacore_tops, _ = tcchain_len_dynacore(small_mats_list[model_name], large_mats_list[model_name])
        dyna_core_tops_list[model_name] = dynacore_tops

    ax.scatter(list(small_mats_list.keys()), list(dual_core_tops_list.values()), 
                marker='o', s=24, alpha=0.6, label=f"dual core", color=f"C0")
    ax.scatter(list(small_mats_list.keys()), list(dyna_core_tops_list.values()), 
                marker='o', s=24, alpha=0.6, label=f"dyna core", color=f"C1")
    
    ax.set_ylabel('TOPs')
    ax.set_xlim(xmin=0)
    # ax[0].set_xlim(xmax=2*1e10)
    ax.set_ylim(ymin=0)
    # ax[0].set_ylim(ymax=100)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    ax.set_xlabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig("res_fig/diff_model_new.pdf")
    plt.cla()

def main():
    # tcchain_len_sweep()
    # tcchain_r_c_sweep()
    small_mats = {\
        "opt-13b":
            [{"size": (1024, 1024, 1024, 128), "row_sparsity": 0.2, "label": "attxv"}] * 40 + \
            [{"size": (1024, 128, 128, 1024), "row_sparsity": 0.0, "label": "att"}] * 40,
        "opt-1.3b":
            [{"size": (1024, 1024, 1024, 64), "row_sparsity": 0.2, "label": "attxv"}] * 24 + \
            [{"size": (1024, 64, 64, 1024), "row_sparsity": 0.0, "label": "att"}] * 24, 
        "opt-350m":
            [{"size": (1024, 1024, 1024, 64), "row_sparsity": 0.2, "label": "attxv"}] * 16 + \
            [{"size": (1024, 64, 64, 1024), "row_sparsity": 0.0, "label": "att"}] * 16, 
        }
    
    large_mats = {\
        "opt-13b":
            [{"size": (1024, 5120, 5120, 5120), "row_sparsity": 0.0, "label": "qkv proj"}] * 3 + \
            [{"size": (1024, 5120, 5120, 20480), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}] + \
            [{"size": (1024, 20480, 20480, 5120), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}],
        "opt-1.3b":
            [{"size": (1024, 2048, 2048, 2048), "row_sparsity": 0.0, "label": "qkv proj"}] * 3 + \
            [{"size": (1024, 2048, 2048, 8192), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}] + \
            [{"size": (1024, 8192, 8192, 2048), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}],
        "opt-350m":
            [{"size": (1024, 1024, 1024, 1024), "row_sparsity": 0.0, "label": "qkv proj"}] * 3 + \
            [{"size": (1024, 1024, 1024, 4096), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}] + \
            [{"size": (1024, 4096, 4096, 1024), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}]
        }

    tops_diff_mats(small_mats, large_mats)
    # explore_dualcore_lat_thrput(small_mats["opt-1.3b"], large_mats["opt-1.3b"])

if __name__ == "__main__":
    main()
