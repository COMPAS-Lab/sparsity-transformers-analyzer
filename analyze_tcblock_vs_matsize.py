
import torch
import matplotlib
import numpy as np
import hw_modeling
import pandas as pd
import matplotlib.pyplot as plt
import skimage.measure
from scipy.spatial.distance import hamming
from typing import List, Union
from math import ceil, floor, sqrt
import random
import multiprocessing
from hw_modeling import StratixDpuModel
from sparse_tensor_analyzer import factor_int

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

def get_util(workload, hw_size, chain_len):
    mat_a_row, mat_a_col, _, mat_b_col = workload
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

def tcchain_len_sweep():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    sizes = [(1024, 240, 240, 1024), 
              (1024, 2048, 2048, 1024), 
              (1024, 2048, 2048, 8192), 
              (1024, 8192, 8192, 2048)]
    
    # sizes = [(2**i, 2**i, 2**i, 2**i) for i in [6, 7, 8, 10, 11, 14, 16]]

    flops_vs_chain_len = {"3": [], "6": [], "9":[], "18": []}
    util_vs_chain_len = {"3": [], "6": [], "9":[], "18": []}
    # tcc_array_shapes = {l: closest_factors_to_target(int(l), 2560) for l in flops_vs_chain_len.keys()}
    tcc_array_shapes = {"3": (48, 16), "6": (24, 16), "9":(16, 16), "18": (8, 16)}
    tcc_array_dot_percentage = {"3": 0., "6": 0., "9":0., "18": 0.}

    print(tcc_array_shapes)

    for chain_len_str in flops_vs_chain_len.keys():
        chain_len = int(chain_len_str)
        tcc_array_shape = tcc_array_shapes[chain_len_str]
        dsp_percentage = tcc_array_shape[0]*tcc_array_shape[1]*(chain_len+2)/3960.0
        dot_size = tcc_array_shape[0]*tcc_array_shape[1]*chain_len/3960.0
        tcc_array_dot_percentage[chain_len_str] = dot_size
        print(f"actual size of len {chain_len_str}: {dsp_percentage:.3f}")

        #sweep across mat size
        flops, util = [], []
        for s in sizes:
            mat_a_row, mat_a_col, mat_b_row, mat_b_col = s
            total_util = get_util(s, tcc_array_shape, chain_len)

            fake_dense_data = np.ones((mat_a_row, mat_a_col))
            curr_model = StratixDpuModel(mat_a_row, mat_a_col, mat_b_row, mat_b_col, 
                                        exp_dat=fake_dense_data, 
                                        freq=300, 
                                        num_tcs=3960, 
                                        tcc_array_shape=tcc_array_shape, 
                                        tcc_chainlen=chain_len)

            curr_model.set_tccore_size(20)
            curr_flops, curr_lat = curr_model.tensor_fpga21_mat_sparse_flops(
                                                        fake_dense_data, \
                                                        True, False, \
                                                        using_single_column=False,
                                                        sparse_block_size=1)
            flops.append(curr_flops)
            util.append(total_util)
        
        flops_vs_chain_len[chain_len_str] = flops
        util_vs_chain_len[chain_len_str] = util

    matmul_size = [i[0]*i[1]*i[3] for i in sizes]

    fig, axs=plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    for idx, clen in enumerate(flops_vs_chain_len.keys()):
        axs[0].plot(matmul_size, flops_vs_chain_len[clen], linestyle='-', 
                marker='s', linewidth=1, alpha=0.8, label=f"chain len={clen}", color=f"C{idx}")
    
    dot_percentage_bar = axs[1].bar(tcc_array_dot_percentage.keys(), 
                [v*100.0 for v in tcc_array_dot_percentage.values()], 
                width=0.3, 
                align="center", 
                color=[f"C{idx}" for idx in range(len(tcc_array_dot_percentage))])
        
    axs[0].set_ylabel('TOPs')
    axs[0].set_xlim(xmin=0)
    axs[0].set_ylim(ymin=0)
    axs[0].set_ylim(ymax=90)
    axs[0].grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    axs[0].set_xlabel('MAC Operations')
    axs[0].legend()
    axs[1].bar_label(dot_percentage_bar, labels=[f"{l:.1f}" for l in dot_percentage_bar.datavalues])
    axs[1].set_xlabel('Chain length')
    axs[1].set_ylabel(r'TC% used for dot-product')
    axs[1].set_ylim(ymin=0)
    axs[1].set_ylim(ymax=105)
    axs[1].grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    fig.tight_layout()
    fig.savefig("res_fig/matsize_sweep_chainlen_dynachange.pdf")
    plt.cla()

def main():
    tcchain_len_sweep()
    # tcchain_r_c_sweep()

if __name__ == "__main__":
    main()