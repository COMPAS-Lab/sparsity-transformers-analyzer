
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

def tcchain_len_sweep():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fsize = 9
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3
    
    mats = [{"size": (1024, 2048, 2048, 1024), "row_sparsity": frac, "label": "sparse_attn_h"} \
            for frac in np.arange(0.2, 1.0, 0.2)
            ]
    
    mats += [
            #13b
            # {"size": (1024, 5120, 5120, 5120), "row_sparsity": 0.0, "label": "qkv proj"}, 
            # {"size": (1024, 1024, 1024, 5120), "row_sparsity": 0.0, "label": "attxv proj"}, 
            # {"size": (1024, 5120, 5120, 20480), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            # {"size": (1024, 20480, 20480, 5120), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
            # 6.7b
            # {"size": (1024, 4096, 4096, 4096), "row_sparsity": 0.0, "label": "qkv proj"}, 
            # {"size": (1024, 1024, 1024, 4096), "row_sparsity": 0.0, "label": "attxv proj"}, 
            # {"size": (1024, 4096, 4096, 16384), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            # {"size": (1024, 16384, 16384, 4096), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
            # 125m
            {"size": (1024, 768, 768, 768), "row_sparsity": 0.0, "label": "qkv proj"}, 
            {"size": (1024, 1024, 1024, 768), "row_sparsity": 0.0, "label": "attxv proj"}, 
            {"size": (1024, 768, 768, 3072), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            {"size": (1024, 3072, 3072, 768), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
            # 350m
            {"size": (1024, 1024, 1024, 1024), "row_sparsity": 0.0, "label": "qkv proj"}, 
            {"size": (1024, 1024, 1024, 1024), "row_sparsity": 0.0, "label": "attxv proj"}, 
            {"size": (1024, 1024, 1024, 4096), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            {"size": (1024, 4096, 4096, 1024), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
            #1.3b
            {"size": (1024, 2048, 2048, 2048), "row_sparsity": 0.0, "label": "qkv proj"}, 
            {"size": (1024, 1024, 1024, 2048), "row_sparsity": 0.0, "label": "attxv proj"}, 
            # {"size": (1024, 2048, 2048, 8192), "row_sparsity": 0.0, "label": "dense_ffn_fc1"}, 
            # {"size": (1024, 8192, 8192, 2048), "row_sparsity": 0.0, "label": "dense_ffn_fc2"}, 
    ]

    mats.sort(key=lambda s: s["size"][0] * s["size"][1] * (1. - s["row_sparsity"]) * s["size"][3])

    chain_len_list = {str(i): [] for i in range(2, 20, 4)}
    flops_vs_chain_len = chain_len_list.copy()
    util_vs_chain_len = chain_len_list.copy()
    tcc_array_shapes = {l: closest_factors_to_target(int(l), 3960) for l in flops_vs_chain_len.keys()}
    # tcc_array_shapes = {"3": (48, 16), "6": (24, 16), "9":(16, 16), "18": (8, 16)}
    tcc_array_util_percentage = {k: (0., 0.) for k in chain_len_list.keys()}

    print(tcc_array_shapes)

    for chain_len_str in flops_vs_chain_len.keys():
        chain_len = int(chain_len_str)
        tcc_array_shape = tcc_array_shapes[chain_len_str]
        dsp_percentage = tcc_array_shape[0]*tcc_array_shape[1]*(chain_len+2)/3960.0
        dot_size = tcc_array_shape[0]*tcc_array_shape[1]*chain_len/3960.0
        tcc_array_util_percentage[chain_len_str] = (dot_size, dsp_percentage)
        print(f"actual size of len {chain_len_str}: {dsp_percentage:.3f}")

        #sweep across mat size
        flops, util = [], []
        for m in mats:
            mat_a_row, mat_a_col, mat_b_row, mat_b_col = m["size"]
            total_util = get_util(m, tcc_array_shape, chain_len)
            # generate fake data to be sent to hw model:
            fake_data = None
            if m["row_sparsity"] > 0.0:
                dense_acol_size = ceil(mat_a_col * (1.-m["row_sparsity"]))
                fake_data = np.ones((mat_a_row, dense_acol_size))
                fake_data = np.pad(fake_data, ((0, 0), (0, mat_a_col-dense_acol_size)), "constant", constant_values = (0,))
                print(f"padded mat size: {fake_data.shape}")
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
        
        flops_vs_chain_len[chain_len_str] = flops
        util_vs_chain_len[chain_len_str] = util

    matmul_size = [i["size"][0]*ceil(i["size"][1]*(1.-i["row_sparsity"]))*i["size"][3] for i in mats]

    fig, axs=plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    for idx, clen in enumerate(flops_vs_chain_len.keys()):
        axs[0].plot(matmul_size, flops_vs_chain_len[clen], linestyle='-', 
                marker='s', linewidth=1, alpha=0.8, label=f"chain len={clen}", color=f"C{idx}")
        axs[1].plot(matmul_size, np.array(util_vs_chain_len[clen]) * tcc_array_util_percentage[clen][0], linestyle='-', 
                marker='s', linewidth=1, alpha=0.8, label=f"chain len={clen}", color=f"C{idx}")
    
    print(flops_vs_chain_len)
    peak_tops = 50.0 * (300./440.) * (3960./2387.) * (20./8.0)
    axs[0].axhline(y=peak_tops, linewidth=1, color='r', linestyle='--', )
    axs[0].text(max(matmul_size), peak_tops+5, "theoratical", fontsize=8, va='center', ha='center')
    
    dsp_percentage_bar = axs[2].bar(tcc_array_util_percentage.keys(), 
                [v[1]*100.0 for v in tcc_array_util_percentage.values()], 
                width=0.3, 
                align="center", 
                color="C9")
    dot_percentage_bar = axs[2].bar(tcc_array_util_percentage.keys(), 
                [v[0]*100.0 for v in tcc_array_util_percentage.values()], 
                width=0.3, 
                align="center", 
                color=[f"C{idx}" for idx in range(len(tcc_array_util_percentage))])
        
    axs[0].set_ylabel('TOPs')
    axs[0].set_xlim(xmin=0)
    # axs[0].set_xlim(xmax=2*1e10)
    axs[0].set_ylim(ymin=0)
    # axs[0].set_ylim(ymax=100)
    axs[0].grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    axs[0].set_xlabel('MAC Operations')
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
    fig.savefig("res_fig/matsize_sweep_chainlen_fixed_totaldsp_lesshwtypes.pdf")
    # pickle.dump(fig, open('res_fig/matsize_sweep_chainlen.fig.pickle', 'wb'))
    plt.cla()

def main():
    tcchain_len_sweep()
    # tcchain_r_c_sweep()

if __name__ == "__main__":
    main()