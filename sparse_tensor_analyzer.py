from audioop import mul
from bz2 import compress
from importlib.metadata import files
from os import minor
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

def plot_multi(
    data: pd.DataFrame,
    x: Union[str, None] = None,
    y: Union[List[str], None] = None,
    spacing: float = 0.1,
    **kwargs
) -> matplotlib.axes.Axes:
    """Plot multiple Y axes on the same chart with same x axis.

    Args:
        data: dataframe which contains x and y columns
        x: column to use as x axis. If None, use index.
        y: list of columns to use as Y axes. If None, all columns are used
            except x column.
        spacing: spacing between the plots
        **kwargs: keyword arguments to pass to data.plot()

    Returns:
        a matplotlib.axes.Axes object returned from data.plot()

    Example:
    >>> plot_multi(df, figsize=(22, 10))
    >>> plot_multi(df, x='time', figsize=(22, 10))
    >>> plot_multi(df, y='price qty value'.split(), figsize=(22, 10))
    >>> plot_multi(df, x='time', y='price qty value'.split(), figsize=(22, 10))
    >>> plot_multi(df[['time price qty'.split()]], x='time', figsize=(22, 10))

    See Also:
        This code is mentioned in https://stackoverflow.com/q/11640243/2593810
    """
    from pandas.plotting._matplotlib.style import get_standard_colors

    # Get default color style from pandas - can be changed to any other color list
    if y is None:
        y = data.columns

    # remove x_col from y_cols
    if x:
        y = [col for col in y if col != x]

    if len(y) == 0:
        return
    colors = get_standard_colors(num_colors=len(y))

    if "legend" not in kwargs:
        kwargs["legend"] = False  # prevent multiple legends

        # First axis
    ax = data.plot(x=x, y=y[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=y[0])
    lines, labels = ax.get_legend_handles_labels()

    for i in range(1, len(y)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines["right"].set_position(("axes", 1 + spacing * (i - 1)))
        data.boxplot(
            ax=ax_new, x=x, y=y[i], color=colors[i % len(colors)], **kwargs
        )
        ax_new.set_ylabel(ylabel=y[i])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax

def factor_int(n: int):
    val = ceil(sqrt(float(n)))
    val2 = int(n/val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n/val)

    if val < val2: val, val2 = val2, val
    return val, val2

## find optimized chain_len of dense matmul
## deprecated
# chain_len = 1
# max_flops = 0.0
# for l in range(1, 35):
#     base_model = hw_modeling.StratixDpuModel(384, 384, 384, 768, \
#                                         freq=400, num_tcs=3960, tcc_chainlen=l)
#     curr_flops, _ = base_model.tensor_fpga21_mat_flops(l, False, False)
#     if curr_flops > max_flops:
#         max_flops = curr_flops
#         chain_len = l
#     elif curr_flops == max_flops and chain_len < l:
#         chain_len = l

# print(max_flops, chain_len)
# hw_array_shape = (22, 12)

def get_mat_sparsity(dat):
    if type(dat) == torch.Tensor:
        return (1. - torch.count_nonzero(dat) / torch.numel(dat))
    else:
        return (1. - np.count_nonzero(dat) / dat.size)

def prepare_mat_dat(data_path, seq_len_path, seq_len_range, samples=-1):
    bfp_att_probes = []
    if seq_len_path is not None and seq_len_range is not None:
        seq_len = torch.load(seq_len_path).cpu().detach().numpy()
        seq_len = np.squeeze(seq_len.astype(int))
        filtered_seq_len_idx = np.where((seq_len <= seq_len_range[1]) & (seq_len > seq_len_range[0]))[0]
        bfp_att_probes_ori = torch.load(data_path).cpu().detach().numpy()
        for i in filtered_seq_len_idx:
            heads = np.squeeze(bfp_att_probes_ori[i,:,:seq_len[i],:seq_len[i]])
            bfp_att_probes += list(heads)
    else:
        data = torch.load(data_path).cpu().detach().numpy()
        bfp_att_probes = list(data.reshape(-1, data.shape[-2], data.shape[-1]))

    if samples > 0:
        bfp_att_probes = random.sample(bfp_att_probes, samples)

    return bfp_att_probes

def compute_matmul_performance(data_path, chain_len, out_w, hw_array_shape, \
                                sort_row_sparsity, using_single_column, sparse_block_size, \
                                seq_len_path=None, seq_len_range=None, mixed_chain_length=None, \
                                short_to_long_ratio=0.0, blocked_pruning=False):

    bfp_att_probes = prepare_mat_dat(data_path, seq_len_path, seq_len_range)
    res = {"latency":[] ,"sparsity": [], "s2l ratio": [], "tp": []}

    total_sparse_lat, total_base_lat, total_min_sparse_lat = 0, 0, 0
    num_insts = len(bfp_att_probes)
    for exps in bfp_att_probes:
        # use a dense mat to calculate dens mat base lat
        fake_dense_data = np.ones(exps.shape)
        base_model = hw_modeling.StratixDpuModel(exps.shape[0], exps.shape[1], exps.shape[1], out_w, \
                                        exp_dat=fake_dense_data, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                        tcc_chainlen=chain_len)
        base_model.set_tccore_size(20)
        base_flops, base_lat = base_model.tensor_fpga21_mat_sparse_flops(fake_dense_data, \
                                                    sort_row_sparsity, False, \
                                                    using_single_column=False, sparse_block_size=sparse_block_size)

        # evaluate sparse model
        sparse_model = hw_modeling.StratixDpuModel(exps.shape[0], exps.shape[1], exps.shape[1], out_w, \
                                        exp_dat=exps, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                        tcc_chainlen=chain_len)
        sparse_model.set_tccore_size(20)
        min_sparse_flops, min_sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(exps, \
                                                    sort_row_sparsity, True, False, sparse_block_size, True)
        if mixed_chain_length is None:
            sparse_flops, sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(exps, \
                                                        sort_row_sparsity, False, \
                                                        using_single_column, sparse_block_size, \
                                                        blocked_pruning=blocked_pruning)
        else:
            sparse_model = hw_modeling.StratixDpuModel(exps.shape[0], exps.shape[1], exps.shape[1], out_w, \
                                            exp_dat=exps, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                            tcc_chainlen=mixed_chain_length)
            sparse_model.set_tccore_size(20)
            sparse_flops, sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(exps, \
                                                        sort_row_sparsity, False, \
                                                        using_single_column, sparse_block_size, False, short_to_long_ratio, \
                                                        blocked_pruning=blocked_pruning)
                                                    
        curr_sparsity = 1. - np.count_nonzero(exps) / exps.size

        total_sparse_lat += sparse_lat
        total_base_lat += base_lat
        total_min_sparse_lat += min_sparse_lat
        res["latency"].append(sparse_lat/base_lat)
        res["sparsity"].append(curr_sparsity)
        res["tp"].append(sparse_flops)
        res["s2l ratio"].append(curr_sparsity * 1000 / sparse_lat)
                
    res_df = pd.DataFrame(res, columns=res.keys())
    res_df.sort_values(by=["sparsity"], inplace=True)
    return res_df, total_sparse_lat/num_insts, total_min_sparse_lat/num_insts, total_base_lat/num_insts

def plot_perf_sparsity(data_path, output_path, chain_len_list, hw_array_shape_list, sort_row_sparsity, \
                            using_single_column, sparse_block_size, \
                            seq_len_path = None, seq_len_range = None, \
                            attached_to_fig_name="", mixed_short_chain_len = 0.0, short_to_long_ratio = 0.0, \
                            plot_latency_by_insts = False, blocked_pruning=False):
    res_df_list, sparselat_list, ideallat_list, baselat_list = [], [], [], []
    sorted_fig_path = "_sorted" if sort_row_sparsity else "_unsorted"

    for chain_len, hw_array_shape in zip(chain_len_list, hw_array_shape_list):
        if mixed_short_chain_len > 0:
            mixed_chain_len = (mixed_short_chain_len, chain_len)
        else:
            mixed_chain_len = None

        res_df, sparse_lat, ideal_lat, base_lat = compute_matmul_performance(data_path, chain_len, 768, \
                                                            hw_array_shape, sort_row_sparsity, \
                                                            using_single_column, sparse_block_size, 
                                                            seq_len_path, seq_len_range, \
                                                            mixed_chain_len, \
                                                            short_to_long_ratio, blocked_pruning)
        res_df_list.append(res_df)
        sparselat_list.append(sparse_lat)
        baselat_list.append(base_lat)
        ideallat_list.append(ideal_lat)
    
    if plot_latency_by_insts:
    # plot latency vs sparsity
        for idx, (chain_len, res_df, sparse_lat, base_lat) in \
                enumerate(zip(chain_len_list, res_df_list, sparselat_list, baselat_list)):
            plt.scatter(x=res_df["sparsity"], y=res_df["latency"], alpha=0.6, \
                                linewidth=0.1, linestyle='-', marker='s', color=f"C{idx}", label=f"chain len={chain_len}")
            plt.axhline(sparse_lat/base_lat, linestyle='--', color=f'C{idx}', alpha=0.5)
            plt.axvline(res_df["sparsity"].mean(), linestyle='--', color='black', alpha=0.5)


        plt.title(f"latency vs. sparsity{attached_to_fig_name}")
        plt.ylim(0.0, 1.1)
        plt.xlabel("sparsity")
        plt.ylabel("sparse latency/dense latency")
        plt.legend()
        plt.grid(linewidth=0.3)
        plt.savefig(output_path + "lat_sparsity" + sorted_fig_path + attached_to_fig_name + ".pdf")
        plt.clf()

        # plot throughput vs sparsity
        for idx, (chain_len, res_df) in enumerate(zip(chain_len_list, res_df_list)):
            plt.scatter(x=res_df["sparsity"], y=res_df["tp"], alpha=0.6, \
                                linewidth=0.1, linestyle='-', marker='s', color=f"C{idx}", label=f"chain len={chain_len}")

        plt.title(f"throughput vs. sparsity{attached_to_fig_name}")
        plt.xlabel("sparsity")
        plt.ylabel("throughput (TFLOPs)")
        plt.legend()
        plt.grid(linewidth=0.3)
        plt.savefig(output_path + "tp_sparsity" + sorted_fig_path + attached_to_fig_name + ".pdf")
        plt.clf()

    print("layer file: ", data_path)
    print("array shape: ", hw_array_shape_list)
    for chain_len, sparse_lat, ideal_lat, base_lat in \
            zip(chain_len_list, sparselat_list, ideallat_list, baselat_list):
        print(f"chain len: {chain_len}, \
                    sparse lat: {sparse_lat:.2f}, \
                    ideal lat: {ideal_lat:.2f}")

    return sparselat_list, ideallat_list, baselat_list

def short_chain_len_profiling(data_path_list, seq_len_list, seq_len_range):
    res_dat = []
    for data_path, seq_len_path in zip(data_path_list, seq_len_list):
        bfp_data = prepare_mat_dat(data_path, seq_len_path, seq_len_range)
        for mat in bfp_data:
            # set sparsity bar
            if get_mat_sparsity(mat) > 0.7:
                none_zeros_row = np.count_nonzero(mat, axis=-1).tolist()
                res_dat+=(none_zeros_row)

    avg_none_zeros = np.mean(res_dat)
    min_len = ceil(avg_none_zeros / 20)
    print(f"avg none zeros: {avg_none_zeros}, min len: {min_len}")
    print(f"quantiles: Q1: {np.quantile(res_dat, 0.25)}," + \
                        f"Q2: {np.quantile(res_dat, 0.5)}," + \
                        f"Q3: {np.quantile(res_dat, 0.75)}," + \
                        f"max: {np.amax(res_dat)}")
    return ceil(np.quantile(res_dat, 0.75) / 20.0)

def plot_perf_sparsity_simple_arglist (data_path, seq_len_path, s2l_ratio=0.3, fixed_chain_len = True, short_chain_len=2):
    if fixed_chain_len:
        chain_len_list = [short_chain_len, 10, 20, 34]
        hw_array_shape_list = [factor_int(floor(3960.0/(i+2.0))) for i in chain_len_list]
        return plot_perf_sparsity(data_path = data_path, output_path="./res_fig/", 
                                chain_len_list = chain_len_list, \
                                hw_array_shape_list = hw_array_shape_list, \
                                sort_row_sparsity=True, \
                                using_single_column=False, sparse_block_size=1, \
                                seq_len_path=seq_len_path, seq_len_range=(0, 2048), \
                                plot_latency_by_insts=False, blocked_pruning=False)
    else:
        chain_len_list = [10, 20, 34]
        hw_array_shape_list = [factor_int(floor(3960.0/(i+2.0))) for i in chain_len_list]
        return plot_perf_sparsity(data_path = data_path, output_path="./res_fig/", \
                                chain_len_list=chain_len_list, \
                                hw_array_shape_list=hw_array_shape_list, \
                                sort_row_sparsity=True, \
                                using_single_column=False, sparse_block_size=1, \
                                seq_len_path=seq_len_path, seq_len_range=(0, 2048), \
                                plot_latency_by_insts=False, \
                                mixed_short_chain_len=short_chain_len, \
                                short_to_long_ratio=s2l_ratio, blocked_pruning=False)

def compute_selfatt_layer_perf(data_path, seq_len, chain_len, hw_array_shape, sort_row_sparsity, layer_idx=0):
    bfp_att_path = data_path + "act/bfp_attprobs/"
    weights_path = data_path + "weights/"
    att = torch.load(bfp_att_path + f"{layer_idx}-0.pt").cpu().detach().numpy()
    wk = torch.load(weights_path + f"key-{layer_idx}.pt").cpu().detach().numpy()
    wq = torch.load(weights_path + f"query-{layer_idx}.pt").cpu().detach().numpy()
    wv = torch.load(weights_path + f"value-{layer_idx}.pt").cpu().detach().numpy()
    wfc = torch.load(weights_path + f"fc1-{layer_idx}.pt").cpu().detach().numpy()
    print(f"wk size: {wk.shape}, sparsity: {get_mat_sparsity(wk)}")
    print(f"wq size: {wq.shape}, sparsity: {get_mat_sparsity(wq)}")
    print(f"wv size: {wv.shape}, sparsity: {get_mat_sparsity(wv)}")
    print(f"wfc size: {wfc.shape}, sparsity: {get_mat_sparsity(wfc)}")

    curr_latency = 0
    total_base_lat = 0

    # latency for q k v computing
    for wei in [wk, wq, wv]:
        base_model = hw_modeling.StratixDpuModel(wei.shape[0], wei.shape[1], wei.shape[1], seq_len, \
                                        freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                        tcc_chainlen=chain_len)
        sparse_model = hw_modeling.StratixDpuModel(wei.shape[0], wei.shape[1], wei.shape[1], seq_len, \
                                        exp_dat=wei, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                        tcc_chainlen=chain_len)

        base_flops, base_lat = base_model.tensor_fpga21_mat_flops(chain_len, False, False)
        sparse_flops, sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(wei,sort_rows_by_sparsity=sort_row_sparsity, ideal=False)

        curr_sparsity = get_mat_sparsity(wei)
        curr_latency += sparse_lat
        total_base_lat += base_lat

    # dense computation
    for h_idx in range(12):
        base_model = hw_modeling.StratixDpuModel(seq_len, 64, 64, seq_len, \
                                        freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                        tcc_chainlen=chain_len)
        base_flops, base_lat = base_model.tensor_fpga21_mat_flops(chain_len, False, False)
        curr_latency += base_lat
        total_base_lat += base_lat

    # attention x v computation
    res = {"latency": [], "sparsity": []}

    for exp in att:
        for h in exp:
            base_model = hw_modeling.StratixDpuModel(h.shape[0], h.shape[1], h.shape[1], 768, \
                                            freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                            tcc_chainlen=chain_len)
            sparse_model = hw_modeling.StratixDpuModel(h.shape[0], h.shape[1], h.shape[1], 768, \
                                            exp_dat=h, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                            tcc_chainlen=chain_len)

            base_flops, base_lat = base_model.tensor_fpga21_mat_flops(chain_len, False, False)
            sparse_flops, sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(h,sort_rows_by_sparsity=sort_row_sparsity, ideal=False)

            curr_sparsity = get_mat_sparsity(h)
            total_base_lat += base_lat

            res["latency"].append(curr_latency + sparse_lat)
            res["sparsity"].append(curr_sparsity)

    res_df = pd.DataFrame(res, columns=res.keys())
    res_df.sort_values(by=["sparsity"], inplace=True)
    print("dense latency: ", total_base_lat)
    return res_df

def plot_selfatt_sparsity(data_path, output_path, chain_len_list, hw_array_shape_list, layer_idx, sort_row_sparsity=False):
    res_df_list = []
    sorted_fig_path = "_sorted" if sort_row_sparsity else "_unsorted"
    
    for chain_len, hw_array_shape in zip(chain_len_list, hw_array_shape_list):
        res_df = compute_selfatt_layer_perf(data_path, 384, chain_len, hw_array_shape, sort_row_sparsity, layer_idx)
        res_df_list.append(res_df)
    
    # plot latency vs sparsity
    for idx, (chain_len, res_df) in enumerate(zip(chain_len_list, res_df_list)):
        plt.scatter(x=res_df["sparsity"], y=res_df["latency"], alpha=0.6, \
                            linewidth=0.1, linestyle='-', marker='s', color=f"C{idx}", label=f"chain len={chain_len}")

    plt.title(f"latency vs. sparsity L{layer_idx}")
    plt.xlabel("sparsity")
    plt.ylabel("latency (cycles)")
    plt.legend()
    plt.grid(linewidth=0.3)
    plt.savefig(output_path + "lat_sparsity" + sorted_fig_path + f"_L{layer_idx}" + ".pdf")
    plt.clf()

    # plot throughput vs sparsity
    # for idx, (chain_len, res_df) in enumerate(zip(chain_len_list, res_df_list)):
    #     plt.scatter(x=res_df["sparsity"], y=res_df["tp"], alpha=0.6, \
    #                         linewidth=0.1, linestyle='-', marker='s', color=f"C{idx}", label=f"chain len={chain_len}")

    # plt.title(f"throughput vs. sparsity{attached_to_fig_name}")
    # plt.xlabel("sparsity")
    # plt.ylabel("throughput (TFLOPs)")
    # plt.legend()
    # plt.grid(linewidth=0.3)
    # plt.savefig(output_path + "tp_sparsity" + sorted_fig_path + attached_to_fig_name + ".pdf")
    # plt.clf()

def single_case_analyzing(mat):

    def create_heatmap(dat, fig_name, latency):
        fig, ax = plt.subplots()
        # ax.set_xlim((0, dat.shape[0]))
        # ax.set_ylim((0, dat.shape[1]))
        ax.set_xticks([0, dat.shape[0]])
        ax.set_xticks(np.arange(0, dat.shape[-1]+20, 20), minor=True)
        ax.set_yticks([0, dat.shape[1]])
        ax.set_yticks(np.arange(0, dat.shape[0]+3, 3), minor=True)
        ax.grid(which="major", alpha=0, color="green")
        ax.grid(which="minor", alpha=1, color="red")
        im = ax.imshow(dat)
        actual_spar = get_mat_sparsity(dat)
        plt.title(f"sparsity={actual_spar:.2f}, latency={latency:.2f}")
        plt.savefig(fig_name)
        plt.clf()

    sort_row_sparsity = True    
    sparse_block_size = 20
    out_w = 768
    hw_array_shape = factor_int(floor(3960.0/(1.0+2.0)))
    print("hw array shape: ", hw_array_shape)

    # first pad the rows to be divisible by 3
    rows_padded = (3 - mat.shape[0] % 3) if (mat.shape[0] % 3 > 0) else 0
    cols_padded = (20 - mat.shape[1] % 20) if (mat.shape[1] % 20 > 0) else 0
    if rows_padded > 0 or cols_padded > 0:
        mat = np.pad(mat, ((0, rows_padded), (0, cols_padded)), "constant", constant_values=0)

    mixed_sparse_list, sparse_list = [],  []
    exps_count = 0

    # use a dense mat to calculate dens mat base lat
    fake_dense_data = np.ones(mat.shape)
    base_model = hw_modeling.StratixDpuModel(mat.shape[0], mat.shape[1], mat.shape[1], out_w, \
                                    exp_dat=fake_dense_data, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                    tcc_chainlen=1)
    base_model.set_tccore_size(20)
    base_flops, base_lat = base_model.tensor_fpga21_mat_sparse_flops(fake_dense_data, \
                                                sort_row_sparsity, False, \
                                                using_single_column=False, sparse_block_size=sparse_block_size)

    dense_mat_map = np.where(mat > 0, 1, 0)
    create_heatmap(dense_mat_map, "./res_fig/dense_heatmap.png", base_lat)

    # evaluate sparse model
    sparse_model = hw_modeling.StratixDpuModel(mat.shape[0], mat.shape[1], mat.shape[1], out_w, \
                                    exp_dat=mat, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                    tcc_chainlen=1)
    sparse_model.set_tccore_size(20)
    sparse_flops, sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(mat, \
                                                    sort_row_sparsity, False, \
                                                    using_single_column=False, sparse_block_size=1, \
                                                    blocked_pruning=False)

    # create mat map for column pruning
    compressed_mat_map = []
    seq_len = mat.shape[-1]

    # then block them into 3xtc core size and select the max length to compute delay
    mat_in_row_grps = np.split(mat, np.arange(3, mat.shape[0], 3), axis=0)
    for row_grp in mat_in_row_grps:
        compressed_row_grp = []
        row_blocks = np.split(row_grp, np.arange(1, row_grp.shape[1], 1), axis=-1)
        for block in row_blocks:
            if np.sum(block) != 0:
                block = compressed_row_grp.append(np.ones(block.shape))
        if len(compressed_row_grp) > 0:
            compressed_row_grp = np.concatenate(compressed_row_grp, axis=-1)

            if compressed_row_grp.shape[-1] % 20 > 0:
                padded_ones = 20 - compressed_row_grp.shape[-1] % 20
                compressed_row_grp = np.pad(compressed_row_grp, \
                                            ((0, 0), (0, padded_ones)), "constant", constant_values = 1)
            if compressed_row_grp.shape[-1] < seq_len:
                compressed_row_grp = np.pad(compressed_row_grp, \
                                            ((0, 0), (0, seq_len - compressed_row_grp.shape[-1])), \
                                            "constant", constant_values=0)
        else:
            compressed_row_grp = np.zeros((3, seq_len))

        # record max none zero values for each 3-row grp
        compressed_mat_map += [compressed_row_grp]

    compressed_mat_map = np.concatenate(compressed_mat_map, axis = 0)
    create_heatmap(compressed_mat_map, "./res_fig/column_prune_heatmap.png", sparse_lat)

    
    sparse_model = hw_modeling.StratixDpuModel(mat.shape[0], mat.shape[1], mat.shape[1], out_w, \
                                    exp_dat=mat, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                    tcc_chainlen=1)
    sparse_model.set_tccore_size(20)
    mixed_sparse_flops, mixed_sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(mat, \
                                                sort_row_sparsity, False, \
                                                using_single_column=False, sparse_block_size=20, \
                                                blocked_pruning=True)
    # create mat map for blocked pruning
    blocked_mat_map = []
    # then block them into 3xtc core size and select the max length to compute delay
    mat_in_row_grps = np.split(mat, np.arange(3, mat.shape[0], 3), axis=0)
    for row_grp in mat_in_row_grps:
        compressed_row_grp = []
        row_blocks = np.split(row_grp, np.arange(20, row_grp.shape[1], 20), axis=-1)
        for block in row_blocks:
            if np.mean(block) > 0.001:
                block = np.ones(block.shape)
            else:
                block = np.zeros(block.shape)
            
            compressed_row_grp.append(block)

        compressed_row_grp = np.concatenate(compressed_row_grp, axis=-1)
        # record max none zero values for each 3-row grp
        blocked_mat_map += [compressed_row_grp]

    blocked_mat_map = np.concatenate(blocked_mat_map, axis = 0)
    create_heatmap(blocked_mat_map, "./res_fig/block_prune_heatmap.png", mixed_sparse_lat)

    print("column pruning and blocked pruning latency: ", base_lat/sparse_lat, base_lat/mixed_sparse_lat)
    print("total instance count: ", exps_count)
    
    return {"base": (0., base_lat), 
            "column": (get_mat_sparsity(compressed_mat_map), sparse_lat), 
            "blocked": (get_mat_sparsity(blocked_mat_map), mixed_sparse_lat)}

def plot_lat_vs_sparsity(file_path, seq_len_path, seq_len_range):
    '''
    illustrate latency under different sparsity
    '''
    bfp_att_probes = prepare_mat_dat(file_path, seq_len_path, seq_len_range)
    res = {"base": [], "column": [], "blocked": []}
    for att in bfp_att_probes:
        temp_lvss_res = single_case_analyzing(att)
        for key in temp_lvss_res.keys():
            res[key].append(temp_lvss_res[key])

    print(f"total number of insts: {len(bfp_att_probes)}")
    # plot latency vs sparsity
    plt.scatter(x=[d[0] for d in res["base"]], y=[d[1] for d in res["base"]], alpha=0.6, \
                    linestyle='None', marker='s', color=f"red", label=f"dense")
    plt.scatter(x=[d[0] for d in res["column"]], y=[d[1] for d in res["column"]], alpha=0.6, \
                    linestyle='None', marker='s', color=f"green", label=f"column prune")
    plt.scatter(x=[d[0] for d in res["blocked"]], y=[d[1] for d in res["blocked"]], alpha=0.6, \
                    linestyle='None', marker='s', color=f"blue", label=f"blocked prune")

    plt.title(f"sparsity vs. latency {seq_len_range}")
    plt.xlabel("sparsity")
    plt.ylabel("latency (cycles)")
    plt.legend()
    plt.grid(linewidth=0.3)
    plt.savefig(f"./res_fig/sparsity_vs_latency{seq_len_range}.png")
    plt.clf()


def analyze_sparsity_blocking(file_path, seq_len_path):
    bfp_att_probes = prepare_mat_dat(file_path, seq_len_path, (0, 1024))

    all_ideal_spar, all_blocked_spar = [], []
    for att in bfp_att_probes:
        ideal_sparsity = get_mat_sparsity(att)

        block_sparse_mat_non_zeros = 0
        zeros_padded = att.shape[0] % 3
        if zeros_padded > 0:
            att = np.pad(att, (0, 3 - zeros_padded), "constant", constant_values=0)
        # then block them into 3xtc core size and select the max length to compute delay
        mat_in_row_grps = np.split(att, np.arange(3, att.shape[0], 3), axis=0)
        for row_grp in mat_in_row_grps:
            compressed_row_grp = []
            row_blocks = np.split(row_grp, np.arange(1, row_grp.shape[1], 1), axis=-1)
            for block in row_blocks:
                if np.sum(block) != 0:
                    compressed_row_grp.append(block)

            compressed_row_grp = np.concatenate(compressed_row_grp, axis=-1)
            # record max none zero values for each 3-row grp
            block_sparse_mat_non_zeros += compressed_row_grp.size

        blocked_sparsity = 1. - block_sparse_mat_non_zeros / att.size
        all_ideal_spar.append(ideal_sparsity)
        all_blocked_spar.append(blocked_sparsity)

        print(f"original: {ideal_sparsity:.2f}, blocked: {blocked_sparsity:.2f}")

    print(f"average original: {np.average(all_ideal_spar):.2f}, average ideal: {np.average(all_blocked_spar):.2f}")
    return np.average(all_ideal_spar), np.average(all_blocked_spar)

def analyze_block_sparsity_compare(file_path, seq_len_path, transpose=True, sort_rows_by_sparsity=True):
    bfp_att_probes = prepare_mat_dat(file_path, seq_len_path, (0, 1024))
    original_spar_list = []
    block_spar_list = []

    for mat in bfp_att_probes:
        #extract block sparsity for original matrix
        original_spar = get_mat_sparsity(mat)

        # sort rows based on the sparsity if sorting is enabled
        if sort_rows_by_sparsity:
            sorted_sparse_mat = mat[(mat == 0.0).sum(axis=-1).argsort()]
            mat = sorted_sparse_mat
            # skip if matrix is fully dense
            if np.count_nonzero(mat) / mat.size < 1.:
                dense_mask = np.where(mat > 0.0, 1, 0)
                zeros_padded = dense_mask.shape[0] % 3
                if zeros_padded > 0:
                    dense_mask = np.pad(dense_mask, (0, 3 - zeros_padded), "constant", \
                                            constant_values=0)
                    mat = np.pad(mat, (0, 3 - zeros_padded), "constant", \
                                            constant_values=0)
                res = []
                h_dist = lambda x, y: hamming(x, y) * len(x)

                while dense_mask.shape[0] > 3:
                    to_compare = dense_mask[0]
                    dense_mask = np.delete(dense_mask, 0, axis=0)
                    res.append(mat[0])
                    mat = np.delete(mat, 0, axis=0)

                    min_hdist = [len(to_compare), len(to_compare)]
                    min_idx = [0, 0]
                    for idx, r in enumerate(dense_mask):
                        c_hdist = h_dist(to_compare, r)
                        if c_hdist < min_hdist[0]:
                            min_hdist = [c_hdist, min_hdist[0]]
                            min_idx = [idx, min_idx[0]]
                        elif c_hdist < min_hdist[1]:
                            min_hdist[1] = c_hdist
                            min_idx[1] = idx
                    
                    res.append(mat[min_idx[0]])
                    res.append(mat[min_idx[1]])
                    mat = np.delete(mat, min_idx, axis=0)
                    dense_mask = np.delete(dense_mask, min_idx, axis=0)

                for r in mat: res.append(r)
                mat = np.array(res)

        zeros_padded = mat.shape[0] % 3
        if transpose:
            mat = np.transpose(mat)
        if zeros_padded > 0:
            mat = np.pad(mat, (0, 3 - zeros_padded), "constant", constant_values=0)
            
        #apply sparse blocks to the matrix
        mean_pooling_res = skimage.measure.block_reduce(mat, (3,20), np.mean)
        block_sparse_mask = np.where(mean_pooling_res > 0.001, 1, 0)
        
        #compute block sparsity for the matrix with block sparse
        block_sparsity = get_mat_sparsity(block_sparse_mask)

        original_spar_list.append(original_spar)
        block_spar_list.append(block_sparsity)

    return np.mean(original_spar_list), np.mean(block_spar_list)

def main():
    data_path = "/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/"
    output_path = "./res_fig/"
    num_layers = 48

    ## analyze performance for the entire attention layer
    # chain_len_list = [3]
    # hw_array_shape_list = [(36, 22)]
    # for layer_idx in range(num_layers):
    #     plot_selfatt_sparsity(data_path, output_path + "selfatt/", chain_len_list, hw_array_shape_list, layer_idx, False)

    ## verify sparsity
    att_path = data_path + f"act/attprobs/6-0.pt"
    seq_len_path = data_path + f"act/seqlen/6-0.pt"
    # sampled_dat = prepare_mat_dat(att_path, seq_len_path, (0, 384), 1)
    # single_case_analyzing(sampled_dat[0])

    # plot_lat_vs_sparsity(att_path, seq_len_path, (300, 384))
    # plot_lat_vs_sparsity(att_path, seq_len_path, (100, 200))
    # plot_lat_vs_sparsity(att_path, seq_len_path, (0, 50))
    # exit()

    ## explore average speedup of different chain lengths:
    files_list = [data_path + f"opt_res/bfp_attn/{i}-0.pt" for i in range(48)]
    seq_len_list = [data_path + f"opt_res/seqlen/{i}-0.pt" for i in range(48)]

    # with multiprocessing.Pool() as pool:
    #     res_list = pool.starmap(analyze_block_sparsity_compare, zip(files_list, seq_len_list))
    # for idx, res_spar in enumerate(res_list):
    #     print(f"layer {idx+1} - original sparsity: {res_spar[0]:.2f}, blocked sparsity: {res_spar[1]:.2f}")
    # blocked_sparsity_spar_mean = [i[1] for i in res_list]

    # # analyze block sparsity
    # with multiprocessing.Pool() as pool:
    #     res_list = pool.starmap(analyze_sparsity_blocking, zip(files_list, seq_len_list))

    # plt.plot(np.arange(1, num_layers+1, 1), [res[0] for res in res_list], \
    #                         linestyle='-', marker='s', color=f"blue", label=f"original")
    # plt.axhline(np.average([res[0] for res in res_list]), linestyle='--', color=f'blue', alpha=0.5)
    # plt.plot(np.arange(1, num_layers+1, 1), [res[1] for res in res_list], \
    #                         linestyle='-', marker='s', color=f"red", label=f"blocking each column")
    # plt.axhline(np.average([res[1] for res in res_list]), linestyle='--', color=f'red', alpha=0.5)
    # plt.plot(np.arange(1, num_layers+1, 1), blocked_sparsity_spar_mean, \
    #                         linestyle='-', marker='s', color=f"orange", label=f"blocked sparsity with mean pooling pruning")
    # plt.axhline(np.average(blocked_sparsity_spar_mean), linestyle='--', color=f'orange', alpha=0.5)          
    # plt.title(f"difference of original and blocked sparsity")
    # plt.xticks(np.arange(0, num_layers+1, 1))
    # plt.xlabel("layer")
    # plt.ylim(0, 1)
    # plt.ylabel("sparsity")
    # plt.legend()
    # plt.grid(linewidth=0.3)
    # plt.savefig(output_path + f"how_bad_block_sparsity_is_forall_opt_stapruning.png")
    # plt.clf()


    ## analyze min len:
    short_chain_len = short_chain_len_profiling(files_list, seq_len_list, (0, 2048))

    print("computing fixed chain len latency...")
    with multiprocessing.Pool() as pool:
        res_list = pool.starmap(plot_perf_sparsity_simple_arglist, \
                                    zip(files_list, seq_len_list, [0.0]*num_layers, [True]*num_layers, [short_chain_len]*num_layers))

    # select several fixed chain length and plot speedup changes across layers
    sparselat_list = np.transpose(np.array([i[0] for i in res_list]))
    ideallat_list = np.transpose(np.array([i[1] for i in res_list]))
    baselat_list = np.transpose(np.array([i[2] for i in res_list]))

    sparselat_speedup_list = baselat_list / sparselat_list
    ideallat_speedup_list = baselat_list / ideallat_list

    # for chain_len_idx in [8, 10, 20]:
    #     plt.plot(np.arange(1, 13, 1), sparselat_speedup_list[chain_len_idx-1], \
    #                             linestyle='-', marker='s', color=f"blue", label=f"w/ TC 3-col limit")
    #     plt.axhline(np.average(sparselat_speedup_list[chain_len_idx-1]), linestyle='--', color=f'blue', alpha=0.5)
    #     plt.plot(np.arange(1, 13, 1), ideallat_speedup_list[chain_len_idx-1], \
    #                             linestyle='-', marker='s', color=f"red", label=f"w/o TC 3-col limit")
    #     plt.axhline(np.average(ideallat_speedup_list[chain_len_idx-1]), linestyle='--', color=f'red', alpha=0.5)
    #     plt.title(f"speedup per layer")
    #     plt.xticks(np.arange(0, num_layers+1, 1))
    #     plt.xlabel("layer")
    #     plt.ylim(0, 9)
    #     plt.ylabel("latency speedup")
    #     plt.legend()
    #     plt.grid(linewidth=0.3)
    #     plt.savefig(output_path + f"speedup_vs_layer_chainlen{chain_len_idx}.png")
    #     plt.clf()
        
    # sparselat_sum = np.add.reduce([i[0] for i in res_list])
    # ideallat_sum = np.add.reduce([i[1] for i in res_list])
    # baselat_sum = np.add.reduce([i[2] for i in res_list])

    # sparselat_speedup = baselat_sum / sparselat_sum
    # ideallat_speedup = baselat_sum / ideallat_sum 

    # plt.plot(np.arange(1, len(sparselat_speedup)+1, 1), sparselat_speedup, \
    #                     linestyle='-', marker='s', color=f"blue", label=f"w/ TC 3-col limit")
    # plt.plot(np.arange(1, len(ideallat_speedup)+1, 1), ideallat_speedup, \
    #                     linestyle='-', marker='s', color=f"red", label=f"w/o TC 3-col limit")

    # plt.title(f"speedup vs. chain length")
    # plt.xlabel("chain length")
    # plt.ylabel("latency speedup")
    # plt.legend()
    # plt.grid(linewidth=0.3)
    # plt.savefig(output_path + "speedup_chainlen_sweep.png")
    # plt.clf()

    ## analyze performance vs sparsity over the layers for attention only
    fixed_chain_len_switch = [False] * num_layers

    # sweep across different ratios:
    curr_min_latency_mean = float("inf")
    sparselat_list_mixed_res = []
    s2l_ratio_res = 0.0

    print("computing mixed chain len latency...")
    if s2l_ratio_res > 0.0:
        with multiprocessing.Pool() as pool:
            res_list = pool.starmap(plot_perf_sparsity_simple_arglist, \
                        zip(files_list, seq_len_list, [s2l_ratio_res]*num_layers, fixed_chain_len_switch, [short_chain_len]*num_layers))

        # select several fixed chain length and plot speedup changes across layers
        sparselat_list_mixed = np.transpose(np.array([i[0] for i in res_list]))
        baselat_list_mixed = np.transpose(np.array([i[2] for i in res_list]))

        mixed_speedup = baselat_list_mixed / sparselat_list_mixed
        sparselat_list_mixed_res = mixed_speedup
    else:
        for s2l_ratio in np.arange(0.02, 0.2, 0.02):
            print(f"sweeping ratio {s2l_ratio}...")
            with multiprocessing.Pool() as pool:
                res_list = pool.starmap(plot_perf_sparsity_simple_arglist, \
                            zip(files_list, seq_len_list, [s2l_ratio]*num_layers,  fixed_chain_len_switch, [short_chain_len]*num_layers))

            # select several fixed chain length and plot speedup changes across layers
            sparselat_list_mixed = np.transpose(np.array([i[0] for i in res_list]))
            baselat_list_mixed = np.transpose(np.array([i[2] for i in res_list]))
            mixed_speedup = baselat_list_mixed / sparselat_list_mixed

            mixed_lat_mean = np.average(sparselat_list_mixed)
            if mixed_lat_mean < curr_min_latency_mean:
                s2l_ratio_res = s2l_ratio
                curr_min_latency_mean = mixed_lat_mean
                sparselat_list_mixed_res = mixed_speedup

    print(f"best s2l ratio: {s2l_ratio_res}")
    print("sparselat mixed: ", sparselat_list_mixed_res)
    for mixed_chainlen_list_idx, chain_len_idx in enumerate([10, 20, 34]):
        plt.plot(np.arange(1, num_layers+1, 1), sparselat_list_mixed_res[mixed_chainlen_list_idx], \
                                linestyle='-', marker='s', color=f"blue", label=f"mixed chain len")
        plt.axhline(np.average(sparselat_list_mixed_res[mixed_chainlen_list_idx]), linestyle='--', color=f'blue', alpha=0.5)
        plt.plot(np.arange(1, num_layers+1, 1), sparselat_speedup_list[mixed_chainlen_list_idx+1], \
                                linestyle='-', marker='s', color=f"red", label=f"chain len={chain_len_idx}")
        plt.axhline(np.average(sparselat_speedup_list[mixed_chainlen_list_idx+1]), linestyle='--', color=f'red', alpha=0.5)
        plt.plot(np.arange(1, num_layers+1, 1), sparselat_speedup_list[0], \
                                linestyle='-', marker='s', color=f"green", label=f"chain len={short_chain_len}")
        plt.axhline(np.average(sparselat_speedup_list[0]), linestyle='--', color=f'green', alpha=0.5)

        plt.title(f"latency of mixed chain len=({short_chain_len}, {chain_len_idx})")
        plt.xticks(np.arange(0, num_layers+1, 1))
        plt.xlabel("layer")
        plt.ylim(0, 6)
        plt.ylabel("speed up")
        plt.legend()
        plt.grid(linewidth=0.3)
        plt.savefig(output_path + f"speedup_vs_layer_mixed_chainlen_opt_stapruning({short_chain_len},{chain_len_idx}).png")
        plt.clf()

if __name__ == "__main__":
    main()
