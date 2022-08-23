from unittest.util import sorted_list_difference
import torch
import matplotlib
import numpy as np
import hw_modeling
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union

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

def compute_matmul_performance(data_path, chain_len, out_w, hw_array_shape, \
                                sort_row_sparsity, using_single_column, sparse_block_size, \
                                seq_len_path=None, seq_len_range=None):

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

    res = {"latency":[] ,"sparsity": [], "s2l ratio": [], "tp": []}

    total_sparse_lat, total_base_lat, total_min_sparse_lat = 0, 0, 0
    for exps in bfp_att_probes:
        # use a dense mat to calculate dens mat base lat
        fake_dense_data = np.random.rand(exps.shape[0], exps.shape[1])
        base_model = hw_modeling.StratixDpuModel(exps.shape[0], exps.shape[1], exps.shape[1], out_w, \
                                        exp_dat=fake_dense_data, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                        tcc_chainlen=chain_len)
        base_model.set_tccore_size(20)
        sparse_model = hw_modeling.StratixDpuModel(exps.shape[0], exps.shape[1], exps.shape[1], out_w, \
                                        exp_dat=exps, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                        tcc_chainlen=chain_len)
        sparse_model.set_tccore_size(20)

        base_flops, base_lat = base_model.tensor_fpga21_mat_sparse_flops(fake_dense_data, \
                                                    sort_row_sparsity, False, \
                                                    using_single_column=False, sparse_block_size=sparse_block_size)
        sparse_flops, sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(exps, \
                                                    sort_row_sparsity, False, \
                                                    using_single_column, sparse_block_size)
        min_sparse_flops, min_sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(exps, \
                                                    sort_row_sparsity, True, False, sparse_block_size, True)
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
    return res_df, total_sparse_lat/total_base_lat, total_sparse_lat/total_min_sparse_lat

def plot_perf_sparsity(data_path, output_path, chain_len_list, hw_array_shape_list, sort_row_sparsity, \
                            using_single_column, sparse_block_size, \
                            seq_len_path = None, seq_len_range = None, \
                            attached_to_fig_name=""):
    res_df_list, relative_slat_list, slat_min_list = [], [], []
    sorted_fig_path = "_sorted" if sort_row_sparsity else "_unsorted"

    for chain_len, hw_array_shape in zip(chain_len_list, hw_array_shape_list):
        res_df, relative_slat, slat_min = compute_matmul_performance(data_path, chain_len, 768, \
                                                            hw_array_shape, sort_row_sparsity, \
                                                            using_single_column, sparse_block_size, 
                                                            seq_len_path, seq_len_range)
        res_df_list.append(res_df)
        relative_slat_list.append(relative_slat)
        slat_min_list.append(slat_min)
    
    # plot latency vs sparsity
    for idx, (chain_len, res_df, relative_slat) in enumerate(zip(chain_len_list, res_df_list, relative_slat_list)):
        plt.scatter(x=res_df["sparsity"], y=res_df["latency"], alpha=0.6, \
                            linewidth=0.1, linestyle='-', marker='s', color=f"C{idx}", label=f"chain len={chain_len}")
        plt.axhline(relative_slat, linestyle='--', color=f'C{idx}', alpha=0.5)
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

    for chain_len, relative_slat, slat_min in zip(chain_len_list, relative_slat_list, slat_min_list):
        print(f"chain len: {chain_len}, relative lat: {relative_slat:.2f}, lat relative to min: {slat_min:.2f}")

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

def main():
    data_path = "/var/services/homes/tianchu.ji/mackeson-home/spar_test_params/jason_res/sample/"
    output_path = "./res_fig/"

    # # analyze performance for the entire attention layer
    # chain_len_list = [3]
    # hw_array_shape_list = [(36, 22)]
    # for layer_idx in range(12):
    #     plot_selfatt_sparsity(data_path, output_path + "selfatt/", chain_len_list, hw_array_shape_list, layer_idx, False)

    # analyze performance vs sparsity over the layers for attention only
    chain_len_list = [3, 10, 20]
    hw_array_shape_list = [(36, 22), (33, 10), (18, 10)]
    files_list = [f"act/attprobs/{i}-0.pt" for i in range(12)]
    seq_len_list = [f"act/seqlen/{i}-0.pt" for i in range(12)]
    for l_idx, (fname, f_seq_len) in enumerate(zip(files_list, seq_len_list)):
        plot_perf_sparsity(data_path + fname, output_path, chain_len_list, \
                                hw_array_shape_list, sort_row_sparsity=True, \
                                using_single_column=False, sparse_block_size=1, \
                                seq_len_path=data_path+f_seq_len, seq_len_range=(200, 384), \
                                attached_to_fig_name=f"_L{l_idx}_spblk_1_advsort")

if __name__ == "__main__":
    main()