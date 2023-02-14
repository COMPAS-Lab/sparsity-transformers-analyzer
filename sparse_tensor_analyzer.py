import torch
import matplotlib
import numpy as np
import hw_modeling
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
from math import ceil, floor, sqrt
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

def get_mat_sparsity(dat, causal_mask = False):
    if type(dat) == torch.Tensor:
        if causal_mask:
            total_num_elems = sum(np.arange(1, dat.size()[-1], 1))
            total_num_elems *= dat.view(-1, dat.size()[-1], dat.size()[-1]).size()[0]
            nonzeros_per_row = torch.count_nonzero(dat, dim=-1)
            proposed_nonzeros = torch.tensor(np.arange(1, dat.size()[-1]+1, 1))
            actual_zeros = torch.sum(proposed_nonzeros - nonzeros_per_row).item()
            return float(actual_zeros) / total_num_elems
        else:
            return (1. - torch.count_nonzero(dat) / torch.numel(dat))
    else:
        return (1. - np.count_nonzero(dat) / dat.size)

def compute_matmul_performance(data_path, chain_len, out_w, hw_array_shape, \
                                sort_row_sparsity, using_single_column, sparse_block_size, \
                                seq_len_path=None, seq_len_range=None, mixed_chain_length=None, short_to_long_ratio=0.0):

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
    num_insts = len(bfp_att_probes)
    for exps in bfp_att_probes:
        # use a dense mat to calculate dens mat base lat
        fake_dense_data = np.random.rand(exps.shape[0], exps.shape[1])
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
                                                        using_single_column, sparse_block_size)
        else:
            sparse_model = hw_modeling.StratixDpuModel(exps.shape[0], exps.shape[1], exps.shape[1], out_w, \
                                            exp_dat=exps, freq=500, num_tcs=3960, tcc_array_shape=hw_array_shape, \
                                            tcc_chainlen=mixed_chain_length)
            sparse_model.set_tccore_size(20)
            sparse_flops, sparse_lat = sparse_model.tensor_fpga21_mat_sparse_flops(exps, \
                                                        sort_row_sparsity, False, \
                                                        using_single_column, sparse_block_size, False, short_to_long_ratio)
                                                    
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
                            attached_to_fig_name="", mixed_chain_len = None, short_to_long_ratio = 0.0, \
                            plot_latency_by_insts = False):
    res_df_list, sparselat_list, ideallat_list, baselat_list = [], [], [], []
    sorted_fig_path = "_sorted" if sort_row_sparsity else "_unsorted"

    for chain_len, hw_array_shape in zip(chain_len_list, hw_array_shape_list):
        res_df, sparse_lat, ideal_lat, base_lat = compute_matmul_performance(data_path, chain_len, 768, \
                                                            hw_array_shape, sort_row_sparsity, \
                                                            using_single_column, sparse_block_size, 
                                                            seq_len_path, seq_len_range, mixed_chain_len, \
                                                            short_to_long_ratio)
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

    for chain_len, sparse_lat, ideal_lat, base_lat in \
            zip(chain_len_list, sparselat_list, ideallat_list, baselat_list):
        print(f'''chain len: {chain_len}, 
                    sparse speedup: {base_lat/sparse_lat:.2f}, 
                    ideal speed up: {base_lat/ideal_lat:.2f}''')

    return sparselat_list, ideallat_list, baselat_list

def plot_perf_sparsity_simple_arglist (data_path, seq_len_path):
    chain_len_list = np.arange(1, 33, 1).tolist()
    hw_array_shape_list = [factor_int(floor(3960.0/(i+2.0))) for i in chain_len_list]

    return plot_perf_sparsity(data_path = data_path, output_path="./res_fig/", \
                            chain_len_list = chain_len_list, \
                            hw_array_shape_list = hw_array_shape_list, \
                            sort_row_sparsity=True, \
                            using_single_column=False, sparse_block_size=1, \
                            seq_len_path=seq_len_path, seq_len_range=(200, 384), \
                            plot_latency_by_insts=False)

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


    # explore average speedup of different chain lengths:
    files_list = [data_path + f"act/attprobs/{i}-0.pt" for i in range(12)]
    seq_len_list = [data_path + f"act/seqlen/{i}-0.pt" for i in range(12)]

    with multiprocessing.Pool() as pool:
        res_list = pool.starmap(plot_perf_sparsity_simple_arglist, zip(files_list, seq_len_list))

    print(len(res_list))
    print(len(res_list[0]))

    sparselat_sum = np.add.reduce([i[0] for i in res_list])
    ideallat_sum = np.add.reduce([i[1] for i in res_list])
    baselat_sum = np.add.reduce([i[2] for i in res_list])

    sparselat_speedup = baselat_sum / sparselat_sum
    ideallat_speedup = baselat_sum / ideallat_sum 

    plt.plot(x=chain_len_list, y=sparselat_speedup, alpha=0.6, \
                        linewidth=0.1, linestyle='-', marker='s', color=f"blue", label=f"w/ TC size limit")
    plt.plot(x=chain_len_list, y=ideallat_speedup, alpha=0.6, \
                        linewidth=0.1, linestyle='-', marker='s', color=f"red", label=f"w/o TC size limit")

    plt.title(f"speedup vs. chain length")
    plt.xlabel("chain length")
    plt.ylabel("latency speedup")
    plt.legend()
    plt.grid(linewidth=0.3)
    plt.savefig(output_path + "speedup_chainlen_sweep.png")
    plt.clf()
    exit()

    # analyze performance vs sparsity over the layers for attention only
    # chain_len_list = [8, 10, 20]
    # hw_array_shape_list = [(22, 18), (33, 10), (18, 10)]
    chain_len_list = [8]
    hw_array_shape_list = [(22, 18)]
    files_list = [f"act/attprobs/{i}-0.pt" for i in range(12)]
    seq_len_list = [f"act/seqlen/{i}-0.pt" for i in range(12)]
    for l_idx, (fname, f_seq_len) in enumerate(zip(files_list, seq_len_list)):
        plot_perf_sparsity(data_path + fname, output_path, chain_len_list, \
                                hw_array_shape_list, sort_row_sparsity=True, \
                                using_single_column=False, sparse_block_size=1, \
                                seq_len_path=data_path+f_seq_len, seq_len_range=(200, 384), \
                                attached_to_fig_name=f"_L{l_idx}_spblk_1_advsort_mixed_chain", \
                                mixed_chain_len=(5, 20), short_to_long_ratio=1.5)

if __name__ == "__main__":
    main()
