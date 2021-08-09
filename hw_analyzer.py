import enum
from math import ceil, floor, exp, log2, sqrt, isnan
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import random
from textwrap import wrap

from itertools import product
from tqdm import tqdm


from hw_modeling import BertModel, DpuModel

def compare_naive_softmax_heads(bert_model):
    fsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(21, 9))

    patches = [# mpatches.Patch(color='black', linestyle='--', label='Value computation'),
                mpatches.Patch(color='C0', label='our softmax'), 
                mpatches.Patch(color='C1', label='baseline softmax')]

    for inst in bert_hw_model.exps:
        num_layers, num_heads, num_rows, _ = inst.shape
        heads = inst.reshape((num_layers*num_heads, num_rows, num_rows))
        softmax_lats_per_head = np.array([bert_hw_model.softmax_lat(dat[:20], p1=8, p2=8) for dat in heads])
        baseline_softmax_lats_per_head = np.array([bert_hw_model.baseline_softmax_lat(dat[:20], pa=8.) for dat in heads])
        v_compute_lats_per_head = bert_hw_model.matmul_lat_qkv_per_head(heads.shape[-1], blk=(32, 32))
        softmax_lats_per_head /= v_compute_lats_per_head
        baseline_softmax_lats_per_head /= v_compute_lats_per_head

        # ax.axhline(v_compute_lats_per_head, linestyle='--', color='black', alpha=0.1, linewidth=2)
        ax.plot(range(len(heads)), softmax_lats_per_head, linestyle='-', color='C0', alpha=0.1, linewidth=2)
        ax.plot(range(len(heads)), baseline_softmax_lats_per_head, linestyle='-', color='C1', alpha=0.1, linewidth=2)
        
    ax.axhline(1., linestyle='--', color='black', alpha=0.5, linewidth=2)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=2)
    # ax.set_ylabel(r'latency/cycle', fontsize=fsize)
    ax.set_ylabel(r'$\frac{softmax\ latency}{V\ compute\ latency}$', fontsize=fsize)
    ax.set_xlabel('heads', fontsize=fsize)
    for idx, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_fontsize(fsize)
    for idx, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_fontsize(fsize)

    fig.tight_layout()
    plt.legend(handles=patches, loc='upper right', fontsize=fsize)
    fig.savefig('res_fig/softmax_lat_analyze_normalized_heads.pdf')
    plt.clf()


def compare_naive_softmax_parallel(bert_model):
    fsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    patches = [# mpatches.Patch(color='black', linestyle='--', label='Value computation'),
                mpatches.Patch(color='C0', label='our softmax'), 
                mpatches.Patch(color='C1', label='baseline softmax')]

    parallel_list = [2**i for i in range(7)]
    softmax_lats, baseline_lats = [], []

    for p in parallel_list:
        softmax_lats_inst_aggs, baseline_softmax_lats_inst_aggs = [], []
        for inst in bert_hw_model.exps:
            num_layers, num_heads, num_rows, _ = inst.shape
            heads = inst.reshape((num_layers * num_heads, num_rows, num_rows))
            softmax_lats_per_head = np.array([bert_hw_model.softmax_lat(dat[:100], p1=p, p2=p) for dat in heads])
            baseline_softmax_lats_per_head = np.array([bert_hw_model.baseline_softmax_lat(dat[:100], pa=p) for dat in heads])
            v_compute_lats_per_head = bert_hw_model.matmul_lat_qkv_per_head(heads.shape[-1])

            softmax_lats_per_head /= v_compute_lats_per_head
            baseline_softmax_lats_per_head /= v_compute_lats_per_head

            softmax_lats_inst_aggs.append(softmax_lats_per_head)
            baseline_softmax_lats_inst_aggs.append(baseline_softmax_lats_per_head)

        softmax_lats.append(np.mean(np.array(softmax_lats_inst_aggs)))
        baseline_lats.append(np.mean(np.array(baseline_softmax_lats_inst_aggs)))
    
    ax.plot(parallel_list, softmax_lats, linestyle='-', color='C0', marker='s', alpha=1, linewidth=2)
    ax.plot(parallel_list, baseline_lats, linestyle='-', color='C1', marker='s', alpha=1, linewidth=2)
    ax.axhline(1., linestyle='--', color='black', alpha=0.5)

    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=2)
    # ax.set_ylabel(r'latency/cycle', fontsize=fsize)
    ax.set_ylabel(r'$\frac{softmax\ latency}{V\ compute\ latency}$', fontsize=fsize)
    ax.set_xlabel('parallelism', fontsize=fsize)
    ax.set_xscale('log', base=2)
    for idx, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_fontsize(fsize)
    for idx, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_fontsize(fsize)

    fig.tight_layout()
    plt.legend(handles=patches, loc='upper right', fontsize=fsize)
    fig.savefig('res_fig/softmax_lat_analyze_normalized_parallel.pdf')
    plt.clf()


def explore_p1_p2(bert_model):
    p1_lst = [2**i for i in range(6)]
    p2_lst = p1_lst

    res_lat_mat = []

    for p1, p2 in product(p1_lst, p2_lst):
        curr_setup_lat = []
        broken_setup = False
        for inst in bert_hw_model.exps:
            num_layers, num_heads, num_rows, _ = inst.shape
            heads = inst.reshape((num_layers * num_heads, num_rows, num_rows))
            softmax_lats_per_head = []
            for idx, dat in enumerate(heads):
                lat = bert_hw_model.softmax_lat(dat[:100], p1=p1, p2=p2)
                softmax_lats_per_head.append(lat)
            
            if np.isinf(np.sum(softmax_lats_per_head)):
                broken_setup = True
                break
            curr_setup_lat.append(softmax_lats_per_head)

        if broken_setup is False:
            res_lat_mat.append(np.mean(np.array(curr_setup_lat)))
        else:
            res_lat_mat.append(float('inf'))

    res_lat_mat = np.array(res_lat_mat).reshape((len(p1_lst), len(p2_lst)))

    print(res_lat_mat)


def search_lowest_latency_in_bins(softmax_lat_lst: list, bin_width=50, resource_key='dsps'):
    softmax_lat_lst.sort(key=lambda x: x[resource_key])
    curr_bin, lst_idx = min([i[resource_key] for i in softmax_lat_lst]), 0
    final_softmax_lat_lst = []
    while curr_bin < max([i[resource_key] for i in softmax_lat_lst]):
        temp_list = []
        while curr_bin <= softmax_lat_lst[lst_idx][resource_key] < (curr_bin + bin_width) :
            temp_list.append(softmax_lat_lst[lst_idx])
            lst_idx += 1
            if lst_idx >= len(softmax_lat_lst):
                break

        temp_list.sort(key=lambda x: x['latency'])
        if len(temp_list) > 0:
            if (len(final_softmax_lat_lst) < 1) or \
                (len(final_softmax_lat_lst) > 0 and temp_list[0]['latency'] < final_softmax_lat_lst[-1]['latency']):
                final_softmax_lat_lst.append(temp_list[0])
        curr_bin += bin_width

    return final_softmax_lat_lst


def compare_lat_res_models(bert_hw_model: BertModel, resource_type = ['dsp', 'mem'], hw_modeling_type = ["softmax", "baseline softmax", "value mvm"], l_range = [200, 100, 50, 25]):
    '''
    compare latency of the models
    '''
    fsize = 9
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    # define parallelism sweeping range
    softmax_range = [(i+1) for i in range(64)]
    baseline_range = [(i+1) for i in range(32)]
    mvm_range = [2*(i) for i in np.arange(8, 33)]
    
    print("mvm range: ", mvm_range)

    value_res_lst = [bert_hw_model.matmul_res_qkv_per_head(blk=(w, w)) for w in mvm_range]

    if len(bert_hw_model.exps) == 1:
        print("selecting only one inst, seq len: ", bert_hw_model.exps[0].shape)

    softmax_lat_lst = []
    if "softmax" in hw_modeling_type:
        for l, p in product(l_range, softmax_range):
            temp_softmax_lats = None
            temp_softmax_incycles = []
            for inst in bert_hw_model.exps:
                num_layers, num_heads, num_rows, _ = inst.shape
                heads = inst.reshape((num_layers * num_heads, num_rows, num_rows))

                temp_heads_lats = np.array([bert_hw_model.softmax_lat(dat[:l], p1=p, p2=p) for dat in heads])
                temp_softmax_incycles.append(l * float(inst.shape[-1]) / p)
                temp_softmax_lats = temp_heads_lats if temp_softmax_lats is None \
                                        else np.concatenate([temp_softmax_lats, temp_heads_lats], axis=0)
                
            temp_softmax_res = bert_hw_model.softmax_resources(p, p, l, 4)
            temp_softmax_incycles = np.mean(temp_softmax_incycles)
            temp_lats = np.mean(temp_softmax_lats)
            softmax_lat_lst.append({'dsps': temp_softmax_res[0], 'mem': temp_softmax_res[1], 'latency': temp_lats, 'in_cycles': temp_softmax_incycles, "l": l, "p": p})

        softmax_lat_lst.sort(key=lambda x: x['dsps'])
        final_softmax_lat_lst = search_lowest_latency_in_bins(softmax_lat_lst, resource_key='dsps')
        softmax_lat_lst.sort(key=lambda x: x['mem'])
        final_softmax_mem_lst = search_lowest_latency_in_bins(softmax_lat_lst, resource_key='mem', bin_width=10)

    baseline_softmax_lat_lst = []
    if "baseline softmax" in hw_modeling_type:
        for l, p in product(l_range, baseline_range):
            temp_baseline_softmax_lats = None
            for inst in bert_hw_model.exps:
                num_layers, num_heads, num_rows, _ = inst.shape
                heads = inst.reshape((num_layers * num_heads, num_rows, num_rows))

                temp_heads_lats = np.array([bert_hw_model.baseline_softmax_lat(dat[:l], pa=p) for dat in heads])
                temp_baseline_softmax_lats = temp_heads_lats if temp_baseline_softmax_lats is None \
                                                else np.concatenate([temp_baseline_softmax_lats, temp_heads_lats], axis=0)

            temp_baseline_softmax_res = bert_hw_model.baseline_softmax_resource(p, l)
            temp_lats = np.mean(temp_baseline_softmax_lats)
            baseline_softmax_lat_lst.append({'dsps': temp_baseline_softmax_res[0], 'mem': temp_baseline_softmax_res[1], 'latency': temp_lats})

        baseline_softmax_lat_lst.sort(key=lambda x: x['dsps'])
        final_baseline_softmax_lat_lst = search_lowest_latency_in_bins(baseline_softmax_lat_lst, resource_key='dsps')
        baseline_softmax_lat_lst.sort(key=lambda x: x['mem'])
        final_baseline_softmax_mem_lst = search_lowest_latency_in_bins(baseline_softmax_lat_lst, resource_key='mem', bin_width=10)


    v_compute_head_lat_lst = []
    v_compute_head_lat_lst_ideal = []
    if "value mvm" in hw_modeling_type:
        for p in mvm_range:
            temp_v_compute_head_lat = []
            temp_v_compute_head_lat_ideal = []
            for inst in bert_hw_model.exps:
                num_layers, num_heads, num_rows, _ = inst.shape
                temp_v_compute_head_lat.append(bert_hw_model.matmul_lat_qkv_per_head(num_rows, blk=(p, p)))
                # temp_v_compute_head_lat_ideal.append(bert_hw_model.matmul_lat_qkv_per_head(num_rows, blk=(p, p), ideal=True))

            v_compute_head_lat_lst.append(np.mean(temp_v_compute_head_lat))
            v_compute_head_lat_lst_ideal.append(np.mean(temp_v_compute_head_lat_ideal))

    qktrans_incycle = bert_hw_model.matmul_lat_qktrans_per_head_ideal(320, 143.0)
    qkv_incycle = bert_hw_model.matmul_lat_qkv_per_head_ideal(320, 143.0)
    ddl = qktrans_incycle + qkv_incycle * 3

    blk_size = int(sqrt(3960*30))
    qktrans_incycle_stratix, _, _ = bert_hw_model.matmul_lat_qktrans_per_head_stratix(320, blk=(blk_size, blk_size), ideal=True)
    qkv_incycle_stratix, _, _ = bert_hw_model.matmul_lat_qkv_per_head_stratix(320, blk=(blk_size, blk_size), ideal=True)
    stratix_ddl = qktrans_incycle_stratix + qkv_incycle_stratix * 3

    # plot lines
    if 'dsp' in resource_type:
        if len(softmax_lat_lst) > 0:
            ax.plot([i['dsps'] for i in final_softmax_lat_lst], [i['latency'] for i in final_softmax_lat_lst], 
                        linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8, label="our softmax")
            ax.plot([i['dsps'] for i in final_softmax_lat_lst], [i['in_cycles'] for i in final_softmax_lat_lst], 
                        linestyle='--', color='C0', marker='s', linewidth=1, alpha=0.8, label="softmax input a cycle")

        if len(baseline_softmax_lat_lst) > 0:
            ax.plot([i['dsps'] for i in final_baseline_softmax_lat_lst], [i['latency'] for i in final_baseline_softmax_lat_lst], 
                        linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8, label="baseline")

        if len(v_compute_head_lat_lst) > 0:
            ax.plot([dsp[0] for dsp in value_res_lst], v_compute_head_lat_lst, linestyle='-', color='C2', marker='s', linewidth=1, label="v compute")
            ax.plot([dsp[0] for dsp in value_res_lst], v_compute_head_lat_lst_ideal, alpha=0.4, linestyle='-', color='C2', marker='s', linewidth=1)

        ax.axhline(ddl, linestyle='--', color='blue', label='softmax deadline (ideal)')
        ax.axhline(stratix_ddl, linestyle='--', color='red', label='softmax deadline (ideal DPU)')

        ax.set_xlabel('AI Tensor Usage', fontsize=fsize)
        ax.set_ylabel('latency', fontsize=fsize)
        ax.set_ylim(ymin=0, ymax=10000)
        ax.set_xlim(xmin=0)
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)

        fig.tight_layout()
        plt.legend(loc='upper right', fontsize=fsize)
        fig.savefig('res_fig/softmax_lat_res_analyze_pe_fixed_len.pdf')
        plt.cla()
    
    if 'mem' in resource_type:
        if len(softmax_lat_lst) > 0:
            ax.plot([i['mem'] for i in final_softmax_mem_lst], [i['latency'] for i in final_softmax_mem_lst], 
                        linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8, label="our softmax")
        if len(baseline_softmax_lat_lst) > 0:
            ax.plot([i['mem'] for i in final_baseline_softmax_mem_lst], [i['latency'] for i in final_baseline_softmax_mem_lst], 
                        linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8, label="baseline")
        if len(v_compute_head_lat_lst) > 0:
            ax.plot([mem[1] for mem in value_res_lst], v_compute_head_lat_lst, linestyle='-', color='C2', marker='s', linewidth=1, label="v compute")


        ax.set_xlabel('Mem Usage/KB', fontsize=fsize)
        ax.set_ylabel('latency', fontsize=fsize)
        ax.set_xlim(xmin=0, xmax=250)
        ax.set_ylim(ymin=0)
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)

        fig.tight_layout()
        plt.legend(loc='upper left', fontsize=fsize)
        fig.savefig('res_fig/softmax_lat_res_analyze_mem.pdf')
        plt.cla()


def compare_softmax_with_model_len(l_range = [200, 100, 50, 25], max_len_range = [128, 256, 320]):
    '''
    sweeping across multiple model size
    '''
    fsize = 9
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 3

    # define parallelism sweeping range
    softmax_range = [2**i for i in range(8)]

    def search_lowest_latency_in_bins(softmax_lat_lst: list, bin_width=100, resource_key='dsps'):
        softmax_lat_lst.sort(key=lambda x: x[resource_key])
        curr_bin, lst_idx = min([i[resource_key] for i in softmax_lat_lst]), 0
        final_softmax_lat_lst = []
        while curr_bin < max([i[resource_key] for i in softmax_lat_lst]):
            temp_list = []
            while curr_bin <= softmax_lat_lst[lst_idx][resource_key] < (curr_bin + bin_width) :
                temp_list.append(softmax_lat_lst[lst_idx])
                lst_idx += 1
                if lst_idx >= len(softmax_lat_lst):
                    break

            temp_list.sort(key=lambda x: x['latency'])
            if len(temp_list) > 0:
                final_softmax_lat_lst.append(temp_list[0])
            curr_bin += bin_width

        return final_softmax_lat_lst

    for max_len in max_len_range:
        exp_path = f"params/scrs_sampled_{max_len}.npy"
        att_path = f"params/attentions_sampled_{max_len}.npy"
        bert_hw_model = BertModel(read_exp_samples=True, exp_sample_path=exp_path, att_sample_path=att_path)

        softmax_lat_lst = []

        for l, p in product(l_range, softmax_range):
            temp_softmax_lats = None
            for inst in bert_hw_model.exps:
                num_layers, num_heads, num_rows, _ = inst.shape
                heads = inst.reshape((num_layers * num_heads, num_rows, num_rows))

                temp_heads_lats = np.array([bert_hw_model.softmax_lat(dat[:l], p1=p, p2=p) for dat in heads])
                temp_softmax_lats = temp_heads_lats if temp_softmax_lats is None \
                                            else np.concatenate([temp_softmax_lats, temp_heads_lats], axis=0)
                    
            temp_softmax_res = bert_hw_model.softmax_resources(p, p, l, 4)
            temp_lats = np.mean(temp_softmax_lats)
            softmax_lat_lst.append({'dsps': temp_softmax_res[0], 'mem': temp_softmax_res[1], 'latency': temp_lats})

        softmax_lat_lst.sort(key=lambda x: x['dsps'])
        final_softmax_lat_lst = search_lowest_latency_in_bins(softmax_lat_lst, resource_key='dsps')

        # plot lines
        if len(softmax_lat_lst) > 0:
            ax.plot([i['dsps'] for i in final_softmax_lat_lst], [i['latency'] for i in final_softmax_lat_lst], 
                        linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8)
            ax.text(x=final_softmax_lat_lst[0]['dsp'] + 10, y=final_softmax_lat_lst[0]['latency'] + 10, s=f"len={max_len}")

    ax.set_xlabel('DSP Usage', fontsize=fsize)
    ax.set_ylabel('latency', fontsize=fsize)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)

    fig.tight_layout()
    fig.savefig('res_fig/softmax_sweep_model_size.pdf')
    plt.cla()


def mem_teardown(bert_hw_model: BertModel):    
    fsize = 9
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    bar_width=2
    
    l = 100
    # define parallelism sweeping range
    softmax_range = [4*(i+1) for i in range(8)]
    softmax_res_lst = [bert_hw_model.softmax_mem_teardown(p, p, l, 4) for p in softmax_range]
    ax.bar(softmax_range, [res['exp mem'] for res in softmax_res_lst], bar_width, color='C0')
    ax.bar(softmax_range, [res['accu mem'] for res in softmax_res_lst], bar_width, bottom=[res['exp mem'] for res in softmax_res_lst], color='C1')
    ax.bar(softmax_range, [res['exp out buffer'] for res in softmax_res_lst], bar_width, bottom=[res['exp mem']+res['accu mem'] for res in softmax_res_lst], color='C2')
   
    ax.set_xlabel('parallelism', fontsize=fsize)
    ax.set_ylabel('mem usage/Kb', fontsize=fsize)
    ax.set_xticks(softmax_range)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    patches = [ mpatches.Patch(color='C2', label='exp out buffer'),
                mpatches.Patch(color='C0', label='exp mem'), 
                mpatches.Patch(color='C1', label='accu mem')]
    fig.tight_layout()
    plt.legend(handles=patches, loc='upper left', fontsize=fsize)
    fig.savefig('res_fig/softmax_mem_teardown.pdf')
    plt.clf()

def v_compute_lat_teardown():
    fsize = 9
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    bar_width=2
    
    # define parallelism sweeping range
    parallelism = [2*(i+1) for i in range(8)]
    v_compute_lat_lst = []
    for p in parallelism:
        dpu = DpuModel(320, 768, 768, 64, p, p*12)
        print(dpu.compute_lat_teardown())
        v_compute_lat_lst.append(dpu.compute_lat_teardown())

    ax.bar(parallelism, [res[0] for res in v_compute_lat_lst], bar_width, color='C0')
    ax.bar(parallelism, [res[1] for res in v_compute_lat_lst], bar_width, bottom=[res[0] for res in v_compute_lat_lst], color='C1')
    ax.bar(parallelism, [res[2] for res in v_compute_lat_lst], bar_width, bottom=[res[0]+res[1] for res in v_compute_lat_lst], color='C2')
   
    ax.set_xlabel('parallelism', fontsize=fsize)
    ax.set_ylabel('latency/cycles', fontsize=fsize)
    ax.set_xticks(parallelism)
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    patches = [ mpatches.Patch(color='C0', label='input cycles'),
                mpatches.Patch(color='C1', label='adder tree cycles'), 
                mpatches.Patch(color='C2', label='accu cycles')]
    fig.tight_layout()
    plt.legend(handles=patches, loc='upper left', fontsize=fsize)
    fig.savefig('res_fig/v_compute_teardown.pdf')
    plt.clf()


def visualize_outer_product_intermediate_size(bert_hw_model: BertModel, word_size=16):
    fsize = 9
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})

    res = bert_hw_model.att_v_outer_product_intermediate_size()
    res = res * word_size / 1024.0
    ax.scatter(range(len(res)), res, color='C0', alpha=0.5, s=3)
   
    ax.set_xlabel('instances index', fontsize=fsize)
    ax.set_ylabel('intermediate data size/Kbits', fontsize=fsize)
    
    ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
    fig.tight_layout()
    fig.savefig('res_fig/outer_prod_intermediate_size.pdf')
    plt.clf()


def visual_heatmap_exps(bert_hw_model: BertModel):
    '''
    Plot the heat map to visualize the relation between each subwords in the
    self attention of each attention head in each layer

    expected data shape: (#layers, #heads, length, length)
    layers: layer_<0-11>
    sparsity_bar: threshold for sparsity calculation
    auto_scale: whether to auto scale the color bar
    binarize: if true, all values > sparsity_bar will be 1 and < will be 0
    '''
    fig_path = "./res_fig/"
    data = random.choice(bert_hw_model.exps)
    for layer_idx, layer in enumerate(data):
        fig, axs = plt.subplots(3, 4, figsize=(19, 12))
        print("Plotting heatmap for layer {}...".format(layer_idx))
        for head_idx, head in enumerate(layer):
            sparsity = (head == 0.0).sum() / head.flatten().shape[0]
            info = 'head_{}, max: {:.4f}, min: {:.4f}, sparsity: {:.4f}'.format(
                head_idx, np.amax(head), np.amin(head), sparsity)
            head = np.array((head > 0)).astype("float")
            ax = axs[int(head_idx/4), int(head_idx % 4)]
            
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            c = ax.pcolormesh(head)
            fig.colorbar(c, ax=ax)
            ax.set_title('\n'.join(wrap(info, 35)))

        fig.suptitle('Heatmap of Layer {}\'s exp out per head'.format(layer_idx), fontsize=21, y=0.99)
        fig.tight_layout()
        plt.savefig(fig_path+'exp_heatmap_layer{}.png'.format(layer_idx), dpi=600)
        plt.clf()
        plt.close(fig)


def compare_mvm_ratio_with_latency_with_given_dsps(bert_hw_model: BertModel, dsps=6840.0, mvm_dsp_percentage=0.70, plot_res=True):
    '''
    dsps: actual dsp slices on the chip, regardless of the data type.
    MVM mapping: Q head->K head->QK head->V head 
    '''
    mvm_dsps = floor(dsps * mvm_dsp_percentage)
    mvm_block_height_list = range(2, floor(mvm_dsps/2.0), 20)
    mvm_blocks = []
    for h in mvm_block_height_list:
        mvm_block_width = floor(mvm_dsps/h)
        for w in range(mvm_block_width, 2, -1):
            desired_dsps = bert_hw_model.matmul_res_qktrans_per_head_stratix((h, w))[0]
            if (mvm_dsps - 50) <= desired_dsps < (mvm_dsps + 50):
                mvm_blocks.append((h, w, desired_dsps, float(w)/h))
                break
    
    # flatten exps:
    real_exp_data = []
    for inst in bert_hw_model.exps:
        num_layers, num_heads, num_rows, _ = inst.shape
        real_exp = inst.reshape((num_layers * num_heads, num_rows, num_rows))
        for h in real_exp: real_exp_data.append(h)

    res_wh, res_lat_ddl, res_relative_lat = [], [], []
    for mvm_block_height, mvm_block_width, actual_mvm_dsps, mvm_block_wh_ratio in mvm_blocks:
        max_softmax_dsp = dsps - actual_mvm_dsps
        softmax_p = range(2, mvm_block_width)
        softmax_possible_p = [p for p in softmax_p if bert_hw_model.baseline_softmax_resource(p, bert_hw_model.max_seq_len)[0] < max_softmax_dsp]
        softmax_p = softmax_possible_p[-1]
        qktrans_incycle, qk_trans_addertree, qk_trans_adder = \
            bert_hw_model.matmul_lat_qktrans_per_head_stratix(320, blk=(mvm_block_height, mvm_block_width))
        qk_trans_lat = qk_trans_addertree + qk_trans_adder + bert_hw_model.DIV_LAT

        softmax_lat = np.mean([qk_trans_lat + bert_hw_model.baseline_softmax_lat(h, softmax_p) for h in real_exp_data])
        # softmax_lat = qk_trans_lat + bert_hw_model.baseline_softmax_lat(pa=softmax_p)

        softmax_ddl = qktrans_incycle + sum(bert_hw_model.matmul_lat_qkv_per_head_stratix(320, blk=(mvm_block_height, mvm_block_width)))

        res_wh.append(mvm_block_wh_ratio)
        res_lat_ddl.append(softmax_ddl-softmax_lat)
        res_relative_lat.append((softmax_ddl-softmax_lat) / softmax_ddl)
        print(mvm_block_wh_ratio, actual_mvm_dsps)
        print(softmax_ddl, softmax_lat)

    if plot_res:
        fsize = 9
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        matplotlib.rcParams.update({'xtick.labelsize': fsize})
        matplotlib.rcParams.update({'ytick.labelsize': fsize})
        matplotlib.rcParams['lines.markersize'] = 3

        ax.plot(res_wh[1:], res_lat_ddl[1:], linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(res_wh[1:], res_relative_lat[1:], linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8)

        ax.set_xlabel(r'$\frac{mvm\ block\ width}{mvm\ block\ height}$', fontsize=fsize)
        ax.set_ylabel('deadline-softmax_lat', fontsize=fsize)
        ax2.set_ylabel(r'$\frac{deadline-softmax\_lat}{deadline}$')
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)

        fig.tight_layout()
        fig.savefig('res_fig/softmax_ddl_head_by_head.pdf')
        plt.cla()

    # plot overhead vs. softmax resource
    cur_wh_ratio = float('inf')
    for w, h, _, wh_ratio in mvm_blocks:
        if abs(wh_ratio - 1.0) < abs(cur_wh_ratio-1.0):
            cur_wh_ratio = wh_ratio
            mvm_block_width = w
            mvm_block_height = h        

    softmax_p = range(2, mvm_block_width)
    softmax_possible_p = [p for p in softmax_p if bert_hw_model.baseline_softmax_resource(p, bert_hw_model.max_seq_len)[0] < max_softmax_dsp]
    qktrans_incycle, qk_trans_addertree, qk_trans_adder = \
            bert_hw_model.matmul_lat_qktrans_per_head(320, blk=(mvm_block_height, mvm_block_width))
    qk_trans_lat = qk_trans_addertree + qk_trans_adder + bert_hw_model.DIV_LAT
    softmax_lat = [np.mean([qk_trans_lat + bert_hw_model.baseline_softmax_lat(h, p) for h in real_exp_data]) for p in softmax_possible_p]
    softmax_lat = np.array(softmax_lat)

    sparse_softmax_possible_p = \
        [p for p in softmax_p if bert_hw_model.softmax_resources(p, p, bert_hw_model.max_seq_len, 4)[0] < max_softmax_dsp]
    sparse_softmax_lats = \
        [np.mean([qk_trans_addertree + bert_hw_model.softmax_lat(h, p1=p, p2=p) for h in real_exp_data]) for p in sparse_softmax_possible_p]
    sparse_softmax_lats = np.array(sparse_softmax_lats)
    # softmax_lat = [qk_trans_lat + bert_hw_model.baseline_softmax_lat(pa=p) for p in softmax_possible_p]
    softmax_ddl = qktrans_incycle + sum(bert_hw_model.matmul_lat_qkv_per_head(320, blk=(mvm_block_height, mvm_block_width)))
    softmax_dsps = np.array([bert_hw_model.baseline_softmax_resource(p, bert_hw_model.max_seq_len)[0] for p in softmax_possible_p])
    sparse_softmax_dsps = np.array([bert_hw_model.softmax_resources(p, p, 320, 4)[0] for p in sparse_softmax_possible_p])
    softmax_overhead = [(softmax_ddl - lat) for lat in softmax_lat]
    sparse_softmax_overhead = [(softmax_ddl - lat) for lat in sparse_softmax_lats]

    if plot_res:
        fsize = 9
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        matplotlib.rcParams.update({'xtick.labelsize': fsize})
        matplotlib.rcParams.update({'ytick.labelsize': fsize})
        matplotlib.rcParams['lines.markersize'] = 3

        ax.plot(softmax_dsps, softmax_overhead, linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8, label='baseline')
        ax.plot(sparse_softmax_dsps, sparse_softmax_overhead, linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8, label='sparse softmax')
        # ax2 = ax.twinx()
        # ax2.plot(res_wh[1:], res_relative_lat[1:], linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8)

        ax.set_xlabel(r'DSPs for softmax', fontsize=fsize)
        ax.set_ylabel('deadline-softmax_lat', fontsize=fsize)
        # ax2.set_ylabel(r'$\frac{deadline-softmax\_lat}{deadline}$')
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
        plt.legend(loc='lower right', fontsize=fsize)
        fig.tight_layout()
        fig.savefig('res_fig/softmax_ddl_head_by_head_pe.pdf')
        plt.cla()

    res = {'ddl': softmax_ddl, 'baseline_lat': softmax_lat, 'baseline_dsps': softmax_dsps, \
            'sparse_lat': sparse_softmax_lats, 'sparse_dsps': sparse_softmax_dsps}
    return res

def compare_mvm_ratio_delayed_v_with_latency_with_given_dsps(bert_hw_model:BertModel, dsps: int, mvm_dsp: int, plot_res=True):
    
    equi_mvm_dsps = mvm_dsp * 30
    mvm_block_height_list = range(2, floor(equi_mvm_dsps/2.0), 20)
    mvm_blocks = []
    for h in mvm_block_height_list:
        mvm_block_width = floor(equi_mvm_dsps/h)
        for w in range(mvm_block_width, 2, -1):
            desired_dsps = bert_hw_model.matmul_res_qktrans_per_head_stratix((h, w))[0]
            if (mvm_dsp - 2) <= desired_dsps < (mvm_dsp + 2):
                mvm_blocks.append((h, w, desired_dsps, float(w)/h))
                break
    
    mvm_blocks.sort(key=lambda x: x[-1])
    res_wh, res_lat_ddl, res_relative_lat, res_softmax_lat = [], [], [], []
    for mvm_block_height, mvm_block_width, actual_mvm_dsps, mvm_block_wh_ratio in mvm_blocks[:10]:
        # check if softmax latency can be hidden by Q K and QK
        max_softmax_dsp = dsps - actual_mvm_dsps
        qktrans_incycle, qk_trans_addertree, qk_trans_adder = \
                bert_hw_model.matmul_lat_qktrans_per_head_stratix(320, blk=(mvm_block_height, mvm_block_width), ideal=True)
        q_incycle, q_addertree, q_adder = \
                bert_hw_model.matmul_lat_qkv_per_head_stratix(320, blk=(mvm_block_height, mvm_block_width), ideal=True)
        single_stage_softmax_deadline = qktrans_incycle + q_incycle * 3

        softmax_p = np.arange(2, mvm_block_width)
        softmax_row_para = np.arange(1, 10)
        softmax_lat_candidates = []
        for r, p in product(softmax_row_para, softmax_p):
            softmax_dsp = bert_hw_model.baseline_softmax_resource(p, ceil(bert_hw_model.max_seq_len / r))[0]
            if softmax_dsp < max_softmax_dsp:
                if bert_hw_model.exps is None:
                    softmax_stg1_incycle = ceil(bert_hw_model.max_seq_len/r) * ceil(bert_hw_model.max_seq_len/p)
                else:
                    softmax_stg1_incycle = np.mean([ceil(h.shape[-1]/r)* ceil(float(h.shape[-1])/p) \
                                            for h in bert_hw_model.exps])
                softmax_lat = softmax_stg1_incycle + qk_trans_adder + qk_trans_addertree
                softmax_lat_candidates.append({'dsps': softmax_dsp, 'latency': softmax_lat})

        if len(softmax_lat_candidates) > 0:
            softmax_lat_candidates.sort(key=lambda x: x['latency'])
            final_softmax_lat = softmax_lat_candidates[0]['latency']
            res_wh.append(mvm_block_wh_ratio)
            res_softmax_lat.append(final_softmax_lat)
            res_lat_ddl.append(single_stage_softmax_deadline)
            res_relative_lat.append((single_stage_softmax_deadline - final_softmax_lat) / single_stage_softmax_deadline)
    
        # if softmax_lat <= single_stage_softmax_deadline:
        #     res_lat_ddl.append(single_stage_softmax_deadline - softmax_lat)
        #     res_relative_lat.append(abs(single_stage_softmax_deadline - softmax_lat) / single_stage_softmax_deadline)
        # else:
        #     pass
    
    if plot_res and len(res_softmax_lat) > 1:
        fsize = 9
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        matplotlib.rcParams.update({'xtick.labelsize': fsize})
        matplotlib.rcParams.update({'ytick.labelsize': fsize})
        matplotlib.rcParams['lines.markersize'] = 3

        ax.plot(res_wh[1:], res_softmax_lat[1:], linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8, label='baseline softmax')
        # ax2 = ax.twinx()
        # ax2.plot(res_wh[1:], res_lat_ddl[1:], linestyle='--', color='black', marker='s', linewidth=1, alpha=0.2)

        ax.set_xlabel(r'$\frac{mvm\ block\ width}{mvm\ block\ height}$', fontsize=fsize)
        ax.set_ylabel('softmax latency', color='C0', fontsize=fsize)
        # ax2.set_ylabel('softmax deadline', color='black', fontsize=fsize)
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
        plt.legend(loc='upper right', fontsize=fsize)
        fig.tight_layout()
        fig.savefig('res_fig/softmax_ddl_delayed_v_mvm_ratio.pdf')
        plt.cla()

    # plot overhead vs. softmax resource
    cur_wh_ratio = float('inf')
    for w, h, act_mvm_dsps, wh_ratio in mvm_blocks:
        if abs(wh_ratio - 1.0) < abs(cur_wh_ratio-1.0):
            cur_wh_ratio = wh_ratio
            mvm_block_width = w
            mvm_block_height = h
            actual_mvm_dsps = act_mvm_dsps

    qktrans_incycle, qk_trans_addertree, qk_trans_adder = \
            bert_hw_model.matmul_lat_qktrans_per_head_stratix(320, blk=(mvm_block_height, mvm_block_width), ideal=True)
    q_incycle, q_addertree, q_adder = \
            bert_hw_model.matmul_lat_qkv_per_head_stratix(320, blk=(mvm_block_height, mvm_block_width), ideal=True)

    softmax_p = np.arange(2, mvm_block_width, 20)
    softmax_row_para = np.arange(1, 50, 5)
    b_softmax_lat_candidates, s_softmax_lat_candidates = [], []
    max_softmax_dsp = dsps - actual_mvm_dsps
    for r, p in product(softmax_row_para, softmax_p):
        b_softmax_dsp = bert_hw_model.baseline_softmax_resource(p, ceil(bert_hw_model.max_seq_len / r))[0]
        s_softmax_dsp = bert_hw_model.softmax_resources(p, p, ceil(bert_hw_model.max_seq_len / r), 4)[0]
        if bert_hw_model.exps is None:
            softmax_stg1_incycle = ceil(bert_hw_model.max_seq_len/r) * ceil(bert_hw_model.max_seq_len/p)
        else:
            softmax_stg1_incycle = np.mean([ceil(h.shape[-1]/r) * ceil(float(h.shape[-1])/p) \
                                        for h in bert_hw_model.exps])

        if b_softmax_dsp < max_softmax_dsp:
            b_softmax_lat = softmax_stg1_incycle + qk_trans_adder + qk_trans_addertree
            b_softmax_att_total_lat = bert_hw_model.attention_lat_stratix(float(mvm_dsp), (r, p), softmax_type="baseline")
            if not isnan(b_softmax_att_total_lat):
                b_softmax_lat_candidates.append({'dsp': b_softmax_dsp, 'softmax_lat': b_softmax_lat, 'att_lat': b_softmax_att_total_lat})


        if s_softmax_dsp < max_softmax_dsp:
            sparse_softmax_lat = softmax_stg1_incycle + qk_trans_adder + qk_trans_addertree
            s_softmax_att_total_lat = bert_hw_model.attention_lat_stratix(float(mvm_dsp), (r, p), softmax_type="sparse")
            if not isnan(s_softmax_att_total_lat):
                s_softmax_lat_candidates.append({'dsp': s_softmax_dsp, 'softmax_lat': sparse_softmax_lat, 'att_lat': s_softmax_att_total_lat})

    # softmax_stg1_incycle = bert_hw_model.max_seq_len * np.ceil(float(bert_hw_model.max_seq_len) / softmax_possible_p)
    if len(b_softmax_lat_candidates) < 1 or len(s_softmax_lat_candidates) < 1:
        return None
    else:
        final_b_softmax_lat = min(b_softmax_lat_candidates, key=lambda x: x['att_lat'])
        final_s_softmax_lat = min(s_softmax_lat_candidates, key=lambda x: x['att_lat'])


    softmax_ddl = qktrans_incycle + q_incycle * 3 + qk_trans_addertree + qk_trans_adder
    softmax_overhead = softmax_ddl - final_b_softmax_lat['softmax_lat']
    sparse_softmax_overhead = softmax_ddl - final_s_softmax_lat['softmax_lat']

    res = {'ddl': softmax_ddl, 'baseline_lat': final_b_softmax_lat['softmax_lat'], 'baseline_dsps': final_b_softmax_lat['dsp'], \
            'baseline_att_lat': final_b_softmax_lat['att_lat'], 
            'sparse_lat': final_s_softmax_lat['softmax_lat'], 'sparse_dsps': final_s_softmax_lat['dsp'], \
            'sparse_att_lat': final_s_softmax_lat['att_lat']}
    return res

def sweep_mvm_softmax_ratio(bert_hw_model: BertModel, num_dsps = 6840.0, \
                                schedule=compare_mvm_ratio_delayed_v_with_latency_with_given_dsps, lat_type="softmax", plot_res=True):
    '''
    sweeping across different mvm/softmax ratio
    '''
    num_mvm_dsps_candidates = np.arange(100, num_dsps, floor((num_dsps-100)/40))

    ddls, num_mvm_dsps = [], []
    baseline_lat, sparse_softmax_lat = [], []
    mvm_dynautil_baseline, mvm_dynautil_sparse = [], []
    for mvm_dsp in tqdm(num_mvm_dsps_candidates):
        latency_res = schedule(bert_hw_model, dsps=num_dsps, mvm_dsp=mvm_dsp, plot_res=False)
        if latency_res is not None:
            ddls.append(latency_res['ddl'])
            mvm_only_lat = bert_hw_model.attention_mvm_only_lat_stratix(float(mvm_dsp))
            num_mvm_dsps.append(mvm_dsp)
            if lat_type == "softmax":
                baseline_lat.append(latency_res['baseline_lat'])
                sparse_softmax_lat.append(latency_res['sparse_lat'])
            elif lat_type == "self_attention":
                mvm_dynautil_baseline.append(mvm_only_lat / latency_res['baseline_att_lat'])
                mvm_dynautil_sparse.append(mvm_only_lat / latency_res['sparse_att_lat'])
                baseline_lat.append(latency_res['baseline_att_lat'])
                sparse_softmax_lat.append(latency_res['sparse_att_lat'])

    if plot_res:
        fsize = 9
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        matplotlib.rcParams.update({'xtick.labelsize': fsize})
        matplotlib.rcParams.update({'ytick.labelsize': fsize})
        matplotlib.rcParams['lines.markersize'] = 3

        ax.plot(num_mvm_dsps, baseline_lat, linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8, label='baseline softmax')
        ax.plot(num_mvm_dsps, sparse_softmax_lat, linestyle = '-', color='C0', marker='s', linewidth=1, alpha=0.8, label='sparse softmax')
        if lat_type == "softmax":
            ax.plot(num_mvm_dsps, ddls, linestyle='--', color='black', marker='s', linewidth=1, alpha=0.8, label='deadline')

        ax.set_xlabel('mvm ai tensors', fontsize=fsize)
        # ax.set_xlabel('softmax dsps', fontsize=fsize)
        ax.set_ylabel('latency', fontsize=fsize)
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
        plt.legend(loc='upper left', fontsize=fsize)
        fig.tight_layout()
        file_name = lat_type + f"_latency_mvm_ai_tensor_tc{int(num_dsps)}_l{int(bert_hw_model.max_seq_len)}.pdf"
        fig.savefig("res_fig/att_latency_analysis/" + file_name)
        plt.cla()

    min_baseline_mvm_dynautil = 0
    min_sparse_mvm_dynautil = 0
    if lat_type == "self_attention":
        min_baseline_mvm_dynautil = mvm_dynautil_baseline[baseline_lat.index(min(baseline_lat))]
        min_sparse_mvm_dynautil = mvm_dynautil_sparse[sparse_softmax_lat.index(min(sparse_softmax_lat))]

    return {'lat': np.amin(baseline_lat), 'mvm_util': min_baseline_mvm_dynautil}, \
            {'lat': np.amin(sparse_softmax_lat), 'mvm_util': min_sparse_mvm_dynautil}

def sweep_dsp_budget_latency(bert_hw_model: BertModel, plot_res=False):
    dsp_budget_list = np.arange(200, 4000, 200)
    baseline_lat, sparse_lat = [], []
    for dsp_budget in dsp_budget_list:
        min_baseline_lat, min_sparse_lat = sweep_mvm_softmax_ratio(bert_hw_model, num_dsps=dsp_budget, \
            schedule=compare_mvm_ratio_delayed_v_with_latency_with_given_dsps, lat_type="self_attention", plot_res=False)
        baseline_lat.append(min_baseline_lat)
        sparse_lat.append(min_sparse_lat)

    if plot_res:
        list_from_dicts = lambda x, k: [i[k] for i in x]

        fsize = 9
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        matplotlib.rcParams.update({'xtick.labelsize': fsize})
        matplotlib.rcParams.update({'ytick.labelsize': fsize})
        matplotlib.rcParams['lines.markersize'] = 3

        ax2 = ax.twinx()
        curs = []
        baseline_lat_sec = np.array(list_from_dicts(baseline_lat, 'lat')) * (1 / bert_hw_model.freq) * 1e-6
        curs += ax.plot(dsp_budget_list, baseline_lat_sec, linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8, label='baseline softmax')
        curs += ax2.plot(dsp_budget_list, list_from_dicts(baseline_lat, 'mvm_util'), linestyle='--', color='C1', marker='s', linewidth=1, alpha=0.6, label='baseline mvm util')
        sparse_lat_sec = np.array(list_from_dicts(sparse_lat, 'lat')) * (1 / bert_hw_model.freq) * 1e-6
        curs += ax.plot(dsp_budget_list, sparse_lat_sec, linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8, label='sparse softmax')
        curs += ax2.plot(dsp_budget_list, list_from_dicts(sparse_lat, 'mvm_util'), linestyle='--', color='C0', marker='s', linewidth=1, alpha=0.6, label='sparse mvm util')

        labels = [l.get_label() for l in curs]
        ax.legend(curs, labels, loc='upper right', bbox_to_anchor=(0.98, 0.85), fontsize=fsize)

        ax.set_xlabel('ai tensors budget', fontsize=fsize)
        ax.set_ylabel('latency/secs', fontsize=fsize)
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        ax2.set_ylim(ymin=0, ymax=1.0)
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
        fig.tight_layout()
        file_name = f"att_latency_ai_tensor_budget_l{str(bert_hw_model.max_seq_len)}.pdf"
        fig.savefig("res_fig/" + file_name)
        plt.cla()
        
    return dsp_budget_list, baseline_lat, sparse_lat

def sweep_seq_len_vs_budget(bert_models: list, plot_res = False):
    if plot_res:
        list_from_dicts = lambda x, k: [i[k] for i in x]

        fsize = 9
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        matplotlib.rcParams.update({'xtick.labelsize': fsize})
        matplotlib.rcParams.update({'ytick.labelsize': fsize})
        matplotlib.rcParams['lines.markersize'] = 3
        ax2 = ax.twinx()

        for idx, bert_model in enumerate(bert_models):
            dsp_budget, baseline, sparse = sweep_dsp_budget_latency(bert_model, plot_res=False)
            baseline_lat = np.array(list_from_dicts(baseline, 'lat')) * (1./bert_model.freq) * 1e-6
            sparse_lat = np.array(list_from_dicts(sparse, 'lat')) * (1./bert_model.freq) * 1e-6
            if idx == 0:
                ax.plot(dsp_budget, baseline_lat, linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8, label='baseline softmax')
                ax.plot(dsp_budget, sparse_lat, linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8, label='sparse softmax')
            else:
                ax.plot(dsp_budget, baseline_lat, linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8)
                ax.plot(dsp_budget, sparse_lat, linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8)

            ax2.plot(dsp_budget, list_from_dicts(baseline, 'mvm_util'), linestyle='--', color='C1', marker='s', linewidth=1, alpha=0.6)
            ax2.plot(dsp_budget, list_from_dicts(sparse, 'mvm_util'), linestyle='--', color='C0', marker='s', linewidth=1, alpha=0.6)
            ax.text(dsp_budget[0]+50, baseline_lat[0], f"len:{bert_model.max_seq_len}", alpha=0.8)

        ax.set_xlabel('ai tensors budget', fontsize=fsize)
        ax.set_ylabel('latency/sec', fontsize=fsize)
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)
        plt.legend(loc='upper right', fontsize=fsize)
        fig.tight_layout()
        file_name = f"att_latency_ai_tensor_budget_diff_seqlen.pdf"
        fig.savefig("res_fig/" + file_name)
        plt.cla()

if __name__ == '__main__':
    bert_hw_model = BertModel(read_exp_samples=False, num_layers=12, num_heads=12, max_seq_len=512)
    # bert_hw_model = BertModel(read_exp_samples=True)
    # filtered_insts = []
    # for inst in bert_hw_model.exps:
    #     if inst.shape[-1] > 300:
    #         filtered_insts.append(inst)

    # bert_hw_model.exps = filtered_insts
    # bert_hw_model.analyzer_void_columns()
    # compare_naive_softmax_heads(bert_hw_model)
    # compare_naive_softmax_parallel(bert_hw_model)
    # compare_lat_res_models(bert_hw_model, resource_type=['dsp'], l_range=np.arange(100, 2, -2), hw_modeling_type=["softmax", "baseline softmax", "value mvm"])
    # compare_lat_res_models(bert_hw_model, resource_type=['dsp'], l_range=np.arange(100, 2, -10), hw_modeling_type=["softmax", "baseline softmax"])
    # visualize_outer_product_intermediate_size(bert_hw_model)
    # visual_heatmap_exps(bert_hw_model)
    # mem_teardown(bert_hw_model)
    # v_compute_lat_teardown()
    # explore_p1_p2(bert_hw_model)
    # compare_softmax_with_model_len()
    # compare_mvm_ratio_with_latency_with_given_dsps(bert_hw_model, mvm_dsp_percentage=0.9)
    # compare_mvm_ratio_delayed_v_with_latency_with_given_dsps(bert_hw_model, dsps=3960.0, mvm_dsp=3900, plot_res=True)
    # bd_res = bert_hw_model.attention_bandwidth_stratix(3900.0, (8, 4), 'baseline')
    # print("test bandwidth: ", bd_res)
    # sweep_mvm_softmax_ratio(bert_hw_model, num_dsps=3960.0, \
    #      schedule=compare_mvm_ratio_delayed_v_with_latency_with_given_dsps, lat_type="self_attention")
    # sweep_dsp_budget_latency(bert_hw_model, plot_res=True)
    bert_models = [BertModel(read_exp_samples=False, num_layers=12, num_heads=12, max_seq_len=i) for i in [512]]
    sweep_dsp_budget_latency(bert_models[0], plot_res=True)
    # sweep_seq_len_vs_budget(bert_models, plot_res=True)