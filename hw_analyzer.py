from math import ceil, floor, exp, log2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import random

from itertools import product

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
            
    
    # for layer_idx, layer in enumerate(data):
    #     fig, axs = plt.subplots(3, 4, figsize=(19, 12))
    #     print("Plotting heatmap for layer {}...".format(layer_idx))
    #     for head_idx, head in enumerate(layer[0]):
    #         sparsity = (head <= sparsity_bar).sum() / head.flatten().shape[0]
    #         info = 'head_{}, max: {:.4f}, min: {:.4f}, spars: {:.4f}, sparsity_bar: {:.4f}'.format(
    #             head_idx, np.amax(head), np.amin(head), sparsity, sparsity_bar)
    #         if binarize:
    #             head = np.array((head > sparsity_bar)).astype("float")
    #         ax = axs[int(head_idx/4), int(head_idx % 4)]
    #         ax.invert_yaxis()
    #         ax.xaxis.tick_top()
    #         c = ax.pcolormesh(head) if auto_scale else ax.pcolormesh(
    #             head, vmin=0.0, vmax=1.0)
    #         fig.colorbar(c, ax=ax)
    #         ax.set_title('\n'.join(wrap(info, 35)))

    #     fig.suptitle('Heatmap of Layer {}\'s Attention per head (batch aggregation={}, {})'
    #                  .format(layer_idx, layer_aggregration, attached_title), fontsize=21, y=0.99)
    #     fig.tight_layout()
    #     fig_path = RES_FIG_PATH+"auto_scale_" if auto_scale else RES_FIG_PATH
    #     fig_path = fig_path+"bin_" if binarize else fig_path
    #     plt.savefig(fig_path+'heatmap_layer{}.png'.format(layer_idx), dpi=600)
    #     plt.clf()
    #     plt.close(fig)


def compare_lat_res_models(bert_hw_model: BertModel, resource_type = ['dsp', 'mem'], hw_modeling_type = ["softmax", "baseline softmax", "value mvm"], l_range = [200, 100, 50, 25]):
    fsize = 9
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    matplotlib.rcParams['lines.markersize'] = 5

    # define parallelism sweeping range
    softmax_range = [2**i for i in range(8)]
    baseline_range = [2**i for i in range(6)]
    mvm_range = [2**(i) for i in np.arange(4, 7)]

    
    print("mvm range: ", mvm_range)

    softmax_res_lst = [[bert_hw_model.softmax_resources(p, p, l, 4) for p in softmax_range] for l in l_range]
    baseline_softmax_res_lst = [[bert_hw_model.baseline_softmax_resource(p, l) for p in baseline_range] for l in l_range]
    value_res_lst = [bert_hw_model.matmul_res_qkv_per_head(blk=(w, w)) for w in mvm_range]

    softmax_lat_lst = []
    if "softmax" in hw_modeling_type:
        for l in l_range:
            softmax_lat_lst_for_l = []
            for p in softmax_range:
                temp_softmax_lats = None
                for inst in bert_hw_model.exps:
                    num_layers, num_heads, num_rows, _ = inst.shape
                    heads = inst.reshape((num_layers * num_heads, num_rows, num_rows))

                    temp_heads_lats = np.array([bert_hw_model.softmax_lat(dat[:l], p1=p, p2=p) for dat in heads])
                    temp_softmax_lats = temp_heads_lats if temp_softmax_lats is None \
                                            else np.concatenate([temp_softmax_lats, temp_heads_lats], axis=0)

                softmax_lat_lst_for_l.append(np.mean(temp_softmax_lats))
            
            softmax_lat_lst.append(softmax_lat_lst_for_l)

    baseline_softmax_lat_lst = []
    if "baseline softmax" in hw_modeling_type:
        for l in l_range:
            baseline_softmax_lat_lst_for_l = []
            for p in baseline_range:
                temp_baseline_softmax_lats = None
                for inst in bert_hw_model.exps:
                    num_layers, num_heads, num_rows, _ = inst.shape
                    heads = inst.reshape((num_layers * num_heads, num_rows, num_rows))

                    temp_heads_lats = np.array([bert_hw_model.baseline_softmax_lat(dat[:l], pa=p) for dat in heads])
                    temp_baseline_softmax_lats = temp_heads_lats if temp_baseline_softmax_lats is None \
                                                    else np.concatenate([temp_baseline_softmax_lats, temp_heads_lats], axis=0)

                baseline_softmax_lat_lst_for_l.append(np.mean(temp_baseline_softmax_lats))

            baseline_softmax_lat_lst.append(baseline_softmax_lat_lst_for_l)

    v_compute_head_lat_lst = []
    if "value mvm" in hw_modeling_type:
        for p in mvm_range:
            temp_v_compute_head_lat = []
            for inst in bert_hw_model.exps:
                num_layers, num_heads, num_rows, _ = inst.shape
                temp_v_compute_head_lat.append(bert_hw_model.matmul_lat_qkv_per_head(num_rows, blk=(p, p)))

            v_compute_head_lat_lst.append(np.mean(temp_v_compute_head_lat))

    legend_lines = [mlines.Line2D([], [], color='C0', label='our softmax', marker='s', linestyle='-'),
                    mlines.Line2D([], [], color='C1', label='baseline softmax', marker='s', linestyle='-'),
                    mlines.Line2D([], [], color='C2', label='Value computation', marker='s', linestyle='-')
                ]

    # plot lines
    if 'dsp' in resource_type:
        if len(softmax_lat_lst) > 0:
            for l, softmax_lat, softmax_res in zip(l_range, softmax_lat_lst, softmax_res_lst):
                ax.plot([dsp[0] for dsp in softmax_res], softmax_lat, linestyle='-', color='C0', marker='s', linewidth=1, alpha=0.8)
                ax.text(softmax_res[0][0]+3, softmax_lat[0]-3, f"row={ceil(320/l)}")
        if len(baseline_softmax_lat_lst) > 0:
            for l, baseline_softmax_lat, baseline_softmax_res in zip(l_range, baseline_softmax_lat_lst, baseline_softmax_res_lst):
                ax.plot([dsp[0] for dsp in baseline_softmax_res], baseline_softmax_lat, linestyle='-', color='C1', marker='s', linewidth=1, alpha=0.8)
                ax.text(baseline_softmax_res[0][0]+3, baseline_softmax_lat[0]-3, f"row={ceil(320/l)}")

        if len(v_compute_head_lat_lst) > 0:
            ax.plot([dsp[0] for dsp in value_res_lst], v_compute_head_lat_lst, linestyle='-', color='C2', marker='s', linewidth=1)

        ax.set_xlabel('DSP Usage', fontsize=fsize)
        ax.set_ylabel('latency', fontsize=fsize)
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)

        fig.tight_layout()
        plt.legend(handles=legend_lines, loc='upper right', fontsize=fsize)
        fig.savefig('res_fig/softmax_lat_res_analyze_pe.pdf')
        plt.cla()
    
    if 'mem' in resource_type:
        if len(softmax_lat_lst) > 0:
            for l, softmax_lat, softmax_res in zip(l_range, softmax_lat_lst, softmax_res_lst):
                ax.plot([mem[1] for mem in softmax_res], softmax_lat, linestyle='-', color='C0', marker='s', linewidth=1)
                ax.text(softmax_res[0][0]+3, softmax_lat[0]-3, f"row={ceil(320/l)}")
        if len(baseline_softmax_lat_lst) > 0:
            for l, baseline_softmax_lat, baseline_softmax_res in zip(l_range, baseline_softmax_lat_lst, baseline_softmax_res_lst):
                ax.plot([mem[1] for mem in baseline_softmax_res], baseline_softmax_lat, linestyle='-', color='C1', marker='s', linewidth=1)
                ax.text(baseline_softmax_res[0][0]+3, baseline_softmax_lat[0]-3, f"row={ceil(320/l)}")
        if len(v_compute_head_lat_lst) > 0:
            ax.plot([mem[1] for mem in value_res_lst], v_compute_head_lat_lst, linestyle='-', color='C2', marker='s', linewidth=1)

        ax.set_xlabel('Mem Usage/Kb', fontsize=fsize)
        ax.set_ylabel('latency', fontsize=fsize)
        ax.grid(linestyle='--', color='grey', alpha=0.5, linewidth=1)

        fig.tight_layout()
        plt.legend(handles=legend_lines, loc='upper left', fontsize=fsize)
        fig.savefig('res_fig/softmax_lat_res_analyze_mem.pdf')
        plt.cla()


def mem_teardown(bert_hw_model: BertModel):    
    fsize = 9
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    matplotlib.rcParams.update({'xtick.labelsize': fsize})
    matplotlib.rcParams.update({'ytick.labelsize': fsize})
    bar_width=2
    
    l = 110
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


if __name__ == '__main__':
    bert_hw_model = BertModel(read_exp_samples=True)
    # bert_hw_model.analyzer_void_columns()
    # compare_naive_softmax_heads(bert_hw_model)
    # compare_naive_softmax_parallel(bert_hw_model)
    compare_lat_res_models(bert_hw_model, resource_type=['dsp'], l_range=[100, 50, 20])
    # mem_teardown(bert_hw_model)
    # v_compute_lat_teardown()
    # explore_p1_p2(bert_hw_model)