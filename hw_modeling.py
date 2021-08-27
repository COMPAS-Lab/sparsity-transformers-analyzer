from math import ceil, floor, exp, log2, gcd, sqrt, pow
from re import template
import numpy as np
import matplotlib.pyplot as plt
import random
import logging

# Helper functions:
def log2Up(x):
    return float(ceil(log2(x)))
def log2Down(x):
    return float(floor(log2(x)))

def lcm(a: float, b: float):
    return a*b/gcd(a, b)

def dspToAlu(dsp, dtype: str):
    return dsp / 2.0 if dtype == 'float32' else dsp

class MatMulDimErr(Exception):
    pass

class DpuModel:
    '''
    This class is used to construct DPU hardware model which does AxB
    '''
    num_mults_per_dpu = 0
    num_dpus = 0
    a_w = 0
    a_h = 0
    b_w = 0
    b_h = 0

    COMP_LAT = 1.0
    ADDER_LAT = 1.0
    MULT_LAT = 1.0

    ADDER_RES = 1
    MULT_RES = 1/3
    DIV_RES = 0
    COMP_RES = 0

    WORD_SIZE = 1

    def __init__(self, a_h, a_w, b_h, b_w, blk_h, blk_w):
        self.a_w = a_w
        self.a_h = a_h
        self.b_w = b_w
        self.b_h = b_h
        self.num_mults_per_dpu = blk_h
        self.num_dpus = blk_w

        if self.b_h != self.a_w:
            raise MatMulDimErr("Error: mat mul dim mismatch")
            

    # derived parameters
    def input_len_per_cycle(self): return self.num_mults_per_dpu
    def num_wei_per_dpu(self, ideal=False): 
        if ideal:
            return self.b_w / self.num_dpus
        else:
            return float(ceil(self.b_w / self.num_dpus))

    def num_grps_in_a(self, ideal=False): 
        if ideal:
            return self.a_w / self.input_len_per_cycle()
        else:
            return float(ceil(self.a_w / self.input_len_per_cycle()))
    
    def add_lat(self): return self.ADDER_LAT
    def mult_lat(self): return self.MULT_LAT
    def adder_tree_lat(self): 
        return self.add_lat() * log2Up(self.input_len_per_cycle())

    def elemul_addtree_lat(self): return self.mult_lat() + self.adder_tree_lat()
    def dpu_lat(self): return self.elemul_addtree_lat() + self.add_lat()
    def compute_lat(self, ideal=False):
        input_cycles = self.num_grps_in_a(ideal=ideal) * self.num_wei_per_dpu(ideal=ideal) * self.a_h
        return input_cycles + self.dpu_lat()
    
    def compute_lat_teardown(self, ideal=False):
        input_cycles = self.num_grps_in_a(ideal=ideal) * self.num_wei_per_dpu(ideal=ideal) * self.a_h
        adder_tree_cycles = self.elemul_addtree_lat()
        adder_lat = self.add_lat()
        return input_cycles, adder_tree_cycles, adder_lat

    def compute_resource(self):
        dpu_mults = self.num_mults_per_dpu * self.num_dpus
        dpu_adders, rest_elems = 0.0, self.num_mults_per_dpu
        while rest_elems > 0.0:
            dpu_adders += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))

        dpu_adders = (dpu_adders-1.0) * self.num_dpus
        # print(f"dpu mults: {dpu_mults}, dpu adders: {dpu_adders}")

        mem_usage = self.b_w * self.b_h * self.WORD_SIZE / 1024.
        return dpu_mults * self.MULT_RES + dpu_adders * self.ADDER_RES, mem_usage

class StratixDpuModel(DpuModel):
    '''
    This class is used to construct Intel Stratix DPU hardware model which does AxB
    '''
    num_mults_per_dpu = 0
    num_dpus = 0
    a_w = 0
    a_h = 0
    b_w = 0
    b_h = 0

    # FIXME: check the availability here
    COMP_LAT = 3.0
    ADDER_LAT = 3.0
    MULT_LAT = 3.0
    MAC_LAT = 4.0

    ADDER_RES = 1/3.0
    MULT_RES = 1.0/30.0
    DIV_RES = 0
    COMP_RES = 0

    URAM_WR_LAT = 1.
    DDR_RD_LAT = 150.

    WORD_SIZE = 1
        
    # derived parameters
    def compute_lat(self, ideal=False, mem_init=False):
        # matrix b init time
        b_size = self.b_w * self.b_h
        if ideal == False:
            mat_b_init_lat = b_size / self.input_len_per_cycle()
        else:
            mat_b_init_lat = float(ceil(b_size/self.input_len_per_cycle()))
        mat_b_init_lat += self.URAM_WR_LAT + self.DDR_RD_LAT
        # actual computation time
        input_cycles = self.num_grps_in_a(ideal=ideal) * self.num_wei_per_dpu(ideal=ideal) * self.a_h
        if mem_init:
            return input_cycles + self.dpu_lat() + mat_b_init_lat
        else:
            return input_cycles + self.dpu_lat()

    def compute_lat_teardown(self, ideal=False, mem_init=False):
        # matrix b init time
        b_size = self.b_w * self.b_h
        if ideal == False:
            mat_b_init_lat = b_size / self.input_len_per_cycle()
        else:
            mat_b_init_lat = float(ceil(b_size/self.input_len_per_cycle()))
        mat_b_init_lat += self.URAM_WR_LAT + self.DDR_RD_LAT
        # actual computation time
        input_cycles = self.num_grps_in_a(ideal=ideal) * self.num_wei_per_dpu(ideal=ideal) * self.a_h
        adder_tree_cycles = (self.input_len_per_cycle()-1)/30.0 * self.ADDER_LAT
        adder_lat = self.add_lat()

        if mem_init:
            return mat_b_init_lat + input_cycles, adder_tree_cycles, adder_lat
        else:
            return input_cycles, adder_tree_cycles, adder_lat

    def compute_resource(self):
        dpu_mults = self.num_mults_per_dpu * self.num_dpus
        dpu_adders, rest_elems = 0.0, self.num_mults_per_dpu
        while rest_elems > 0.0:
            dpu_adders += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))

        dpu_adders = (dpu_adders-1.0) * self.num_dpus
        # print(f"dpu mults: {dpu_mults}, dpu adders: {dpu_adders}")

        mem_usage = self.b_w * self.b_h * self.WORD_SIZE / 1024.
        return dpu_mults * self.MULT_RES, mem_usage

class NpuDpuModel(DpuModel):
    '''
    This class is used to construct NPU-based DPU hardware model which does AxB
    '''
    num_mults_per_dpu = 0
    num_dpus = 0
    a_w = 0
    a_h = 0
    b_w = 0
    b_h = 0

    # FIXME: check the availability here
    COMP_LAT = 3.0
    ADDER_LAT = 3.0
    MULT_LAT = 3.0
    MAC_LAT = 4.0

    ADDER_RES = 1/3.0
    MULT_RES = 1.0/30.0
    DIV_RES = 0
    COMP_RES = 0

    URAM_WR_LAT = 1.
    DDR_RD_LAT = 150.

    WORD_SIZE = 1

    def compute_lat(self, ideal=False):
        # matrix b init time
        b_size = self.b_w * self.b_h
        if ideal == False:
            mat_b_init_lat = b_size / self.input_len_per_cycle()
        else:
            mat_b_init_lat = float(ceil(b_size/self.input_len_per_cycle()))
        mat_b_init_lat += self.URAM_WR_LAT + self.DDR_RD_LAT
        # actual computation time
        input_cycles = self.num_grps_in_a(ideal=ideal) * self.num_wei_per_dpu(ideal=ideal) * self.a_h
        return input_cycles + self.dpu_lat() + mat_b_init_lat

    def compute_lat_teardown(self, ideal=False):
        # matrix b init time
        b_size = self.b_w * self.b_h
        if ideal == False:
            mat_b_init_lat = b_size / self.input_len_per_cycle()
        else:
            mat_b_init_lat = float(ceil(b_size/self.input_len_per_cycle()))
        mat_b_init_lat += self.URAM_WR_LAT + self.DDR_RD_LAT
        # actual computation time
        input_cycles = self.num_grps_in_a(ideal=ideal) * self.num_wei_per_dpu(ideal=ideal) * self.a_h
        adder_tree_cycles = (self.input_len_per_cycle()-1)/30.0 * self.ADDER_LAT
        adder_lat = self.add_lat()
        return mat_b_init_lat + input_cycles, adder_tree_cycles, adder_lat

    def compute_resource(self):
        dpu_mults = self.num_mults_per_dpu * self.num_dpus
        dpu_adders, rest_elems = 0.0, self.num_mults_per_dpu
        while rest_elems > 0.0:
            dpu_adders += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))

        dpu_adders = (dpu_adders-1.0) * self.num_dpus
        # print(f"dpu mults: {dpu_mults}, dpu adders: {dpu_adders}")

        mem_usage = self.b_w * self.b_h * self.WORD_SIZE / 1024.
        return dpu_mults * self.MULT_RES, mem_usage
        
class IdealMvmModel:
    tops = 0.0
    tmacs = 0.0
    freq = 600.0

    a_w = 0
    a_h = 0
    b_w = 0
    b_h = 0
    def __init__(self, a_h, a_w, b_h, b_w, perf_type: str, perf_val:float):
        if perf_type == f'{self.tops=}'.split('=')[0][5:]:
            self.tops = perf_val
        elif perf_type == f'{self.tmacs=}'.split('=')[0][5:]:
            self.tmacs = perf_val

        if self.tmacs == 0:
            self.tmacs = self.tops/2

        self.a_w = a_w
        self.a_h = a_h
        self.b_w = b_w
        self.b_h = b_h

        if self.b_h != self.a_w:
            raise MatMulDimErr("Error: mat mul dim mismatch")

    def compute_lat(self):
        tmac_ops = self.a_w * self.b_w * self.a_h / 1e12
        in_cycles = tmac_ops / self.tmacs / (1e-6 * 1.0/self.freq)
        return in_cycles


class BertModel:
    '''
    This class is used to construct bert hardware model
    '''
    exps = None
    matmul_v_model = None
    matmul_q_model = None
    matmul_k_model = None
    num_layers = 0.0
    num_heads = 0.0
    embd_size = 0.0
    max_seq_len = 320.

    freq = 500

    COMP_LAT = 3.0
    ADDER_LAT = 3.0
    MULT_LAT = 3.0
    MAC_LAT = 4.0
    DIV_LAT = 15

    ADDER_RES = 1.0/3.0
    MULT_RES = 1./30.
    DIV_RES = 0
    COMP_RES = 0

    WORD_SIZE = 2

    def __init__(self, embd_size=768.0, num_layers=0.0, num_heads=0.0, read_exp_samples=False, max_seq_len=320., exp_sample_path='params/scrs_sampled.npy', att_sample_path='params/attentions_sampled.npy'):
        if read_exp_samples:
            self.load_exp_out(exp_sample_path, att_sample_path)
            self.num_layers = self.exps[0].shape[0]
            self.num_heads = self.exps[0].shape[1]
        else:
            self.num_layers = num_layers
            self.num_heads = num_heads
            
        self.embd_size = embd_size
        self.max_seq_len = max_seq_len
        
        if self.num_layers == 0 or self.num_heads == 0:
            raise Exception("BertModel init error: lack of critical parameters")

    def analyzer_void_columns(self):
        if self.exps is None:
            return 0.0
        else:
            num_void_columns = []
            for inst in self.exps:
                for layer in inst:
                    for head in layer:
                        num_void_columns.append((np.sum(np.sum(head, axis=0) == 0.), head.shape[-1]))
        
            print(np.mean([num_void_column/num_column for num_void_column, num_column in num_void_columns]))
            return num_void_columns

    def load_exp_out(self, exp_path, atten_path):
        exps = []

        with open(atten_path, "rb") as attention_file:
            atten_len, _ = (np.load(attention_file))[0], []
        
        with open(exp_path, 'rb') as exps_file: 
            for i in range(atten_len): exps.append(np.load(exps_file))

        self.exps = exps

    def qkv_size(self):
        '''
        return size in MB        
        '''
        qkv_size = 0.0
        if self.exps is not None:
            qkv_size = np.mean([inst.shape[-2] for inst in self.exps]) * self.embd_size / self.num_heads
        else:
            qkv_size = self.max_seq_len * self.embd_size / self.num_heads

        qkv_size *= self.WORD_SIZE / pow(1024, 3)
        return qkv_size

    def att_size(self):
        att_size = 0.0
        if self.exps is not None:
            att_size = np.mean([pow(inst.shape[-1], 2) for inst in self.exps])
        else:
            att_size = pow(self.max_seq_len, 2)

        att_size *= self.WORD_SIZE / pow(1024, 3)
        return att_size

    def probe_exps(self):
        fig, ax = plt.subplots(1, 1, figsize=(24, 4))
        indices = ["{}".format(i+1) for i in range(144)]
        for i in range(144):
            if i%12 == 6: indices[i] = "layer {}".format(int(i/12) + 1)

        dat_sampled = random.sample(self.exps, 10)
        for dat in dat_sampled:
            print(dat.shape)
            num_zeros = np.count_nonzero((dat == 0.), axis=-1) / dat.shape[-1]
            
            num_zeros = num_zeros.reshape((num_zeros.shape[0]*num_zeros.shape[1], num_zeros.shape[2]))
            for i in range(num_zeros.shape[-1]):
                ax.plot(indices, num_zeros[:, i], 'o', color='black', alpha=0.06, markersize=2)

        ax.grid(linestyle='--', color='grey', alpha=0.4)
        ax.margins(0.002)

        ax.set_ylabel('#zeros in a row', fontsize=22)
        for l in range(12):
            ax.axvspan(l*12-0.5, l*12+12-0.5, alpha=0.2, facecolor='C{}'.format(l))
        ax.set_xticklabels(indices, Fontsize=22)
        for idx, tick in enumerate(ax.xaxis.get_major_ticks()):
            if idx % 12 !=6:
                tick.label1.set_visible(False)
            else: tick.label1.set_visible(True)
            
        for idx, tick in enumerate(ax.yaxis.get_major_ticks()):
            tick.label.set_fontsize(22)

        # ax.legend(handles=patches, loc='upper right', ncol=1, fontsize=22)

        fig.tight_layout()
        fig.savefig('res_fig/exps_count_zeros.png')
        plt.clf()

    def softmax_resources(self, p1, p2, l3, quant_bits):
        exp_resources = p1 * (2 ** quant_bits - 1) * self.COMP_RES
        exp_mem = p1 * (2 ** quant_bits * self.num_heads) / 2.0

        adder_tree_adders, rest_elems = 0.0, p1
        while rest_elems > 1.0:
            adder_tree_adders += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))

        adder_tree_adders = ceil(self.ADDER_RES * (p1-1))

        accu_mem = l3 * 2
        # calculate exp out buffer
        density = 0.3
        exp_out_buffer = (np.ceil(self.max_seq_len / p1) - 1) * l3 * p1 * density

        div_resources = p2 * self.DIV_RES

        row_parallelism = self.max_seq_len / l3
        total_mem = row_parallelism * (exp_mem + accu_mem + exp_out_buffer) * self.WORD_SIZE / 1024
        total_res = ceil(row_parallelism * (exp_resources + adder_tree_adders + 1 + div_resources))

        return total_res, total_mem

    def softmax_mem_teardown(self, p1, p2, l3, quant_bits):
        exp_mem = p1 * (2 ** quant_bits * self.num_heads) / 2.0

        accu_mem = l3 * 2
        # calculate exp out buffer
        density = 0.3
        exp_out_buffer = (np.ceil(self.max_seq_len / p1) - 1) * l3 * p1 * density

        row_parallelism = np.ceil(self.max_seq_len / l3)
        exp_mem = row_parallelism * exp_mem * self.WORD_SIZE / 1024
        accu_mem = row_parallelism * accu_mem * self.WORD_SIZE / 1024
        exp_out_buffer = row_parallelism * exp_out_buffer * self.WORD_SIZE / 1024

        return {'exp mem': exp_mem, "accu mem": accu_mem, "exp out buffer": exp_out_buffer}

    def qp_exp_lat(self):
        lut_decoder_lat = 0.0
        lut_lat = 2.0
        return self.COMP_LAT + lut_decoder_lat + lut_lat

    def softmax_lat(self, exp_dat=None, p1=1., p2=1., exp_h=-1):
        '''
        argument:
        exp_dat - output of attention exponent func  
        '''
        adder_tree_stages = p1-1
        if exp_dat is None:
            row_itlve_len = exp_h
            a_cols = 320 * 0.2
            grp_cols = ceil(exp_h / p1)
        else:
            row_itlve_len = exp_dat.shape[0]
            a_cols = np.count_nonzero(exp_dat, axis=-1)
            grp_cols = ceil(exp_dat.shape[-1] / p1)

        # padding zeros for unaligned parallel sub-cols
        a_cols_1 = np.ceil(a_cols / p1)
        a_cols_2 = np.ceil(a_cols / p2)
        lat = self.qp_exp_lat()
        lat += ((adder_tree_stages+1) * self.ADDER_LAT)
        
        #check parallelism eligibility
        p2_consuming = np.amax(a_cols_2) * row_itlve_len
        p1_producing = grp_cols * row_itlve_len

        if np.sum(a_cols) == 0:
            return 0.0

        if p1_producing < p2_consuming:
            return float("inf")

        if row_itlve_len <= self.ADDER_LAT: 
            lat += self.ADDER_LAT * (np.max(a_cols_1) - 1)
        else:
            rows_remained = row_itlve_len
            a_cols_1_copied = a_cols_1
            # as the computation goes, the remained number of rows decreases
            # as soon as the rest num of rows is smaller than adder latency,
            # no rows can be interleaved to hide the latency any more.
            while rows_remained > self.ADDER_LAT:
                lat += self.ADDER_LAT + rows_remained - self.ADDER_LAT
                a_cols_1_copied -= 1
                rows_remained=np.sum(a_cols_1_copied > 0)
            
            if rows_remained > 0:
                lat += self.ADDER_LAT * (np.max(a_cols_1_copied) - 1)

        lat += self.DIV_LAT + np.sum(a_cols_2)

        return lat

    def baseline_softmax_resource(self, p, l):
        exp_resource = self.MULT_RES
        log_resource = self.ADDER_RES

        tree_elems, rest_elems = 0.0, p
        while rest_elems > 1.0:
            tree_elems += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))
        
        res_all = (tree_elems + 1) * self.COMP_RES
        res_all += ceil(self.ADDER_RES * p)
        res_all += exp_resource * p
        res_all += (p-1) * self.ADDER_RES
        res_all += ceil(self.ADDER_RES * 2 * p)
        res_all += exp_resource * p
        res_all += log_resource

        row_parallelism = np.ceil(self.max_seq_len/l)
        res_all *= row_parallelism

        exp_lat = self.ADDER_LAT + self.MULT_LAT + 1 + 2
        stg_2_lat = self.ADDER_LAT + exp_lat + (p-1) * self.ADDER_LAT + self.ADDER_LAT

        exp_mem = 64
        log_mem = 64 + 32
        buffer_mem = p * stg_2_lat

        mem_all = row_parallelism * (exp_mem * p * 2 + log_mem + buffer_mem) * self.WORD_SIZE / 1024

        return res_all, mem_all


    def baseline_softmax_lat(self, exp_dat=None, pa=4., exp_h=-1, return_2ndstg_lat = False):
        exp_lat = self.ADDER_LAT + self.MULT_LAT + 1 + 2
        ln_lat = 2 + self.ADDER_LAT

        dat_h = exp_h if exp_dat is None else exp_dat.shape[0]
        dat_h = self.max_seq_len if dat_h < 0 else dat_h
        dat_w = self.max_seq_len if exp_dat is None else exp_dat.shape[1]
        
        stg_1_lat = log2Up(pa) * self.COMP_LAT
        # stg_1_lat += max(self.COMP_LAT, dat_h-1) * (ceil(dat_w / pa) - 1)

        stg_2_lat = self.ADDER_LAT + exp_lat + log2Up(pa) * self.ADDER_LAT + self.ADDER_LAT
        # stg_2_lat += max(self.ADDER_LAT, dat_h-1) * (ceil(dat_w / pa) - 1)

        stg_3_lat = ln_lat + self.ADDER_LAT + exp_lat
        stg_3_lat += dat_h * dat_w / pa

        pipeline_lat = stg_1_lat + stg_2_lat + stg_3_lat

        if return_2ndstg_lat:
            return stg_2_lat
        else:
            return pipeline_lat

    def matmul_lat_qkv_per_head(self, seq_len, blk=(64.0, 64.0), ideal=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
        
            in_cycles, adder_trees, adders = [], [], []
            for l in actual_seq_len:
                dpu_model = DpuModel(l, self.embd_size,  self.embd_size, self.embd_size/self.num_heads, blk[0], blk[1])
                in_cycle, adder_tree, adder = dpu_model.compute_lat_teardown(ideal=ideal)
                in_cycles.append(in_cycle)
                adder_trees.append(adder_tree)
                adders.append(adder)

            return np.mean(in_cycles), np.mean(adder_trees), np.mean(adders)
        else:
            dpu_model = DpuModel(seq_len, self.embd_size,  self.embd_size, self.embd_size/self.num_heads, blk[0], blk[1])
            return dpu_model.compute_lat_teardown(ideal=ideal)    


    def matmul_res_qkv_per_head(self, blk, seq_len=320):
        dpu_model = DpuModel(seq_len, self.embd_size,  self.embd_size, self.embd_size/self.num_heads, blk[0], blk[1])
        return dpu_model.compute_resource()

    def matmul_lat_qkv_per_head_ideal(self, seq_len, tops):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
        
            in_cycles = []
            for l in actual_seq_len:
                dpu_model = IdealMvmModel(l, self.embd_size, self.embd_size, self.embd_size/self.num_heads, "tops", tops)
                in_cycle = dpu_model.compute_lat()
                in_cycles.append(in_cycle)

            return np.mean(in_cycles)
        else:
            dpu_model = IdealMvmModel(seq_len, self.embd_size, self.embd_size, self.embd_size/self.num_heads, "tops", tops)
            return dpu_model.compute_lat()    

    def matmul_lat_qkv_per_head_stratix(self, seq_len, blk, ideal=False, mem_init=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
        
            in_cycles, adder_trees, adders = [], [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, self.embd_size,  self.embd_size, self.embd_size/self.num_heads, blk[0], blk[1])
                in_cycle, adder_tree, adder = dpu_model.compute_lat_teardown(ideal=ideal, mem_init=mem_init)
                in_cycles.append(in_cycle)
                adder_trees.append(adder_tree)
                adders.append(adder)

            return np.mean(in_cycles), np.mean(adder_trees), np.mean(adders)
        else:
            dpu_model = StratixDpuModel(seq_len, self.embd_size,  self.embd_size, self.embd_size/self.num_heads, blk[0], blk[1])
            return dpu_model.compute_lat_teardown(ideal=ideal, mem_init=mem_init)   
    
    def matmul_lat_qktrans_per_head(self, seq_len, blk=(64.0, 64.0), ideal=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, adder_trees, adders = [], [], []
            for l in actual_seq_len:
                dpu_model = DpuModel(l, self.embd_size,  self.embd_size, l, blk[0], blk[1])
                in_cycle, adder_tree, adder = dpu_model.compute_lat_teardown(ideal=ideal)
                in_cycles.append(in_cycle)
                adder_trees.append(adder_tree)
                adders.append(adder)

            return np.mean(in_cycles), np.mean(adder_trees), np.mean(adders)
        else:
            dpu_model = DpuModel(seq_len, self.embd_size,  self.embd_size, seq_len, blk[0], blk[1])
            return dpu_model.compute_lat_teardown(ideal=ideal)

    def matmul_res_qktrans_per_head(self, blk, seq_len=320):
        dpu_model = DpuModel(seq_len, self.embd_size,  self.embd_size, seq_len, blk[0], blk[1])
        return dpu_model.compute_resource()

    def matmul_res_qktrans_per_head_stratix(self, blk, seq_len=320):
        dpu_model = StratixDpuModel(seq_len, self.embd_size, self.embd_size, seq_len, blk[0], blk[1])
        return dpu_model.compute_resource()

    def matmul_lat_qktrans_per_head_ideal(self, seq_len, tops):
        if self.exps is not None:
            print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles = []
            for l in actual_seq_len:
                dpu_model = IdealMvmModel(l, self.embd_size, self.embd_size, l, "tops", tops)
                in_cycle = dpu_model.compute_lat()
                in_cycles.append(in_cycle)

            return np.mean(in_cycles)
        else:
            dpu_model = IdealMvmModel(seq_len, self.embd_size, self.embd_size, seq_len, "tops", tops)
            return dpu_model.compute_lat()

    def matmul_lat_qktrans_per_head_stratix(self, seq_len, blk=(64.0, 64.0), ideal=False, mem_init=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, adder_trees, adders = [], [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, self.embd_size,  self.embd_size, l, blk[0], blk[1])
                in_cycle, adder_tree, adder = dpu_model.compute_lat_teardown(ideal=ideal, mem_init=mem_init)
                in_cycles.append(in_cycle)
                adder_trees.append(adder_tree)
                adders.append(adder)

            return np.mean(in_cycles), np.mean(adder_trees), np.mean(adders)
        else:
            dpu_model = StratixDpuModel(seq_len, self.embd_size,  self.embd_size, seq_len, blk[0], blk[1])
            return dpu_model.compute_lat_teardown(ideal=ideal, mem_init=mem_init)

    def matmul_lat_vatt_per_head_stratix(self, seq_len, blk=(64, 64), ideal=False, mem_init=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, adder_trees, adders = [], [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, l, l, (self.embd_size / self.num_heads), blk[0], blk[1])
                in_cycle, adder_tree, adder = dpu_model.compute_lat_teardown(ideal=ideal, mem_init=mem_init)
                in_cycles.append(in_cycle)
                adder_trees.append(adder_tree)
                adders.append(adder)

            return np.mean(in_cycles), np.mean(adder_trees), np.mean(adders)
        else:
            dpu_model = StratixDpuModel(seq_len, seq_len, seq_len, (self.embd_size / self.num_heads), blk[0], blk[1])
            return dpu_model.compute_lat_teardown(ideal=ideal, mem_init=mem_init)

    def matmul_lat_selfatt_out_stratix(self, seq_len, blk=(64, 64), ideal=False, mem_init=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, adder_trees, adders = [], [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, self.embd_size, self.embd_size, self.embd_size, blk[0], blk[1])
                in_cycle, adder_tree, adder = dpu_model.compute_lat_teardown(ideal=ideal, mem_init=mem_init)
                in_cycles.append(in_cycle)
                adder_trees.append(adder_tree)
                adders.append(adder)

            return np.mean(in_cycles), np.mean(adder_trees), np.mean(adders)
        else:
            dpu_model = StratixDpuModel(seq_len, self.embd_size, self.embd_size, self.embd_size, blk[0], blk[1])
            return dpu_model.compute_lat_teardown(ideal=ideal, mem_init=mem_init)

    def att_v_outer_product_intermediate_size(self):
        if self.exps is not None:
            per_inst_intermediate_size = []
            for inst in self.exps:
                inst_flatten = inst.reshape(-1, inst.shape[-2], inst.shape[-1])
                col_density = np.sum((inst_flatten <= 0), axis=1)
                intermediate_res_size = col_density * self.embd_size
                total_size = np.sum(intermediate_res_size, axis=-1)
                per_inst_intermediate_size += total_size.tolist()

            return np.array(per_inst_intermediate_size)
        else:
            return None

    def check_softmax_memory(self, mvm_tcore: float, softmax_tcore: tuple, softmax_type = "baseline"):
        # starting to compare the latency of softmax and q,k,v,qxkT 
        # by the time the first attention score is generated.
        equi_mvm_blk_size = int(sqrt(mvm_tcore*30))
        mvm_in_cycles, mvm_accumu_lat, mvm_adder_lat = \
            self.matmul_lat_qkv_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True, mem_init=False)

        qktrans_in_cycles, qktrans_accumu_lat, qktrans_adder_lat = \
            self.matmul_lat_qktrans_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True, mem_init=False)

        # start of new score outputs:
        new_score_output_lat = mvm_in_cycles*4 + qktrans_accumu_lat + qktrans_adder_lat
        # start of softmax stg3:
        r, p = softmax_tcore
        start_of_softmax_stg3_lat = qktrans_accumu_lat + qktrans_adder_lat + \
                                        self.baseline_softmax_lat(pa=p, exp_h=ceil(self.max_seq_len/r), return_2ndstg_lat=True)
        if new_score_output_lat < start_of_softmax_stg3_lat:
            logging.warning("softmax mem in danger of overflow!")
            return False

        return True
        

    def attention_lat_stratix(self, mvm_tcore: float, softmax_tcore: tuple, softmax_type = "baseline", consider_mem_init=False):
        '''
        softmax_tcore: r, p -> r: row parallelism, p -> column parallelism
        '''
        equi_mvm_blk_size = int(sqrt(mvm_tcore*30))
        mvm_in_cycles, mvm_accumu_lat, mvm_adder_lat = \
            self.matmul_lat_qkv_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True, mem_init=consider_mem_init)
        
        single_head_iter = mvm_in_cycles * 2
        single_head_iter += mvm_accumu_lat + mvm_adder_lat

        # first 11 heads to cover the softmax latency by mvm compute:
        qktrans_in_cycles, qktrans_accumu_lat, qktrans_adder_lat = \
            self.matmul_lat_qktrans_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True, mem_init=consider_mem_init)
        qkv_qktrans_compute_lat = qktrans_in_cycles + mvm_in_cycles * 3 + qktrans_accumu_lat + qktrans_adder_lat
        r, p = softmax_tcore
        if self.exps is None:
            softmax_stg1_incycle = ceil(self.max_seq_len/r) * ceil(self.max_seq_len/p)
        else:
            softmax_stg1_incycle = np.mean([ceil(h.shape[-1]/r) * ceil(float(h.shape[-1])/p) \
                                            for h in self.exps])

        flatten_exps = []
        if self.exps is not None:
            for inst in self.exps:
                num_layers, num_heads, num_rows, _ = inst.shape
                real_exp = inst.reshape((num_layers * num_heads, num_rows, num_rows))
                for h in real_exp: flatten_exps.append(h)

        if softmax_type == "baseline":
            temp_lats = []
            if len(flatten_exps) > 0:
                for inst in flatten_exps:
                    effective_inst = inst[:ceil(inst.shape[-1]/r), :]
                    temp_lats.append(self.baseline_softmax_lat(effective_inst, p))
            else:
                temp_lats.append(self.baseline_softmax_lat(pa=p, exp_h=ceil(self.max_seq_len/r)))

            softmax_lat = softmax_stg1_incycle + np.mean(np.array(temp_lats))
        else:
            temp_lats = []
            if len(flatten_exps) > 0:
                for inst in flatten_exps:
                    effective_inst = inst[:ceil(inst.shape[-1]/r), :]
                    temp_lats.append(self.softmax_lat(exp_dat=effective_inst, p1=p, p2=p))
            else:
                temp_lats.append(self.softmax_lat(p1=p, p2=p, exp_h=ceil(self.max_seq_len/r)))
                
            softmax_lat = softmax_stg1_incycle + np.mean(np.array(temp_lats))

        # select dominate intermediate latency: softmax in cycles or q k v compute
        intermediate_lat = max(qkv_qktrans_compute_lat, softmax_stg1_incycle)
        if qkv_qktrans_compute_lat > softmax_stg1_incycle:
            logging.info(f"{__name__}: softmax hidden succeeded")
        else:
            logging.info(f"{__name__}: softmax hidden failed")

        predessesor_heads_lat = mvm_in_cycles * 2 + intermediate_lat * (self.num_heads-1)
        # softmax finish time (absolute time)
        softmax_first_finish_time = mvm_in_cycles * 2 + qktrans_adder_lat + qktrans_accumu_lat + \
                                        softmax_stg1_incycle + softmax_lat
        softmax_first_ddl = predessesor_heads_lat + intermediate_lat
        if softmax_first_ddl < softmax_first_finish_time:
            logging.info(softmax_first_finish_time, softmax_first_ddl)

        # accumulate VxAtt
        vatt_mult_incycles, _, _ = \
            self.matmul_lat_vatt_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True, mem_init=consider_mem_init)
        vatt_mult_incycles *= self.num_heads

        # last step: FC layer for output
        output_fc_incycles, output_fc_accu, output_fc_adder = \
            self.matmul_lat_selfatt_out_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True, mem_init=consider_mem_init)
        output_fc_lat = output_fc_incycles + output_fc_accu + output_fc_adder

        if softmax_first_ddl < softmax_first_finish_time:
            # softmax latency cannot be covered by 12 heads
            first_vatt_finish = softmax_lat + vatt_mult_incycles
            second_vatt_start = intermediate_lat + softmax_lat
            vatt_gap = second_vatt_start - first_vatt_finish
            vatt_gap = 0 if vatt_gap < 0 else vatt_gap
            res = mvm_in_cycles * 2 + qktrans_accumu_lat + qktrans_adder_lat
            res += softmax_lat + (vatt_mult_incycles + vatt_gap) * self.num_heads
            res += output_fc_lat
        else: 
            # softmax latency can be covered by 12 heads
            res = intermediate_lat + predessesor_heads_lat + vatt_mult_incycles + output_fc_lat
        
        return res

    def attention_mvm_only_lat_stratix(self, mvm_tcore: float):
        '''
        compute mvm only latency of the attention, used to estimate dynamic utilization of the mvm unit.
        '''
        equi_mvm_blk_size = int(sqrt(mvm_tcore*30))
        mvm_in_cycles, mvm_accumu_lat, mvm_adder_lat = \
            self.matmul_lat_qkv_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True)

        # first 11 heads to cover the softmax latency by mvm compute:
        qktrans_in_cycles, qktrans_accumu_lat, qktrans_adder_lat = \
            self.matmul_lat_qktrans_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True)
        qkv_qktrans_compute_lat = (mvm_in_cycles * 3 + qktrans_in_cycles) * self.num_heads
        
        vatt_mult_incycles, _, _ = \
            self.matmul_lat_vatt_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True)
        vatt_mult_incycles *= self.num_heads

        # last step: FC layer for output
        output_fc_incycles, output_fc_accu, output_fc_adder = \
            self.matmul_lat_selfatt_out_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True)
        output_fc_lat = output_fc_incycles + output_fc_accu + output_fc_adder

        total_lat = qkv_qktrans_compute_lat + vatt_mult_incycles + output_fc_lat

        return total_lat

    def attention_bandwidth_stratix(self, mvm_tcore: float, softmax_tcore: tuple, softmax_type = "baseline"):
        equi_mvm_blk_size = int(sqrt(mvm_tcore*30))
        mvm_in_cycles, mvm_accumu_lat, mvm_adder_lat = \
            self.matmul_lat_qkv_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True)

        # first 11 heads to cover the softmax latency by mvm compute:
        qktrans_in_cycles, qktrans_accumu_lat, qktrans_adder_lat = \
            self.matmul_lat_qktrans_per_head_stratix(self.max_seq_len, (equi_mvm_blk_size, equi_mvm_blk_size), ideal=True)
        qkv_qktrans_compute_lat = qktrans_in_cycles + mvm_in_cycles * 3 + qktrans_accumu_lat + qktrans_adder_lat
        r, p = softmax_tcore
        softmax_stg1_incycle = np.mean([ceil(h.shape[-1]/r) * ceil(float(h.shape[-1])/p) \
                                            for h in self.exps])

        flatten_exps = []
        for inst in self.exps:
            num_layers, num_heads, num_rows, _ = inst.shape
            real_exp = inst.reshape((num_layers * num_heads, num_rows, num_rows))
            for h in real_exp: flatten_exps.append(h)

        if softmax_type == "baseline":
            temp_lats = []
            for inst in flatten_exps:
                effective_inst = inst[:ceil(inst.shape[-1]/r), :]
                temp_lats.append(self.baseline_softmax_lat(effective_inst, p))

            softmax_lat = softmax_stg1_incycle + np.mean(np.array(temp_lats))
        else:
            temp_lats = []
            for inst in flatten_exps:
                effective_inst = inst[:ceil(inst.shape[-1]/r), :]
                temp_lats.append(self.softmax_lat(effective_inst, p, p))

            softmax_lat = softmax_stg1_incycle + np.mean(np.array(temp_lats))

        # select dominate intermediate latency: softmax in cycles or q k v compute
        intermediate_lat = max(qkv_qktrans_compute_lat, softmax_stg1_incycle)

        if qkv_qktrans_compute_lat > softmax_stg1_incycle:
            logging.info(f"{__name__}: softmax hidden succeeded")
        else:
            logging.info(f"{__name__}: softmax hidden failed")

        predessesor_heads_lat = mvm_in_cycles * 2 + intermediate_lat * (self.num_heads-1)
        # softmax finish time (absolute time)
        softmax_first_finish_time = mvm_in_cycles * 2 + qktrans_adder_lat + qktrans_accumu_lat + \
                                        softmax_stg1_incycle + softmax_lat
        softmax_first_ddl = predessesor_heads_lat + intermediate_lat
        if softmax_first_ddl < softmax_first_finish_time:
            print(softmax_first_finish_time, softmax_first_ddl)

        dat_moving_time = max(softmax_lat, intermediate_lat * self.num_heads)
        dat_moving_time *= 1./self.freq * 1e-6
        v_size = self.qkv_size()
        att_size = self.att_size()
        dat_bw = (v_size + att_size) / dat_moving_time
        
        return dat_bw
    

if __name__ == '__main__':
    # bert_hw_model = BertModel(read_exp_samples=False)
    # bert_hw_model.probe_exps()
    static_bert_models = BertModel(read_exp_samples=False, num_layers=12, num_heads=12, max_seq_len=512)
    stratix_head_lat = static_bert_models.matmul_lat_qkv_per_head_stratix(512, blk=(128, 128), ideal=True)
    npu_head_lat = static_bert_models.matmul_lat_qkv_per_head_npu(512, blk=(128, 128), ideal=True)
    print(stratix_head_lat, npu_head_lat)
