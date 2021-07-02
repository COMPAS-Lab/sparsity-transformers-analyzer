from math import ceil, floor, exp, log2, gcd
import numpy as np
import matplotlib.pyplot as plt
import random

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

class StratixDpuModel:
    '''
    This class is used to construct Intel Stratix DPU hardware model which does AxB
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

    COMP_LAT = 2.0
    ADDER_LAT = 3.0
    MULT_LAT = 3.0
    DIV_LAT = 15

    ADDER_RES = 1.0/3.0
    MULT_RES = 2
    DIV_RES = 0
    COMP_RES = 0

    WORD_SIZE = 2

    def __init__(self, embd_size=768.0, num_layers=0.0, num_heads=0.0, read_exp_samples=False, exp_sample_path='params/scrs_sampled.npy', att_sample_path='params/attentions_sampled.npy'):
        if read_exp_samples:
            self.load_exp_out(exp_sample_path, att_sample_path)
            self.num_layers = self.exps[0].shape[0]
            self.num_heads = self.exps[0].shape[1]
        else:
            self.num_layers = num_layers
            self.num_heads = num_heads
            
        self.embd_size = embd_size
        
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

        adder_tree_adders *= ceil(self.ADDER_RES)

        accu_mem = l3 * 2
        # calculate exp out buffer
        density = 0.3
        exp_out_buffer = (np.ceil(self.max_seq_len / p1) - 1) * l3 * p1 * density

        div_resources = p2 * self.DIV_RES

        row_parallelism = np.ceil(self.max_seq_len / l3)
        total_mem = row_parallelism * (exp_mem + accu_mem + exp_out_buffer) * self.WORD_SIZE / 1024
        total_res = row_parallelism * (exp_resources + adder_tree_adders + 1 + div_resources)

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

    def softmax_lat(self, exp_dat, p1=1., p2=1.):
        '''
        argument:
        exp_dat - output of attention exponent func  
        '''
        adder_tree_stages = log2(p1)
        row_itlve_len = exp_dat.shape[0]

        a_cols = np.count_nonzero(exp_dat, axis=-1)
        # padding zeros for unaligned parallel sub-cols
        a_cols_1 = np.ceil(a_cols / p1)
        a_cols_2 = np.ceil(a_cols / p2)
        lat = self.qp_exp_lat()
        lat += ((adder_tree_stages+1) * self.ADDER_LAT)
        
        #check parallelism eligibility
        p2_consuming = np.amax(a_cols_2) * row_itlve_len
        p1_producing = np.ceil(exp_dat.shape[-1] / p1) * row_itlve_len

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
        exp_resource = max(self.MULT_RES, self.ADDER_RES)
        log_resource = self.ADDER_RES

        tree_elems, rest_elems = 0.0, p
        while rest_elems > 1.0:
            tree_elems += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))
        
        res_all = (tree_elems + 1) * self.COMP_RES
        res_all += ceil(self.ADDER_RES * p)
        res_all += exp_resource * p
        res_all += (tree_elems + 1) * self.ADDER_RES
        res_all += ceil(self.ADDER_RES * 2 * p)
        res_all += exp_resource * p
        res_all += log_resource

        row_parallelism = np.ceil(self.max_seq_len/l)
        res_all *= row_parallelism

        exp_lat = self.ADDER_LAT + self.MULT_LAT + 1 + 2
        stg_2_lat = self.ADDER_LAT + exp_lat + log2Up(p) * self.ADDER_LAT + self.ADDER_LAT

        exp_mem = 64
        log_mem = 64 + 32
        buffer_mem = p * stg_2_lat

        mem_all = row_parallelism * (exp_mem * p * 2 + log_mem + buffer_mem) * self.WORD_SIZE / 1024

        return res_all, mem_all


    def baseline_softmax_lat(self, exp_dat=None, pa=4., exp_h=-1):
        exp_lat = self.ADDER_LAT + self.MULT_LAT + 1 + 2
        ln_lat = 2 + self.ADDER_LAT

        dat_h = exp_h if exp_dat is None else exp_dat.shape[0]
        dat_h = self.max_seq_len if dat_h < 0 else dat_h
        dat_w = self.max_seq_len if exp_dat is None else exp_dat.shape[1]
        
        stg_1_lat = log2Up(pa) * self.COMP_LAT
        stg_1_lat += max(self.COMP_LAT, dat_h-1) * (ceil(dat_w / pa) - 1)

        stg_2_lat = self.ADDER_LAT + exp_lat + log2Up(pa) * self.ADDER_LAT + self.ADDER_LAT
        stg_2_lat += max(self.ADDER_LAT, dat_h-1) * (ceil(dat_w / pa) - 1)

        stg_3_lat = ln_lat + self.ADDER_LAT + exp_lat
        stg_3_lat += dat_h * dat_w

        pipeline_lat = stg_1_lat + stg_2_lat + stg_3_lat

        return pipeline_lat

    def matmul_lat_qkv_per_head(self, seq_len, blk=(64.0, 64.0), ideal=False):
        if self.exps is not None:
            print(__name__+": using acutal size of exp")
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
            print(__name__+": using acutal size of exp")
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

    def matmul_lat_qkv_per_head_stratix(self, seq_len, blk, ideal=False):

        if self.exps is not None:
            print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
        
            in_cycles, adder_trees, adders = [], [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, self.embd_size,  self.embd_size, self.embd_size/self.num_heads, blk[0], blk[1])
                in_cycle, adder_tree, adder = dpu_model.compute_lat_teardown(ideal=ideal)
                in_cycles.append(in_cycle)
                adder_trees.append(adder_tree)
                adders.append(adder)

            return np.mean(in_cycles), np.mean(adder_trees), np.mean(adders)
        else:
            dpu_model = StratixDpuModel(seq_len, self.embd_size,  self.embd_size, self.embd_size/self.num_heads, blk[0], blk[1])
            return dpu_model.compute_lat_teardown(ideal=ideal)   
        pass

    
    def matmul_lat_qktrans_per_head(self, seq_len, blk=(64.0, 64.0), ideal=False):
        if self.exps is not None:
            print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, adder_trees, adders = [], [], []
            for l in actual_seq_len:
                dpu_model = DpuModel(l, self.embd_size,  self.embd_size, seq_len, blk[0], blk[1])
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

    def matmul_lat_qktrans_per_head_stratix(self, seq_len, blk=(64.0, 64.0), ideal=False):
        if self.exps is not None:
            print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, adder_trees, adders = [], [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, self.embd_size,  self.embd_size, seq_len, blk[0], blk[1])
                in_cycle, adder_tree, adder = dpu_model.compute_lat_teardown(ideal=ideal)
                in_cycles.append(in_cycle)
                adder_trees.append(adder_tree)
                adders.append(adder)

            return np.mean(in_cycles), np.mean(adder_trees), np.mean(adders)
        else:
            dpu_model = StratixDpuModel(seq_len, self.embd_size,  self.embd_size, seq_len, blk[0], blk[1])
            return dpu_model.compute_lat_teardown(ideal=ideal)

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
    

if __name__ == '__main__':
    bert_hw_model = BertModel(read_exp_samples=True)
    # bert_hw_model.probe_exps()

    temp_lat = bert_hw_model.softmax_lat(bert_hw_model.exps[0][0, 0, :20, :20], p1=8, p2=16)
    print(temp_lat)