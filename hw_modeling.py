from math import ceil, floor, exp, log2, gcd, sqrt, pow
from re import template
import numpy as np
import matplotlib.pyplot as plt
import random
import sys, logging
from numpy.core.fromnumeric import nonzero, size
import sympy as sp

log = logging.getLogger(__name__)

class OutOfResourceError(Exception):
    pass

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

    def total_ops(self, exp_dat=None):
        if exp_dat is None:
            return  self.a_h * self.a_w * 2 * self.b_w
        else:
            # count none zeros per row
            none_zeros = np.count_nonzero(exp_dat, axis=-1)
            # mimicing the padding zeros to every 3 rows
            zeros_padded = none_zeros.size % 3
            none_zeros = np.pad(none_zeros, (0, 3 - zeros_padded), "constant", constant_values=0)
            none_zeros = np.split(none_zeros, np.arange(3, none_zeros.size, 3))
            max_none_zeros_per_grp = np.array([np.amax(i) for i in none_zeros])
            print("max none zeros per 3 rows: ", max_none_zeros_per_grp)
            total_ops = sum([dense_vals * 2 * self.b_w * 3 for dense_vals in max_none_zeros_per_grp])
            return total_ops

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

    ADDER_RES = 1/3.0
    MULT_RES = 1.0/30.0
    DIV_RES = 0
    COMP_RES = 0

    URAM_WR_LAT = 1.
    DDR_RD_LAT = 150.

    WORD_SIZE = 1

    FREQ = 500.0
    NUM_TCs = 3960.0
    __exp_dat = None

    def __init__(self, a_h, a_w, b_h, b_w, exp_dat=None, freq=0.0, num_tcs=0.0):
        super().__init__(a_h, a_w, b_h, b_w, 16, 16)
        if exp_dat is not None:
            self.__exp_dat = exp_dat
        if freq > 0:
            self.FREQ = freq
        if num_tcs > 0:
            self.NUM_TCs = num_tcs
        
    # derived parameters
    def compute_lat(self, cascade_len: int, ideal=False):
        '''
        compute latency based on fpga 21 paper
        '''
        chain_loading_lat = 3 * (cascade_len + 1)

        if self.__exp_dat is None:
            round = lambda x: x if ideal else ceil(x)
            matA_size = (self.a_h, self.a_w)
            matB_size = (self.b_h, self.b_w)

            total_ops =  self.total_ops()
            block_matA_size = (3, matA_size[1])
            block_matB_size = (matA_size[1], chain_loading_lat)
            a_loading_grps = round(matA_size[1] / (cascade_len * 10))

            compute_block_ops = block_matA_size[0] * block_matA_size[1] * 2 * block_matB_size[1]
            
            total_latency = chain_loading_lat + chain_loading_lat * a_loading_grps + \
                                4 + cascade_len * 2

            num_cores = round(self.NUM_TCs / (cascade_len + 2))
            num_blocks = round(total_ops / compute_block_ops)
            pipeline_iters = round(num_blocks / num_cores)
            total_latency *= pipeline_iters

            return total_latency
        else:
            round = lambda x: x if ideal else np.ceil(x)

            if len(self.__exp_dat.shape) != 2:
                raise Exception("Wrong exp shape in compute latency!")
            else:
                total_ops = self.total_ops(self.__exp_dat)

                # count none zeros per row
                none_zeros = np.count_nonzero(self.__exp_dat, axis=-1)
                # mimicing the padding zeros to every 3 rows
                zeros_padded = none_zeros.size % 3
                none_zeros = np.pad(none_zeros, (0, 3 - zeros_padded), "constant", constant_values=0)
                none_zeros = np.split(none_zeros, np.arange(3, none_zeros.size, 3))
                max_none_zeros_per_grp = np.array([np.amax(i) for i in none_zeros])

                a_loading_grps = round(max_none_zeros_per_grp / (cascade_len * 10))
                grp_loading_lats = chain_loading_lat + chain_loading_lat * a_loading_grps + \
                                4 + cascade_len * 2
                grp_loading_lats *= round(self.b_w / chain_loading_lat)
                num_cores = round(self.NUM_TCs / (cascade_len + 2))

                ideal_lat = round(np.sum(grp_loading_lats) / num_cores)
                greedy_lat = ideal_lat * ((4*num_cores-1)/(3*num_cores))
                return greedy_lat


    def compute_lat_teardown(self, cascade_len: int, ideal=False):
        '''
        compute latency based on fpga 21 paper
        '''
        round = lambda x, f: x if ideal else f(x)

        matA_size = (self.a_h, self.a_w)
        matB_size = (self.b_h, self.b_w)

        total_ops =  self.total_ops()
        chain_loading_lat = 3 * (cascade_len + 1)
        block_matA_size = (3, matA_size[1])
        block_matB_size = (matA_size[1], chain_loading_lat)
        a_loading_grps = round(matA_size[1] / (cascade_len * 10), ceil)

        compute_block_ops = block_matA_size[0] * block_matA_size[1] * 2 * block_matB_size[1]
        
        in_cycles = chain_loading_lat + chain_loading_lat * a_loading_grps
        lat = 4 + cascade_len * 2

        num_cores = round(self.NUM_TCs / (cascade_len + 2), floor)
        num_blocks = round(total_ops / compute_block_ops, ceil)
        pipeline_iters = round(num_blocks / num_cores, ceil)

        return in_cycles * pipeline_iters, lat * pipeline_iters


    def compute_resource(self, tcs_cascade_len: int, num_cores: int):
        '''
        compute resources by assuming the tcs cascade length does not include head/tail tcs.
        '''
        tcs = (tcs_cascade_len+2) * num_cores
        mem_usage = self.fpga21_output_ram_size(tcs_cascade_len)
        return tcs, mem_usage

    def tensor_mat_flops(self, in_cascade_len, acc_cascade_len, num_cols_grp_loading_chain=2):
        '''
        compute single block matrix size, and the flops of it on stratix nx chain.

        mat a: each block in the init chain has 3 rows from A, and n_cols * 10 elems for cols from A
        then we have acc_cascade_len of such cols from A
        mat b: infer rows from B by cols from A, and each 3 cols. If increasing the tile parallelism, 
        the cols will be 3 * num_tiles
        '''
        matA_size = (in_cascade_len * 3, 10 * acc_cascade_len * num_cols_grp_loading_chain)
        matB_size = (10 * acc_cascade_len * num_cols_grp_loading_chain, 3)
        input_ops = in_cascade_len * 3 * (10 * acc_cascade_len * num_cols_grp_loading_chain) * 2 * 3

        total_latency = 6 * (in_cascade_len - 1) + num_cols_grp_loading_chain * 3 + 4 + 3

        total_latency =  total_latency * 1./self.FREQ * 1e-6
        flops = input_ops / total_latency / 1e12
        print("mat size: ", matA_size, matB_size)
        print(total_latency)

        return flops

    def tensor_fpt20_mat_flops(self, initcas_len, acccas_len, num_initcas_lane, sym=False):
        '''
        compute flops for fpt20 paper, with aggregated initialization.
        '''
        if not sym:
            matA_size = (self.a_h, self.a_w)
            matB_size = (self.b_h, self.b_w)

            flops = matA_size[0] * matA_size[1] * 2 * matB_size[1]
            compute_block_flops = initcas_len * (3 * initcas_len) * matA_size[1] * 2 * (3 * num_initcas_lane)
            a_cols_grps = ceil(matA_size[1] / (10 * acccas_len))
            total_latency = 3 * initcas_len + (3*initcas_len) * a_cols_grps + 4 + acccas_len * 2

            num_cores = floor(self.NUM_TCs / ((initcas_len + 1) * (acccas_len + 1) * num_initcas_lane))
            if num_cores < 1:
                raise OutOfResourceError("Error: not enough resources on the chip")
            num_blocks = ceil(flops / compute_block_flops)
            pipeline_iters = ceil(num_blocks / num_cores)
            total_latency *= pipeline_iters * 1./self.FREQ * 1e-6

            # print("npu cores: ", num_cores, "npu core iterations: ", pipeline_iters)

            flops = flops / total_latency / 1e12
        else:
            # simplify the equation and export latex code
            mA_row, mA_col, mB_col = sp.symbols('arow acol bcol')
            mB_row = mA_col
            init_len, acc_len, init_grp = sp.symbols('init\_len acc\_len init\_grp')

            flops = mA_row * mA_col * 2 * mB_col
            compute_block_flops = init_len * (3 * init_len) * mA_col * 2 * (3 * init_grp)
            a_cols_grps = (mA_col / (10 * acc_len))
            total_latency = 3 * init_len + (3*init_len) * a_cols_grps + 4 + acc_len * 2

            num_cores = (self.NUM_TCs / ((init_len + 1) * (acc_len + 1) * init_grp))
            num_blocks = (flops / compute_block_flops)
            pipeline_iters = (num_blocks / num_cores)
            total_latency *= pipeline_iters * 1./self.FREQ * 1e-6
            flops = flops / total_latency / 1e12
            flops = flops.subs({init_len: initcas_len, acc_len: acccas_len, init_grp: num_initcas_lane})
            print("lim: ", sp.latex(sp.simplify(sp.limit(flops, mA_col, sp.oo))))
            flops = sp.latex(flops)
            
        return flops

    def tensor_fpt20fixed_mat_flops(self, lanes, cores, tiles, dpes, sym=False):
        '''
        compute flops for fpt20 paper, with aggregated initialization.
        FIXME: this function returns significant different result compared to the 
        original paper.
        '''
        if not sym:
            matA_size = (self.a_h, self.a_w)
            matB_size = (self.b_h, self.b_w)

            ops = self.total_ops()
            init_latency = lanes / 10 * 3
            total_lat = init_latency + max(matA_size[0] / cores, init_latency) * \
                            ceil(matB_size[0] / (tiles * lanes / 10)) * ceil(matB_size[1] / (3*dpes)) + \
                            3 + tiles * lanes / 10
            # total_lat = init_latency + max(matB_size[1] / cores, init_latency) * \
            #                 (matA_size[1] / (tiles * lanes / 10)) * (matA_size[0] / (3*dpes)) + \
            #                   3 + tiles * lanes / 10 
            total_lat *= 1./self.FREQ * 1e-6

            required_res = cores * tiles * dpes * lanes / 10
            if required_res > self.NUM_TCs:
                raise OutOfResourceError("Error: not enough resources on the chip")

            print(f"init lat: {init_latency}, input cycles: {matA_size[0] / cores}")
            # print("npu cores: ", num_cores, "npu core iterations: ", pipeline_iters)
            flops = ops / total_lat / 1e12
        else:
            flops = 'a+b'
            # # simplify the equation and export latex code
            # mA_row, mA_col, mB_col = sp.symbols('arow acol bcol')
            # mB_row = mA_col
            # init_len, acc_len, init_grp = sp.symbols('init\_len acc\_len init\_grp')

            # flops = mA_row * mA_col * 2 * mB_col
            # compute_block_flops = init_len * (3 * init_len) * mA_col * 2 * (3 * init_grp)
            # a_cols_grps = (mA_col / (10 * acc_len))
            # total_latency = 3 * init_len + (3*init_len) * a_cols_grps + 4 + acc_len * 2

            # num_cores = (self.NUM_TCs / ((init_len + 1) * (acc_len + 1) * init_grp))
            # num_blocks = (flops / compute_block_flops)
            # pipeline_iters = (num_blocks / num_cores)
            # total_latency *= pipeline_iters * 1./self.FREQ * 1e-6
            # flops = flops / total_latency / 1e12
            # flops = flops.subs({init_len: initcas_len, acc_len: acccas_len, init_grp: num_initcas_lane})
            # print("lim: ", sp.latex(sp.simplify(sp.limit(flops, mA_col, sp.oo))))
            # flops = sp.latex(flops)
            
        return flops

    def tensor_fpga21_mat_flops(self, cascade_len, sym=False, ideal: bool=False):
        ''' 
        compute flops with a given number of cascaded chain and b cols
        a loading grps: the number of groups that a chain is responsible for along the a rows.
        assuming a chain must finish 3 entire A rows at least.
        '''

        round = lambda x: x if ideal else ceil(x)

        if not sym:
            matA_size = (self.a_h, self.a_w)
            matB_size = (self.b_h, self.b_w)

            total_ops =  self.total_ops()
            chain_loading_lat = 3 * (cascade_len + 1)
            block_matA_size = (3, matA_size[1])
            block_matB_size = (matA_size[1], chain_loading_lat)
            a_loading_grps = round(matA_size[1] / (cascade_len * 10))

            compute_block_ops = block_matA_size[0] * block_matA_size[1] * 2 * block_matB_size[1]
            
            total_latency = chain_loading_lat + chain_loading_lat * a_loading_grps + \
                                4 + cascade_len * 2

            num_cores = round(self.NUM_TCs / (cascade_len + 2))
            num_blocks = round(total_ops / compute_block_ops)
            pipeline_iters = round(num_blocks / num_cores)
            total_latency *= pipeline_iters * 1./self.FREQ * 1e-6

            # print("num_cores for fpga 21: ", num_cores)

            flops = total_ops / total_latency / 1e12
        else:
            # simplify the equation and export latex code
            mA_row, mA_col, mB_col = sp.symbols('arow acol bcol')
            mB_row = mA_col
            cas_len = sp.symbols('len')

            total_ops =  mA_row * mA_col * 2 * mB_col
            chain_loading_lat = 3 * (cas_len + 1)
            block_matA_size = (3, mA_col)
            block_matB_size = (mA_col, chain_loading_lat)
            a_loading_grps = (mA_col / (cas_len * 10))

            compute_block_ops = block_matA_size[0] * block_matA_size[1] * 2 * block_matB_size[1]
            
            total_latency = chain_loading_lat + chain_loading_lat * a_loading_grps + \
                                4 + cas_len * 2

            num_cores = (self.NUM_TCs / (cas_len + 2))
            num_blocks = (total_ops / compute_block_ops)
            pipeline_iters = (num_blocks / num_cores)
            total_latency *= pipeline_iters * 1./self.FREQ * 1e-6

            flops = total_ops / total_latency / 1e12
            print("total_lat: ", sp.latex(sp.simplify(total_latency)))
            flops = flops.subs({cas_len: cascade_len})
            print("lim: ", sp.latex(sp.simplify(sp.limit(flops, mA_col, sp.oo))))

            flops = sp.latex(flops)

        return flops

    def ideal_tops(self):
        ops = (10*2*3) * self.NUM_TCs
        latency = 1/self.FREQ * 1e-6
        tops = ops / latency / 1e12
        return tops

    def fpga21_output_ram_size(self, cascade_len):
        '''
        compute ram requirments for the fifo after each chain
        '''
        num_elements = 3 * cascade_len * 3
        num_cores = ceil(self.NUM_TCs / (cascade_len + 2))
        ram_size = num_cores * num_elements * 24 / 8 / 1024
        return ram_size

    def fpga21_num_ports(self, cascade_len):
        '''
        compute ports and fanin/fanout of the fpga21 core
        '''
        num_cores = ceil(self.NUM_TCs / (cascade_len + 2))
        num_inports_per_core = cascade_len + 1
        num_outports_per_core = 3
        in_ports = num_cores * num_inports_per_core
        outports = num_outports_per_core * num_cores
        return in_ports, outports

    def fpga21_bandwidth(self, cascade_len, peak=False):
        '''
        compute input and output bandwidth of the fpga21 core
        '''

        def datawidth_to_bandwidth(dwidth, latency = 1):
            dwidth /= (8 * (2**30))
            bandwidth = dwidth / (latency * 1./self.FREQ * 1e-6)
            return bandwidth

        if peak:        
            datawidth = 8 * 10
            inports, outports = self.fpga21_num_ports(cascade_len)
            peak_input_bandwidth = inports * datawidth
            peak_out_bandwidth = outports * 24
            
            return datawidth_to_bandwidth(peak_input_bandwidth), \
                    datawidth_to_bandwidth(peak_out_bandwidth)
        else:
            total_inputs = self.a_h * self.a_w + self.b_w * self.b_h
            total_inputs *= 8
            total_outputs = self.a_h * self.b_w * 24
            matA_size = (self.a_h, self.a_w)
            matB_size = (self.b_h, self.b_w)

            total_ops =  self.total_ops()
            chain_loading_lat = 3 * (cascade_len + 1)
            block_matA_size = (3, matA_size[1])
            block_matB_size = (matA_size[1], chain_loading_lat)
            a_loading_grps = ceil(matA_size[1] / (cascade_len * 10))

            compute_block_ops = block_matA_size[0] * block_matA_size[1] * 2 * block_matB_size[1]
            
            total_lat = chain_loading_lat + chain_loading_lat * a_loading_grps + \
                                4 + cascade_len * 2

            num_cores = ceil(self.NUM_TCs / (cascade_len + 2))
            num_blocks = ceil(total_ops / compute_block_ops)
            pipeline_iters = ceil(num_blocks / num_cores)
            total_lat *= pipeline_iters
            return datawidth_to_bandwidth(total_inputs, total_lat), \
                    datawidth_to_bandwidth(total_outputs, total_lat)

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
    
    BFP16_ADDER_LAT = 3.0
    BFP16_MULT_LAT = 3.0
    
    FP16_DIV_LAT = 8.0
    FP16_COMP_LAT = 2.0
    FP16_ADDER_LAT = 16.0
    FP16_MULT_LAT = 8.0
    FP16_SQRT_LAT = 8.0

    BFP16_ADDER_RES = 1.0/3.0
    BFP16_MULT_RES = 1./30.
    
    FP16_ADDER_RES = 200
    FP16_MULT_RES = 100
    FP16_COMP_RES = 10
    FP16_DIV_RES = 100

    WORD_SIZE = 2

    def __init__(self, embd_size=768.0, num_layers=0.0, num_heads=0.0, read_exp_samples=False, max_seq_len=320., \
                    exp_sample_path='params/scrs_sampled.npy', att_sample_path='params/attentions_sampled.npy'):
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
            atten_len, all_attens = (np.load(attention_file))[0], []
            for i in range(atten_len): all_attens.append(np.load(attention_file))

        # with open(exp_path, 'rb') as exps_file: 
        #     for i in range(atten_len): exps.append(np.load(exps_file))

        self.exps = all_attens

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

    def probe_exps(self, num_samples = -1):
        '''
        extracting features of the exponential function outputs inside the softmax
        '''
        if num_samples == -1:
            num_samples = len(self.exps)

        dat_sampled = random.sample(self.exps, num_samples)
        
        # num zeros in each head
        fig, ax = plt.subplots(1, 1, figsize=(24, 4))
        indices = ["{}".format(i+1) for i in range(144)]
        for i in range(144):
            if i%12 == 6: indices[i] = "layer {}".format(int(i/12) + 1)

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

        #distribution of the nonzero numbers inside the exp dat
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        nonzero_per_row = []
        for dat in dat_sampled:
            new_nonzero_per_row = np.count_nonzero(dat, axis=-1).flatten()
            nonzero_per_row.append(new_nonzero_per_row)

        nonzero_per_row = np.concatenate(nonzero_per_row)
        # calculate ratio of none zeros
        ratio_nzeros = (nonzero_per_row > 30).sum() / nonzero_per_row.size
        print("ratio of #none-zeros larger than 30: ", ratio_nzeros)
        # calculate lengths of nonzeros larger than 30
        nonzero_large = nonzero_per_row[nonzero_per_row > 30]
        print("average/std len of nonzeros larger than 30: ", np.mean(nonzero_large), np.std(nonzero_large))
        # calculate zeros
        ratio_zeros = (nonzero_per_row == 0).sum() / nonzero_per_row.size
        print("proportion of zeros: ", ratio_zeros)
        hists, bins, _ = ax.hist(nonzero_per_row, bins=40, range=(0, self.max_seq_len), weights=[1./nonzero_per_row.size]*nonzero_per_row.size)
        print("summation of the hists: ", sum(hists))
        ax.grid(linestyle='--', color='grey', alpha=0.4)
        ax.set_xlabel("#nonzero values")
        ax.set_ylabel("#rows")
        ax.set_xlim(xmin=0, xmax=self.max_seq_len)
        ax.set_ylim(ymin=0)
        fig.tight_layout()
        fig.savefig("res_fig/exps_nonzero_dist.pdf")
        plt.clf()

    def softmax_resources(self, p1, p2, l3, quant_bits, mvm_tcore):
        exp_resources = p1 * (2 ** quant_bits - 1) * self.FP16_COMP_RES
        exp_mem = p1 * (2 ** quant_bits * self.num_heads) / 2.0
        row_parallelism = self.max_seq_len / l3

        adder_tree_adders, rest_elems = 0.0, p1
        while rest_elems > 1.0:
            adder_tree_adders += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))

        adder_tree_adders = ceil(self.FP16_ADDER_RES * (p1-1))

        accu_mem = l3 * 2
        # counting the needed exp out buffer by p1 p2 and exp latency
        # and also the buffer needed to store the inputs
        # calculate exp out buffer
        theortical_max_elements = self.max_seq_len * l3
        # FIXME: lacking of the exp out buffer for the adder tree delay
        exp_out_buffer = (ceil(theortical_max_elements / p1) - ceil(theortical_max_elements / p2)) * p1
        div_resources = p2 * self.FP16_DIV_RES
        # FIXME: fixed mvm core size
        _, _, softmax_incycle_delayed, softmax_inputs_hiding = \
            self.softmax_incycles_lat_stratix((10, floor(mvm_tcore / (10+2))), (row_parallelism, p1))
        softmax_input_buffer = 0
        if not softmax_inputs_hiding:
            # we need the buffer holding all the attention scores waiting for the softmax input
            last_head_softmax_incycles_lat = softmax_incycle_delayed * (self.num_heads - 1)
            softmax_input_buffer = last_head_softmax_incycles_lat * p1 * row_parallelism
        
        total_mem = row_parallelism * (exp_mem + accu_mem + max(exp_out_buffer, 0.0) + softmax_input_buffer) * self.WORD_SIZE / 1024
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
        return self.FP16_COMP_LAT + lut_decoder_lat + lut_lat

    def softmax_lat(self, exp_dat=None, p1=1., p2=1., exp_h=-1):
        '''
        computing the sparse softmax latency
        exp_dat - output of attention exponent func  
        '''
        adder_tree_stages = p1-1
        if exp_dat is None:
            row_itlve_len = exp_h
            a_cols = self.max_seq_len * 0.2
            total_elems = a_cols * exp_h
            grp_cols = ceil(exp_h / p1)
        else:
            row_itlve_len = exp_dat.shape[0]
            a_cols = np.count_nonzero(exp_dat, axis=-1)
            total_elems = np.count_nonzero(exp_dat)
            grp_cols = ceil(exp_dat.shape[-1] / p1)

        # padding zeros for unaligned parallel sub-cols
        a_cols_1 = np.ceil(a_cols / p1)
        a_cols_2 = np.ceil(a_cols / p2)
        lat = self.qp_exp_lat()
        lat += ((adder_tree_stages+1) * self.FP16_ADDER_LAT)

        # check if extra exp out latency exists
        if p1 > p2:
            lat += ceil(total_elems / p2) - ceil(total_elems / p1) + 2

        return lat
        
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

    def baseline_softmax_resource(self, p, l, mvm_tcore):
        '''
        computing the baseline softmax dsps and memories
        '''
        exp_resource = self.FP16_MULT_RES + self.FP16_ADDER_RES
        log_resource = self.FP16_ADDER_RES

        tree_elems, rest_elems = 0.0, p
        while rest_elems > 1.0:
            tree_elems += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))
        
        res_all = (tree_elems + 1) * self.FP16_COMP_RES
        res_all += ceil(self.FP16_ADDER_RES * p)
        res_all += exp_resource * p
        res_all += (tree_elems + 1) * self.FP16_ADDER_RES
        res_all += ceil(self.FP16_ADDER_RES * 2 * p)
        res_all += exp_resource * p
        res_all += log_resource

        row_parallelism = np.ceil(self.max_seq_len/l)
        res_all *= row_parallelism

        exp_lat = self.FP16_ADDER_LAT + self.FP16_MULT_LAT + 1 + 2
        stg_2_lat = self.FP16_ADDER_LAT + exp_lat + (p-1) * self.FP16_ADDER_LAT + self.FP16_ADDER_LAT

        # FIXME: fixed mvm core shape
        _, _, softmax_incycle_delayed, softmax_inputs_hiding = \
            self.softmax_incycles_lat_stratix((10, floor(mvm_tcore / (10+2))), (row_parallelism, p))
        softmax_input_buffer = 0
        if softmax_inputs_hiding:
            # we need the buffer holding all the attention scores waiting for the softmax input
            last_head_softmax_incycles_lat = softmax_incycle_delayed * (self.num_heads - 1)
            softmax_input_buffer = last_head_softmax_incycles_lat * p * row_parallelism

        exp_mem = 64
        log_mem = 64 + 32
        buffer_mem = p * stg_2_lat

        mem_all = row_parallelism * (exp_mem * p * 2 + log_mem + buffer_mem + softmax_input_buffer) \
                    * self.WORD_SIZE / 1024

        return res_all, mem_all


    def baseline_softmax_lat(self, exp_dat=None, pa=4., exp_h=-1, return_2ndstg_lat = False):
        exp_lat = self.FP16_ADDER_LAT + self.FP16_MULT_LAT + 1 + 2
        ln_lat = 2 + self.FP16_ADDER_LAT

        dat_h = exp_h if exp_dat is None else exp_dat.shape[0]
        dat_h = self.max_seq_len if dat_h < 0 else dat_h
        dat_w = self.max_seq_len if exp_dat is None else exp_dat.shape[1]
        
        stg_1_lat = log2Up(pa) * self.FP16_COMP_LAT
        # stg_1_lat += max(self.COMP_LAT, dat_h-1) * (ceil(dat_w / pa) - 1)

        stg_2_lat = self.FP16_ADDER_LAT + exp_lat + log2Up(pa) * self.FP16_ADDER_LAT + self.FP16_ADDER_LAT
        # stg_2_lat += max(self.ADDER_LAT, dat_h-1) * (ceil(dat_w / pa) - 1)

        stg_3_lat = ln_lat + self.FP16_ADDER_LAT + exp_lat
        stg_3_lat += dat_h * dat_w / pa

        pipeline_lat = stg_1_lat + stg_2_lat + stg_3_lat

        if return_2ndstg_lat:
            return stg_2_lat
        else:
            return pipeline_lat

    def layernorm_res(self, row_parallel, p):
        '''
        compute layer norm resources
        '''
        # adder tree cost
        adder_tree_adders, rest_elems = 0.0, p
        while rest_elems > 1.0:
            adder_tree_adders += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))

        adder_tree_adders = ceil(self.FP16_ADDER_RES * (p-1))

        dsps = adder_tree_adders + self.FP16_ADDER_RES + self.FP16_MULT_RES
        dsps += (self.FP16_ADDER_RES + self.FP16_MULT_RES) * p + adder_tree_adders + self.FP16_ADDER_RES + self.FP16_MULT_RES
        dsps += (self.FP16_MULT_RES *3 + self.FP16_ADDER_RES) * p

        return dsps


    def layernorm_latency(self, row_parallel, p, mat_blk=(64.0, 64.0)):
        '''
        compute layer norm latency
        '''
        r = ceil(mat_blk[0] / row_parallel)
        tree_elems, rest_elems = 0.0, p
        while rest_elems > 1.0:
            tree_elems += float(2 ** int(log2Down(rest_elems)))
            rest_elems -= float(2 ** int(log2Down(rest_elems)))
        
        adder_tree_lat = (tree_elems + 1) * self.FP16_ADDER_LAT

        # FIXME: maybe need to change the mult here to a shifter
        stg1_lat = adder_tree_lat + self.FP16_ADDER_LAT + self.FP16_MULT_LAT
        # input row lat
        stg1_hidden_lat = max(self.FP16_ADDER_LAT, r)
        stg1_input_cycle = stg1_hidden_lat * ceil(mat_blk[1]/p)
        # FIXME: consider to change the 2nd multipler to a special square unit
        stg2_lat = self.FP16_ADDER_LAT + self.FP16_MULT_LAT + adder_tree_lat + \
                    self.FP16_ADDER_LAT + self.FP16_MULT_LAT + 2
        stg2_hidden_lat = max(self.FP16_ADDER_LAT, r)
        stg2_input_cycle = stg2_hidden_lat * ceil(mat_blk[1]/p)
        stg3_lat = self.FP16_SQRT_LAT + self.FP16_MULT_LAT * 2 + self.FP16_ADDER_LAT

        res = stg1_lat + stg1_input_cycle + stg2_lat + stg2_input_cycle + stg3_lat
        return res

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

    def matmul_lat_qkv_per_head_stratix(self, seq_len, cascade_len, num_tcs, ideal=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
        
            in_cycles, compute_lats = [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, self.embd_size,  self.embd_size, self.embd_size/self.num_heads, num_tcs=num_tcs)
                in_cycle, compute_lat = dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)
                in_cycles.append(in_cycle)
                compute_lats.append(compute_lat)

            return np.mean(in_cycles), np.mean(compute_lats)
        else:
            dpu_model = StratixDpuModel(seq_len, self.embd_size,  self.embd_size, self.embd_size/self.num_heads, num_tcs=num_tcs)
            return dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)
    
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

    def matmul_res_qktrans_per_head_stratix(self, tc_cascade_len, num_cores, seq_len=320):
        dpu_model = StratixDpuModel(seq_len, self.embd_size, self.embd_size, seq_len)
        return dpu_model.compute_resource(tc_cascade_len, num_cores)

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

    def matmul_lat_qktrans_per_head_stratix(self, seq_len, cascade_len, num_tcs, ideal=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, out_lats = [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, self.embd_size,  self.embd_size, l, num_tcs=num_tcs)
                in_cycle, out_lat = dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)
                in_cycles.append(in_cycle)
                out_lats.append(out_lat)

            return np.mean(in_cycles), np.mean(out_lats)
        else:
            dpu_model = StratixDpuModel(seq_len, self.embd_size,  self.embd_size, seq_len, num_tcs=num_tcs)
            return dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)

    def matmul_lat_vatt_per_head_stratix(self, seq_len, cascade_len, num_tcs, ideal=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, out_lats = [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, l, l, (self.embd_size / self.num_heads), num_tcs=num_tcs)
                in_cycle, out_lat = dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)
                in_cycles.append(in_cycle)
                out_lats.append(out_lat)

            return np.mean(in_cycles), np.mean(out_lats)
        else:
            dpu_model = StratixDpuModel(seq_len, seq_len, seq_len, (self.embd_size / self.num_heads), num_tcs=num_tcs)
            return dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)

    def matmul_lat_sparse_vatt_per_head_stratix(self, seq_len, cascade_len, num_tcs, ideal=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, out_lats = [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, l*0.2, l*0.2, (self.embd_size / self.num_heads), num_tcs=num_tcs)
                in_cycle, out_lat = dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)
                in_cycles.append(in_cycle)
                out_lats.append(out_lat)

            return np.mean(in_cycles), np.mean(out_lats)
        else:
            dpu_model = StratixDpuModel(seq_len, seq_len*0.2, seq_len*0.2, (self.embd_size / self.num_heads), num_tcs=num_tcs)
            return dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)

    def matmul_lat_selfatt_out_stratix(self, seq_len, cascade_len, num_tcs, ideal=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, out_lats = [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, self.embd_size, self.embd_size, self.embd_size, num_tcs=num_tcs)
                in_cycle, out_lat = dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)
                in_cycles.append(in_cycle)
                out_lats.append(out_lat)

            return np.mean(in_cycles), np.mean(out_lats)
        else:
            dpu_model = StratixDpuModel(seq_len, self.embd_size, self.embd_size, self.embd_size, num_tcs=num_tcs)
            return dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)

    def matmul_lat_post_att_fcs0_stratix(self, seq_len, cascade_len, num_tcs, ideal=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, out_lats = [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, 768, 768, 3072, num_tcs=num_tcs)
                in_cycle, out_lat = dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)
                in_cycles.append(in_cycle)
                out_lats.append(out_lat)

            return np.mean(in_cycles), np.mean(out_lats)
        else:
            dpu_model = StratixDpuModel(seq_len, 768, 768, 3072, num_tcs=num_tcs)
            return dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)

    def matmul_lat_post_att_fcs1_stratix(self, seq_len, cascade_len, num_tcs, ideal=False):
        if self.exps is not None:
            # print(__name__+": using acutal size of exp")
            actual_seq_len = [i.shape[-1] for i in self.exps]
            
            in_cycles, out_lats = [], []
            for l in actual_seq_len:
                dpu_model = StratixDpuModel(l, 3072, 3072, 768, num_tcs=num_tcs)
                in_cycle, out_lat = dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)
                in_cycles.append(in_cycle)
                out_lats.append(out_lat)

            return np.mean(in_cycles), np.mean(out_lats)
        else:
            dpu_model = StratixDpuModel(seq_len, 3072, 3072, 768, num_tcs=num_tcs)
            return dpu_model.compute_lat_teardown(cascade_len, ideal=ideal)
    

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
        
        equi_mvm_blk_size = (10, floor(mvm_tcore/12))
        mvm_in_cycles, _ = \
            self.matmul_lat_qkv_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)

        qktrans_in_cycles, qktrans_compute_lat = \
            self.matmul_lat_qktrans_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)

        # start of new score outputs:
        new_score_output_lat = mvm_in_cycles*4 + qktrans_compute_lat
        # start of softmax stg3:
        r, p = softmax_tcore
        start_of_softmax_stg3_lat = qktrans_compute_lat + \
                                        self.baseline_softmax_lat(pa=p, exp_h=ceil(self.max_seq_len/r), return_2ndstg_lat=True)
        if new_score_output_lat < start_of_softmax_stg3_lat:
            logging.warning("softmax mem in danger of overflow!")
            return False

        return True
        
    def softmax_incycles_lat_stratix(self, equi_mvm_blk_size, softmax_tcore):
        mvm_in_cycles, mvm_compute_delay = \
            self.matmul_lat_qkv_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        
        qktrans_in_cycles, qktrans_compute_delay = \
            self.matmul_lat_qktrans_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        qkv_qktrans_compute_lat = qktrans_in_cycles + mvm_in_cycles * 3 + qktrans_compute_delay
        r, p = softmax_tcore
        if self.exps is None:
            softmax_stg1_incycle = ceil(self.max_seq_len/r) * ceil(self.max_seq_len/p)
        else:
            softmax_stg1_incycle = np.mean([ceil(h.shape[-1]/r) * ceil(float(h.shape[-1])/p) \
                                            for h in self.exps])

        intermediate_lat = max(qkv_qktrans_compute_lat, softmax_stg1_incycle)
        softmax_stg1_incycle_delayed = 0
        softmax_inputs_hiding = True
        if qkv_qktrans_compute_lat < softmax_stg1_incycle:
            softmax_stg1_incycle_delayed = softmax_stg1_incycle - qkv_qktrans_compute_lat
            softmax_inputs_hiding = False
            logging.warn(f"{__name__}: softmax input cycles hidden failed")

        return intermediate_lat, softmax_stg1_incycle, softmax_stg1_incycle_delayed, softmax_inputs_hiding


    def attention_lat_stratix(self, mvm_tcore: float, softmax_tcore: tuple, softmax_type = "baseline"):
        '''
        softmax_tcore: r, p -> r: row parallelism, p -> column parallelism
        '''
        equi_mvm_blk_size = (10, floor(mvm_tcore / (10+2)))
        mvm_in_cycles, mvm_compute_delay = \
            self.matmul_lat_qkv_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        qktrans_in_cycles, qktrans_compute_delay = \
            self.matmul_lat_qktrans_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)

        # first 11 heads to cover the softmax latency by mvm compute:
        # compute latency of one head, the softmax incycles and if there's delay
        # intermediate latency: num of cycles to finish one head of q, k, v and qktrans computation, or
        # the number of softmax input cycles, depending on which one is larger.
        # it's the num of cycles to feed in one entire head into the self attention computation essentially.
        intermediate_lat, softmax_stg1_incycle, softmax_stg1_incycle_delayed, softmax_inputs_hiding = \
            self.softmax_incycles_lat_stratix(equi_mvm_blk_size, softmax_tcore)

        r, p = softmax_tcore
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

            softmax_lat = np.mean(np.array(temp_lats))
        else:
            temp_lats = []
            if len(flatten_exps) > 0:
                for inst in flatten_exps:
                    effective_inst = inst[:ceil(inst.shape[-1]/r), :]
                    temp_lats.append(self.softmax_lat(exp_dat=effective_inst, p1=p, p2=p))
            else:
                temp_lats.append(self.softmax_lat(p1=p, p2=p, exp_h=ceil(self.max_seq_len/r)))
                
            softmax_lat = np.mean(np.array(temp_lats))

        # predessesor heads: latency from the self attention starts to the last head softmax start
        predessesor_heads_lat = mvm_in_cycles * 2 + (intermediate_lat + softmax_stg1_incycle_delayed) * (self.num_heads-1)
        # softmax finish time (absolute time)
        # checking the first softmax finish latency
        # softmax_first_finish_time = mvm_in_cycles * 2 + qktrans_compute_delay + \
        #                                 softmax_stg1_incycle + softmax_lat
        # softmax_first_ddl = predessesor_heads_lat + intermediate_lat
        # softmax_compute_hiding = True
        # if softmax_first_ddl < softmax_first_finish_time:
        #     softmax_compute_hiding = False
        #     logging.warn("softmax compute hidden failed", softmax_first_finish_time, softmax_first_ddl)

        # accumulate VxAtt
        vatt_mult_incycles, _ = \
            self.matmul_lat_vatt_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        vatt_mult_incycles *= self.num_heads

        # last step: FC layer for output
        output_fc_incycles, output_fc_compute_delay = \
            self.matmul_lat_selfatt_out_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        output_fc_lat = output_fc_incycles + output_fc_compute_delay

        # check if the softmax latency is longer than the v att mult deadline
        res = predessesor_heads_lat + intermediate_lat + softmax_stg1_incycle_delayed
        res += max(vatt_mult_incycles, softmax_lat)
        softmax_compute_hiding = True
        if vatt_mult_incycles < softmax_lat:
            softmax_compute_hiding = False
            logging.warn("softmax compute hidden failed: {}, {}".format(vatt_mult_incycles, softmax_lat))

        res += output_fc_lat
        
        return res, softmax_compute_hiding

    def attention_linearfunc_lat_stratix(self, mvm_tcore: float):
        equi_mvm_blk_size = (10, floor(mvm_tcore / (10+2)))
        # last step: FC layer for output
        postatt_fc0_incycles, postatt_fc0_compute_delay = \
            self.matmul_lat_post_att_fcs0_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        postatt_fc1_incycles, postatt_fc1_compute_delay = \
            self.matmul_lat_post_att_fcs1_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)

        post_att_fc_lat = postatt_fc0_compute_delay + postatt_fc1_compute_delay + postatt_fc0_incycles
        ln_layer = self.layernorm_latency(2, 6, mat_blk=(self.max_seq_len, 768))

        print("ln layers {}, post att fc latency {}".format(ln_layer, post_att_fc_lat))
        print("ratio: {}".format(ln_layer / (ln_layer + post_att_fc_lat)))
        return 

    def attention_mvm_only_lat_stratix(self, mvm_tcore: float):
        '''
        compute mvm only latency of the attention, used to estimate dynamic utilization of the mvm unit.
        '''
        equi_mvm_blk_size = (10, floor(mvm_tcore / 10))
        mvm_in_cycles, _ = \
            self.matmul_lat_qkv_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)

        # first 11 heads to cover the softmax latency by mvm compute:
        qktrans_in_cycles, _ = \
            self.matmul_lat_qktrans_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        qkv_qktrans_compute_lat = (mvm_in_cycles * 3 + qktrans_in_cycles) * self.num_heads
        
        vatt_mult_incycles, _ = \
            self.matmul_lat_vatt_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        vatt_mult_incycles *= self.num_heads

        # last step: FC layer for output
        output_fc_incycles, output_fc_compute_lat = \
            self.matmul_lat_selfatt_out_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        output_fc_lat = output_fc_incycles + output_fc_compute_lat

        total_lat = qkv_qktrans_compute_lat + vatt_mult_incycles + output_fc_lat

        return total_lat

    def attention_bandwidth_stratix(self, mvm_tcore: float, softmax_tcore: tuple, softmax_type = "baseline"):
        equi_mvm_blk_size = (10, floor(mvm_tcore / 12))
        mvm_in_cycles, _ = \
            self.matmul_lat_qkv_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)

        # first 11 heads to cover the softmax latency by mvm compute:
        qktrans_in_cycles, qktrans_compute_lat = \
            self.matmul_lat_qktrans_per_head_stratix(self.max_seq_len, equi_mvm_blk_size[0], equi_mvm_blk_size[1], ideal=True)
        qkv_qktrans_compute_lat = qktrans_in_cycles + mvm_in_cycles * 3 + qktrans_compute_lat
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
        softmax_first_finish_time = mvm_in_cycles * 2 + qktrans_compute_lat + \
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
    bert_hw_model_d = BertModel(read_exp_samples=False, num_layers=12, num_heads=12, max_seq_len=320)
    bert_hw_model_s = BertModel(read_exp_samples=True)
    bert_hw_model_s.probe_exps()
    exit()

    stratix_dpu_d = StratixDpuModel(bert_hw_model_d.max_seq_len, bert_hw_model_d.max_seq_len, bert_hw_model_d.max_seq_len, bert_hw_model_d.embd_size / bert_hw_model_d.num_heads, freq=500, num_tcs=3960)

    all_mvm_lat_s = []
    for exps in bert_hw_model_s.exps:
        for l in exps:
            for h in l:
                stratix_dpu_s = StratixDpuModel(bert_hw_model_d.max_seq_len, bert_hw_model_d.max_seq_len, bert_hw_model_d.max_seq_len, bert_hw_model_d.embd_size / bert_hw_model_d.num_heads, exp_dat=h, freq=500, num_tcs=3960)
                all_mvm_lat_s.append(stratix_dpu_s.compute_lat(10, ideal=False))

    mvm_lat_d = stratix_dpu_d.compute_lat(10, ideal=False)
    mvm_lat_s = np.average(all_mvm_lat_s)

    print(mvm_lat_d, mvm_lat_s)

    # print("Tflops: ", 2 * tensor_mat_flops(20, 7, 2))