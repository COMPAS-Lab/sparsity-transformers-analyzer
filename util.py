from os import listdir
from os.path import isfile
import json

def get_pts_under_dir(base_path: str, postfix: str, datatype = None, fname_filter = None) -> list[str]:
    inst_list = []
    if base_path[-1] != "/":
        base_path = base_path + "/"

    if datatype:
        for f in listdir(base_path):
            if isfile(base_path + f) and f.endswith("." + postfix):
                if f.split("_")[-1].split(".")[0] == datatype:
                    if fname_filter:
                        if fname_filter in f:
                            inst_list.append(base_path + f)
                    else:
                        inst_list.append(base_path + f)
    else:
        for f in listdir(base_path):
            if isfile(base_path + f) and f.endswith("." + postfix):
                if fname_filter:
                    if fname_filter in f:
                        inst_list.append(base_path + f)
                else:
                    inst_list.append(base_path + f)
        
    return inst_list

def read_lat_files(r, c, cl):
    with open(f"res_fig/block_prune/spmm_rr_lat_diff_wfifo_200_r{r}_c{c}_cl{cl}.json") as f:
        a = json.load(f)
        print(a["inst_0"]["dense_avg_tops"])
        print(a["inst_0"]["avg_tops"])
        print(a["inst_0"]["max_out_bd_req"])
        print(a["inst_0"]["max_out_bd_req_bfp12"])