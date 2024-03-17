from os import listdir
from os.path import isfile


def get_pts_under_dir(base_path: str) -> list[str]:
    inst_list = [f.split(".")[0] \
                 for f in listdir(base_path) \
                    if isfile(base_path + f) and f[0] == "i" and f.endswith(".pt")]
    inst_list = [base_path + i + ".pt" for i in inst_list]
    return inst_list