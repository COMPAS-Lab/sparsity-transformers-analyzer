from os import listdir
from os.path import isfile


def get_pts_under_dir(base_path: str, postfix: str, datatype=None) -> list[str]:
    inst_list = []
    if base_path[-1] != "/":
        base_path = base_path + "/"

    if datatype:
        for f in listdir(base_path):
            if isfile(base_path + f) and f[0] == "i" and f.endswith("." + postfix):
                if f.split("_")[-1].split(".")[0] == datatype:
                    inst_list.append(base_path + f)
    else:
        for f in listdir(base_path):
            if isfile(base_path + f) and f[0] == "i" and f.endswith("." + postfix):
                inst_list.append(base_path + f)
        
    return inst_list