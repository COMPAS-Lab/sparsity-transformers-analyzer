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

def find_positive_integer_pairs(x: int, filter) -> list[tuple[int, int]]:
    """
    Finds all possible combinations of a pair of positive integers (x1, x2)
    that satisfy x1 * (x2 + 2) = x.

    Args:
        x: The target positive integer.

    Returns:
        A list of tuples, where each tuple (x1, x2) is a pair of positive integers
        whose product with (x2 + 2) is x. Returns an empty list if x is not
        a positive integer or no such pairs exist.
    """
    # Ensure x is a positive integer
    if not isinstance(x, int) or x <= 0:
        print(f"Input x must be a positive integer. Received: {x}")
        return []

    pairs = []
    # Iterate for x1 from 1 up to x (inclusive).
    # Since x1 is a factor of x, and x1 and x2 are positive, x1 cannot be greater than x.
    for x1 in range(1, x + 1):
        # Check if x1 is a divisor of x.
        # If x % x1 == 0, then x1 divides x evenly.
        if x % x1 == 0:
            # Let temp_val be (x2 + 2). So, temp_val = x / x1.
            temp_val = x // x1
            
            # For x2 to be a positive integer (x2 > 0), temp_val must be greater than 2.
            # Because x2 = temp_val - 2, if temp_val is 1 or 2, x2 would be 
            # non-positive.
            if temp_val > 2:
                x2 = temp_val - 2  # Calculate the corresponding x2

                # We already ensured x2 > 0 with the temp_val > 2 check,
                # but an explicit check reinforces the constraint.
                if x2 > 1 and filter(x1, x2):
                    pairs.append((x1, x2))
            
    # Sort the pairs for consistent output, primarily by x1 then by x2.
    pairs.sort()
    
    return pairs
