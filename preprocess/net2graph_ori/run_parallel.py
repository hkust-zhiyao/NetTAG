import os, time, json, re
from multiprocessing import Pool

def run_one_design(design_name):
    print(design_name)
    tmp_dir = f"./parse_split_template/"
    new_dir = f"./parse_split_tmp_{design_name}"
    os.system(f"cp -r {tmp_dir} {new_dir}")

    auto_run_tmp_dir = f"{new_dir}/net2graph_template.py"
    auto_run_dir = f"{new_dir}/net2graph.py"
    with open (auto_run_tmp_dir, 'r') as f:
        lines = f.readlines()
    
    with open (auto_run_dir, 'w') as f:
        for line in lines:
            line = re.sub(r"DESIGN_NAME_HERE", design_name, line)
            f.writelines(line)

    os.chdir(new_dir)
    os.system("python3 net2graph.py")
    os.chdir("../")
    os.system(f"rm -rf {new_dir}")


def run_all_parallel(design_lst):
    with Pool(50) as p:
        p.map(run_one_design, design_lst)
        p.close()
        p.join()

if __name__ == '__main__':

    with open(f"../../data_collect/data_js/design_list.json", 'r') as f:
        design_lst = json.load(f)

    run_all_parallel(design_lst)

    # for design in design_lst:
    #     run_one_design(design)
    #     print(f"Finish: {design}\n")

    # run_all_parallel(['b20', 'b14'])