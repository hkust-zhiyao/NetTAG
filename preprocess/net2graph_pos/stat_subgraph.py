### check the number of files in the directory
import os
import json

def check_dir(dir_path):
    return len(os.listdir(dir_path))

if __name__ == '__main__':
    dir_path = "./saved_graph_split"
    idx = 0
    for dir in list(os.listdir(dir_path)):
        design_name = dir
        ll = check_dir(f"{dir_path}/{dir}")
        print(f"Current Design: {design_name}, Number of files: {ll}")
        if os.path.exists(f"{dir_path}/{dir}/{design_name}_node_dict.pkl"):
            # pass
            print("Finish")
            os.system(f"rm -rf ./parse_split_tmp_{design_name}")
        else:
            print("Not Finish")
        idx += ll
    print(f"Total number of files: {idx}")