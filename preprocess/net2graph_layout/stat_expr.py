### check the number of files in the directory
import os
import json

def check_dir(dir_path):
    return len(os.listdir(dir_path))

if __name__ == '__main__':
    dir_path = "./saved_expr"
    idx = 0
    for dir in list(os.listdir(dir_path)):
        design_name = dir
        ll = check_dir(f"{dir_path}/{dir}")
        print(f"Current Design: {design_name}, Number of files: {ll}")
        idx += ll
    print(f"Total number of files: {idx}")