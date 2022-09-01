#!/usr/bin/env python3

import os
import subprocess


def remove_all(path_name):
    files = os.listdir(path_name)
    for file in files:
        subprocess.call(("rm", "-rf", path_name + "/" + file))


def remove():
    paths = ["build", "bin"]
    for path in paths:
        remove_all(path)


def remove_lightgbm_src():
    subprocess.call(("rm", "-rf", "lightgbm_src/lightgbm"))
    subprocess.call(("rm", "-rf", "lightgbm_src/lib_lightgbm.so"))


def remove_pynb():
    files = os.listdir(".")
    for file in files:
        if file.startswith(".ipynb"):
            subprocess.call(("rm", "-rf", file))


if __name__ == "__main__":
    remove()
    remove_lightgbm_src()
    remove_pynb()
