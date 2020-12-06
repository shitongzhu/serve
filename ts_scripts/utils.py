import os
import sys


def is_gpu_instance():
    return True if os.system("nvidia-smi") == 0 else False


def is_conda_env():
    return True if os.system("conda") == 0 else False


def check_python_version():
    req_version = (3, 6)
    cur_version = sys.version_info

    if cur_version >= req_version:
        print("System version" + str(cur_version))
        print("TorchServe supports Python 3.6 and higher only. Please upgrade")
        exit(1)