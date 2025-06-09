import logging
import os
from pathlib import Path
import shutil
import sys

import ray

from ..simulation.flow_wrapper import Wrapper
from ..samplers.tinyda_flow import TinyDAFlowWrapper
from ..samplers.idata_tools import save_idata_to_file
from ..plotting.flow_plots import generate_all_flow_plots

from ..definitions import ROOT_DIR

script_dir = os.path.dirname(os.path.realpath(__file__))

def sample():
    # probably not the best solution
    # 107 char limit for socket path
    #tmp_dir_symlink = os.path.join(os.path.expanduser("~"), ".r")
    #logging.info(tmp_dir_symlink)
    #if not os.path.islink(tmp_dir_symlink):
    #    raise Exception("Missing symlink for Ray temp storage")
    tmp_dir_mount = sys.argv[1]
    cfg_path = sys.argv[2]
    #ray.init(_temp_dir=tmp_dir_mount)

    os.chdir(script_dir)
    observe_path = Path(cfg_path).absolute()
    #template_dir = Path(cfg_path).absolute()
    workdir = os.environ.get("SCRATCHDIR")
    if workdir is None:
        work_dir = Path(ROOT_DIR, "output", "test12").absolute()
    else:
        work_dir = Path(os.path.join(workdir, "")).absolute()

    logging.info("Using workdir %s", work_dir)

    # copy template to workdir
    #shutil.copytree(template_dir, work_dir, dirs_exist_ok=True)

    # init wrapper - load config
    wrap = Wrapper(work_dir)

    # add observe path to config
    wrap.set_observe_path(observe_path)

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    # run sampling process
    idatas = tinyda_wrapper.sample()

    # save results
    for level, idata in enumerate(idatas):
        idata_name = f"{tinyda_wrapper.number_of_chains}x{tinyda_wrapper.sample_count}_{level}.idata"
        save_idata_to_file(idata, folder_path=work_dir, filename=idata_name)

    # generate plots
    generate_all_flow_plots(idatas[-1], folder=work_dir)

if __name__ == "__main__":
    sample()