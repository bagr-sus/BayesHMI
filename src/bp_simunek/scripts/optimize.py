import logging
import os
from pathlib import Path
import shutil
import sys

import ray

from ..simulation.flow_wrapper import Wrapper
from ..samplers.pycma_flow import PyCMAFlowWrapper
from ..plotting.plotting_tools import save_plot

from definitions import ROOT_DIR

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
    ray.init(_temp_dir=tmp_dir_mount)

    os.chdir(script_dir)
    observe_path = Path(cfg_path).absolute()
    template_dir = Path(cfg_path).absolute()
    workdir = os.environ.get("SCRATCHDIR")
    if workdir is None:
        work_dir = Path(ROOT_DIR, "output", "test12").absolute()
    else:
        work_dir = Path(os.path.join(workdir, "")).absolute()

    logging.info("Using workdir %s", work_dir)

    # copy template to workdir
    shutil.copytree(template_dir, work_dir, dirs_exist_ok=True)

    # init wrapper - load config
    wrap = Wrapper(work_dir)

    # add observe path to config
    wrap.set_observe_path(observe_path)

    # tinyda + flow123 wrapper
    pycma_wrapper = PyCMAFlowWrapper(wrap)

    # run sampling process
    es = pycma_wrapper.optimize()
    pycma_wrapper.save_results_to_file(es, os.path.join(work_dir, "results.txt"))
    es.plot()
    save_plot(os.path.join(work_dir, "es_plot.pdf"))
    es.save(name="results.txt")
    

if __name__ == "__main__":
    sample()