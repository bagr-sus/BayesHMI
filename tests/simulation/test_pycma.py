import os
from pathlib import Path
import logging
import shutil

from cma import CMADataLogger

from definitions import ROOT_DIR
from bp_simunek.simulation.flow_wrapper import Wrapper
from bp_simunek.samplers.pycma_flow import PyCMAFlowWrapper
from bp_simunek.plotting.plotting_tools import save_plot


script_dir = os.path.dirname(os.path.realpath(__file__))


def test_pycma1():
    os.chdir(script_dir)
    observe_path = Path(script_dir, "../measured_data").absolute()
    template_dir = Path("templates", "test_workdir11").absolute()
    work_dir = Path(ROOT_DIR, "output", "test11_pycma").absolute()

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
    es = pycma_wrapper.optimize(maxevals=50)
    os.chdir(script_dir)
    es.logger.plot()
    save_plot(os.path.join(work_dir, "es_plot.pdf"))
    pycma_wrapper.save_results_to_file(es, os.path.join(work_dir, "results.txt"))
