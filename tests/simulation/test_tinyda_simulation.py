import os
from pathlib import Path
import logging
import shutil

#import arviz as az
import pytest
import ray

from definitions import ROOT_DIR
from bp_simunek.simulation.flow_wrapper import Wrapper
from bp_simunek.samplers.tinyda_flow import TinyDAFlowWrapper
from bp_simunek.samplers.idata_tools import save_idata_to_file
from bp_simunek.plotting.flow_plots import generate_all_flow_plots

script_dir = os.path.dirname(os.path.realpath(__file__))

#@pytest.mark.skip
def test_simulation11():
    #ray.init(runtime_env={"working_dir": ROOT_DIR})
    os.chdir(script_dir)
    observe_path = Path(script_dir, "../measured_data").absolute()
    template_dir = Path("templates", "test_workdir11").absolute()
    work_dir = Path(ROOT_DIR, "output", "test11").absolute()

    # clean workdir
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)

    # copy template to workdir
    shutil.copytree(template_dir, work_dir)

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

@pytest.mark.skip
def test_simulation11_fail():
    #ray.init(runtime_env={"working_dir": ROOT_DIR})
    os.chdir(script_dir)
    observe_path = Path(script_dir, "../measured_data").absolute()
    template_dir = Path("templates", "test_workdir11_fail").absolute()
    work_dir = Path(ROOT_DIR, "output", "test11_fail").absolute()

    # clean workdir
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)

    # copy template to workdir
    shutil.copytree(template_dir, work_dir)

    # init wrapper - load config
    wrap = Wrapper(work_dir)

    # add observe path to config
    wrap.set_observe_path(observe_path)

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    # run sampling process
    idata = tinyda_wrapper.sample()

    # check if sampling was successful
    assert idata
    assert idata["posterior"].sizes["draw"] == tinyda_wrapper.sample_count + 1

    # save results
    save_idata_to_file(idata, folder_path=work_dir, filename="flow.sim11.idata")

    # generate plots
    generate_all_flow_plots(idata, folder=work_dir)

@pytest.mark.skip
def test_simulation12_mlda():
    os.chdir(script_dir)
    observe_path = Path(script_dir, "../measured_data").absolute()
    template_dir = Path("templates", "test_workdir12").absolute()
    work_dir = Path(ROOT_DIR, "output", "test12").absolute()

    # clean workdir
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)

    # copy template to workdir
    shutil.copytree(template_dir, work_dir)

    # init wrapper - load config
    wrap = Wrapper(work_dir)

    # add observe path to config
    wrap.set_observe_path(observe_path)

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    # run sampling process
    idata = tinyda_wrapper.sample()

    # check if sampling was successful
    assert idata
    assert idata["posterior"].sizes["draw"] == tinyda_wrapper.sample_count + 1

    # save results
    save_idata_to_file(idata, folder_path=work_dir, filename="flow.sim12.idata")

    # generate plots
    generate_all_flow_plots(idata, folder=work_dir)


