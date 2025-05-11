import os
import subprocess
import time
import hashlib

import logging
import numpy as np
import pyvista as pv
import itertools
import collections
import shutil
import csv
import yaml
from typing import List
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

from bp_simunek import common

def generate_time_axis(config_dict):
    end_time = float(config_dict["end_time"])
    output_times = config_dict["output_times"]

    print("end time %d", end_time)
    print("output times")
    print(output_times)

    # create time axis
    times = []
    for dt in output_times:
        b = float(dt["begin"])
        s = float(dt["step"])
        e = float(dt["end"])
        times.extend(np.arange(b, e, s))
    times.append(end_time)
    return times


class Flow123dSimulation:

    def __init__(self, config, clean):

        if "sample_subdir" in config:
            self.work_dir = Path(config["sample_subdir"])
            print(config["sample_subdir"])
        else:
            self.work_dir = Path(config["work_dir"])


        self.env_path = None
        # workdir override to allow correct paths for multinode
        # paths will be resolved from env_path env variable
        if "workdir_override" in config:
            workdir_override = config["workdir_override"]
        else:
            workdir_override=False

        if workdir_override:
            if "env_path" not in config:
                logging.warning("No env_path provided for workdir override, defaulting to SCRATCHDIR")
                self.env_path = "SCRATCHDIR"
            else:
                logging.info("Workdir override detected, using %s env variable", config["env_path"])
                self.env_path = config["env_path"]


        self.clean = clean
        self._config = config
        self.sample_counter = -1
        self.sample_dir = Path(".")
        self.sample_output_dir = "output"
        self.param_hash = ""
        self.stdout_path = ""
        self.stderr_path = ""

    def set_parameters(self, data_par):
        # if env_path is set, use it to overwrite work_dir
        # set_parameters is called on the correct nodes, so resolution takes place here
        if self.env_path is not None:
            self._config["work_dir"] = os.environ[self.env_path]
            self._config["common_files_dir"] = os.path.join(self._config["work_dir"], "common_files")
            self.work_dir = Path(self._config["work_dir"])
            # set self.env_path to None to only do this fix once
            self.env_path = None
            logging.info("Workdir override set to %s", self._config["work_dir"])

        param_list = self._config["parameters"]
        assert(len(data_par) == len(param_list))

        # init hasher object
        hasher = hashlib.md5()

        for idx, param in enumerate(param_list):
            # append every parameter to hasher's input
            hasher.update(data_par[idx].tobytes())
            pname = param["name"]
            assert pname in self._config["hm_params"], pname + " not in hm_params"
            self._config["hm_params"][pname] = data_par[idx]

        # hash into hex, only taking the first 16 chars
        self.param_hash = hasher.hexdigest()[:16]

    def get_observations(self):
        try:
            print("get observations from flow_wrapper")
            res = self.calculate(self._config)
            return res
        except ValueError:
            print("flow_wrapper failed for unknown reason.")
            return -1000, np.empty(0)

    def calculate(self, config_dict):
        """
        The program changes to <work_dir> directory.
        does all the data preparation, passing
        running simulation
        extracting results
        """

        # create sample dir
        #self.sample_counter = self.sample_counter + 1
        #self.sample_dir = self.work_dir/("solver_" + str(config_dict["solver_id"]).zfill(2) +
        #                             "_sample_" + str(self.sample_counter).zfill(3))

        # use param hash as folder name
        self.sample_dir = self.work_dir/(self.param_hash)
        last_hash = self.param_hash
        hasher = hashlib.md5(last_hash.encode("utf-8"))
        # while loop to avoid hash collisions, just in case
        while True:
            if not self.sample_dir.exists():
                break
            logging.warning("Hash collision detected in folder names, fixing")
            new_hash = hasher.hexdigest()[:16]
            hasher = hashlib.md5(new_hash.encode("utf-8"))
            self.sample_dir = self.work_dir/(new_hash)

        self.sample_dir.mkdir(mode=0o775, exist_ok=True)
        assert self.sample_dir.exists()

        #logging.info("=========================== RUNNING CALCULATION " +
        #      "solver {} ".format(config_dict["solver_id"]).zfill(2) +
        #      "sample {} ===========================".format(self.sample_counter).zfill(3))
        logging.info("RUNNING SIMULATION\n")
        logging.info(self.sample_dir)
        os.chdir(self.sample_dir)

        # collect only
        # not used in any project (was used for dev in WGC2020)
        # if config_dict["collect_only"]:
        #     return 2, self.collect_results(config_dict)

        logging.info("Creating mesh...")
        comp_mesh = self.prepare_mesh(config_dict, cut_tunnel=False)

        mesh_bn = os.path.basename(comp_mesh)
        config_dict["hm_params"]["mesh"] = mesh_bn

        # endorse_2Dtest.read_physical_names(config_dict, comp_mesh)

        if config_dict["mesh_only"]:
            return -10, None  # tag, value_list

        # endorse_2Dtest.prepare_hm_input(config_dict)
        hm_succeed, fo = self.call_flow(config_dict, 'hm_params', result_files=["flow_observe.yaml"])

        if not hm_succeed:
            # raise Exception("HM model failed.")
            # "Flow123d failed (wrong input or solver diverged)"
            logging.warning("Flow123d failed.")
            # still try collect results
            try:
                collected_values = self.collect_results(config_dict, fo)
                logging.info("Sample results collected.")
                return 3, collected_values  # tag, value_list
            except:
                logging.error("Collecting sample results failed:")
                logging.error(traceback.format_exc())
                # traceback.print_exc()
                return -3, None
            # return -1, None  # tag, value_list
        print("Running Flow123d - HM...finished")

        logging.info("Finished computation")

        # collected_values = self.collect_results(config_dict)
        # print("Sample results collected.")
        # return 1, collected_values  # tag, value_list

        try:
            collected_values = self.collect_results(config_dict, fo)
            logging.info("Sample results collected.")
            return 1, collected_values  # tag, value_list
        except:
            logging.error("Collecting sample results failed:")
            logging.error(traceback.format_exc())
            # traceback.print_exc()
            return -3, None

    # def check_data(self, data, minimum, maximum):
    #     n_times = len(endorse_2Dtest.result_format()[0].times)
    #     if len(data) != n_times:
    #         raise Exception("Data not corresponding with time axis.")
    #
    #     if np.isnan(np.sum(data)):
    #         raise Exception("NaN present in extracted data.")
    #
    #     min = np.amin(data)
    #     if min < minimum:
    #         raise Exception("Data out of given range [min].")
    #     max = np.amax(data)
    #     if max > maximum:
    #         raise Exception("Data out of given range [max].")

    def collect_results(self, config_dict, fo: common.FlowOutput):
        data = None
        if config_dict["collect_results"]["collect_vtk"]:
            data = self.collect_results_vtk(config_dict, fo)
        if config_dict["collect_results"]["collect_observe"]:
            data = self.collect_results_observe(config_dict, fo)

        #if config_dict["clean_sample_dir"]:
        #    shutil.rmtree(self.sample_dir)

        return data

    def clean_sample_dir(self, config_dict):
        if config_dict["clean_sample_dir"]:
            shutil.rmtree(self.sample_dir)

    def copy_sample_dir(self):
        shutil.copytree(self.sample_dir, os.path.join(self.work_dir, "error_samples", self.param_hash))

    def collect_results_vtk(self, config_dict, fo: common.FlowOutput):
        # Load the PVD file
        # pvd_file_path = os.path.join(self.sample_output_dir, "flow.pvd")
        field_name = "pressure_p0"
        pvd_reader = pv.PVDReader(fo.hydro.spatial_file.path)

        field_data_list = []
        for time_frame in range(len(pvd_reader.time_values)):
            pvd_reader.set_active_time_point(time_frame)
            mesh = pvd_reader.read()[0]  # MultiBlock mesh with only 1 block

            field_data = mesh[field_name]
            field_data_list.append(field_data)

        sample_data = np.stack(field_data_list)
        sample_data = sample_data.reshape((1, *sample_data.shape))  # axis 0 - sample

        return sample_data

    def collect_results_observe(self, config_dict, fo: common.FlowOutput):
        pressure_points2collect = config_dict["observe_points"]
        cond_points2collect = config_dict["conductivity_observe_points"]

        values = np.empty((0,))

        # the times defined in input
        times = np.array(generate_time_axis(config_dict))
        with open(fo.hydro.observe_file.path, "r") as f:
            loaded_yaml = yaml.safe_load(f)

            vals = self.get_from_observe(loaded_yaml, pressure_points2collect, 'pressure_p0', times)
            values = np.concatenate((values, vals), axis=None)

            vals = self.get_from_observe(loaded_yaml, cond_points2collect, 'conductivity', times[-1])
            vals = np.log10(vals)  # consider log10!
            values = np.concatenate((values, vals), axis=None)

        if self._config["make_plots"]:
            self.observe_time_plot(config_dict, fo)
        # flatten to format: [Point0_all_all_times, Point1_all_all_times, Point2_all_all_times, ...]
        res = values.flatten()
        return res


    def get_from_observe(self, observe_dict, point_names, field_name, select_times=None):
        points = observe_dict['points']
        all_point_names = [p["name"] for p in points]
        # logging.info('all_point_names' + str(all_point_names))
        # logging.info('point_names' + str(point_names))
        points2collect_indices = []
        for p2c in point_names:
            tmp = [i for i, pn in enumerate(all_point_names) if pn == p2c]
            assert len(tmp) == 1
            points2collect_indices.append(tmp[0])

        # print("Collecting results for observe points: ", point_names)
        data = observe_dict['data']
        data_values = np.array([d[field_name] for d in data])
        values = data_values[:, points2collect_indices]
        obs_times = np.array([d["time"] for d in data]).transpose()
        if select_times is not None:
            # check that observe data are computed at all times of defined time axis
            all_times_computed = np.all(np.isin(select_times, obs_times))
            #logging.warning(select_times)
            #logging.warning(obs_times)
            if not all_times_computed:
                raise Exception("Observe data not computed at all times as defined by input!")
            # skip the times not specified in input
            t_indices = np.isin(obs_times, select_times).nonzero()
            values = values[t_indices]
        values = values.transpose()

        # if "smooth_factor" in config_dict.keys():
        #     smooth_factor = config_dict["smooth_factor"]
        #     for i in range(len(values)):
        #         values[i] = self.smooth_ode(times, values[i], smooth_factor)

        return values

    def call_flow(self, config_dict, param_key, result_files) -> (bool, common.FlowOutput):
        """
        Redirect sstdout and sterr, return true on succesfull run.
        :param result_files: Files to be computed - skip computation if already exist.
        :param param_key: config dict parameters key
        :param config_dict:
        :return:
        """

        params = config_dict[param_key]
        arguments = config_dict["local"]["flow_executable"].copy()

        if all([os.path.isfile(os.path.join(self.sample_output_dir, f)) for f in result_files]):
            status = True
            completed_process = subprocess.CompletedProcess(args=arguments, returncode=0)
        else:
            fname = params["in_file"]
            input_template = common.File(Path(config_dict["common_files_dir"])/(fname + '_tmpl.yaml'))
            completed_process, stdout, stderr = common.flow_call(self.sample_dir, arguments, input_template, params)
            self.stdout_path = stdout.path
            self.stderr_path = stderr.path
        status, fo = common.flow_check(self.sample_dir, completed_process, result_files)

        return status, fo


    def prepare_mesh(self, config_dict, cut_tunnel):
        mesh_name = config_dict["mesh"]

        logging.info(mesh_name)

        if cut_tunnel:
            mesh_name = mesh_name + "_cut"
        # mesh_file = mesh_name + ".msh"
        # mesh_healed = mesh_name + "_healed.msh"
        mesh_healed = mesh_name + ".msh"
        logging.info(mesh_healed)
        # suppose that the mesh was created/copied during preprocess
        print(os.path.join(config_dict["common_files_dir"]))
        print(mesh_healed)

        assert os.path.isfile(os.path.join(config_dict["common_files_dir"], mesh_healed))
        shutil.copyfile(os.path.join(config_dict["common_files_dir"], mesh_healed), mesh_healed)
        return mesh_healed

        # if os.path.isfile(os.path.join(config_dict["common_files_dir"], mesh_healed)):
        #     shutil.copyfile(os.path.join(config_dict["common_files_dir"], mesh_healed), mesh_healed)
        #     return mesh_healed

        # if not os.path.isfile(mesh_healed):
        #     if not os.path.isfile(mesh_file):
        #         self.make_mesh_A(config_dict, mesh_name, mesh_file, cut_tunnel=cut_tunnel)
        #         # self.make_mesh_B(config_dict, mesh_name, mesh_file, cut_tunnel=cut_tunnel)
        #     hm = heal_mesh.HealMesh.read_mesh(mesh_file, node_tol=1e-4)
        #     hm.heal_mesh(gamma_tol=0.01)
        #     hm.stats_to_yaml(mesh_name + "_heal_stats.yaml")
        #     hm.write()
        #     assert hm.healed_mesh_name == mesh_healed
        # return mesh_healed

    # @staticmethod
    # def prepare_hm_input(config_dict):
    #     """
    #     Prepare FieldFE input file for the TH simulation.
    #     :param config_dict: Parsed config.yaml. see key comments there.
    #     """
    #     bc_pressure_csv = 'bc_pressure_tunnel.csv'
    #     if os.path.exists(bc_pressure_csv):
    #         return
    #
    #     end_time = 17
    #     time_step = 1
    #     times = np.arange(0, end_time, time_step)
    #     n_steps = len(times)
    #     times = np.append(times, end_time)
    #
    #     start_val = 300
    #     end_val = 0
    #     val_step = (end_val-start_val)/n_steps
    #     values = np.arange(start_val, end_val, val_step)
    #     values = np.append(values, end_val)
    #
    #     header = "time bc_pressure_tunnel"
    #     fmt = "%g"
    #     list_rows = np.column_stack((times, values))
    #     np.savetxt(bc_pressure_csv, list_rows, fmt=fmt, delimiter=' ', header=header)
    #     # with open(bc_pressure_csv, 'w', newline='') as csv_file:
    #     #     writer = csv.writer(csv_file)
    #     #     writer.writerow(["time", "bc_pressure_tunnel"])
    #     #     for t, v in zip(times, values):
    #     #         writer.writerow([t, v])

        # PREPARE normal traction on tunnel boundary evolving in time






    # @staticmethod
    # def extract_time_series(yaml_stream, regions, extract):
    #     """
    #
    #     :param yaml_stream:
    #     :param regions:
    #     :return: times list, list: for every region the array of value series
    #     """
    #     data = yaml.load(yaml_stream, yaml.CSafeLoader)['data']
    #     times = set()
    #     reg_series = {reg: [] for reg in regions}
    #
    #     for time_data in data:
    #         region = time_data['region']
    #         if region in reg_series:
    #             times.add(time_data['time'])
    #             power_in_time = extract(time_data)
    #             reg_series[region].append(power_in_time)
    #     times = list(times)
    #     times.sort()
    #     series = [np.array(region_series, dtype=float) for region_series in reg_series.values()]
    #     return np.array(times), np.array(series)
    #
    # @staticmethod
    # def extract_th_results(output_dir, out_regions, bc_regions):
    #     with open(os.path.join(output_dir, "energy_balance.yaml"), "r") as f:
    #         power_times, reg_powers = endorse_2Dtest.extract_time_series(f, bc_regions, extract=lambda frame: frame['data'][0])
    #         power_series = -sum(reg_powers)
    #
    #     with open(os.path.join(output_dir, "Heat_AdvectionDiffusion_region_stat.yaml"), "r") as f:
    #         temp_times, reg_temps = endorse_2Dtest.extract_time_series(f, out_regions, extract=lambda frame: frame['average'][0])
    #     with open(os.path.join(output_dir, "water_balance.yaml"), "r") as f:
    #         flux_times, reg_fluxes = endorse_2Dtest.extract_time_series(f, out_regions, extract=lambda frame: frame['data'][0])
    #     sum_flux = sum(reg_fluxes)
    #
    #     reg_temps = reg_temps - endorse_2Dtest.zero_temperature_offset
    #
    #     avg_temp_flux = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux
    #     return avg_temp_flux, power_series

    # @staticmethod
    # def plot_exchanger_evolution(temp_times, avg_temp, power_times, power_series):
    #     year_sec = 60 * 60 * 24 * 365
    #
    #     import matplotlib.pyplot as plt
    #     fig, ax1 = plt.subplots()
    #     temp_color = 'red'
    #     ax1.set_xlabel('time [y]')
    #     ax1.set_ylabel('Temperature [C deg]', color=temp_color)
    #     ax1.plot(temp_times[1:] / year_sec, avg_temp[1:], color=temp_color)
    #     ax1.tick_params(axis='y', labelcolor=temp_color)
    #
    #     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #     pow_color = 'blue'
    #     ax2.set_ylabel('Power [MW]', color=pow_color)  # we already handled the x-label with ax1
    #     ax2.plot(power_times[1:] / year_sec, power_series[1:] / 1e6, color=pow_color)
    #     ax2.tick_params(axis='y', labelcolor=pow_color)
    #
    #     fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #     plt.show()

    def observe_time_plot(self, config_dict, fo: common.FlowOutput):

        pressure_points2collect = config_dict["observe_points"]

        with open(fo.hydro.observe_file.path, "r") as f:
            loaded_yaml = yaml.safe_load(f)
            data = loaded_yaml['data']
            times = np.array([d["time"] for d in data]).transpose()

            values = self.get_from_observe(loaded_yaml, pressure_points2collect, 'pressure_p0')

            fig, ax1 = plt.subplots()
            temp_color = ['red', 'green', 'violet', 'blue']
            ax1.set_xlabel('time [d]')
            ax1.set_ylabel('pressure [m]')
            for i in range(0, len(pressure_points2collect)):
                ax1.plot(times, values[i, 0:], color=temp_color[i], label=pressure_points2collect[i])

            ax1.tick_params(axis='y')
            ax1.legend()

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            # plt.show()
            plt.savefig("observe_pressure.pdf")

    def smooth_ode(self, times, values, smooth_factor):

        pw = scipy.interpolate.CubicSpline(times, values, bc_type='natural')

        p0 = values[0]
        tspan = [times[0], times[-1]]

        p0V0 = np.pi * 0.0025 * 1 * p0
        def ode_func(t, y):
            return y*y/p0V0 * smooth_factor * (pw(t)-y)

        sol = scipy.integrate.solve_ivp(fun=ode_func, t_span=tspan, y0=[p0], t_eval=times)
        return sol.y

