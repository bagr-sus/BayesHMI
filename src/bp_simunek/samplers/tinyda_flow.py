import os
import time
import traceback
import logging
from functools import partial
from enum import Enum
import re
from copy import deepcopy
import sys

import numpy as np
import scipy.stats as sps
import arviz as az
import ray
import tinyDA as tda
from tinyDA.sampler import ray_is_available

from ..simulation.measured_data import MeasuredData
from ..common.memoize import File
from ..samplers.surrogates.flow_torch_wrapper import Wrapper as NNWrapper
from ..simulation.flow_wrapper import Wrapper as FlowWrapper
from ..samplers.nn_model import NNModel
from ..samplers.transforms import new_to_old_model, transform_params
from ..samplers.idata_tools import thin_inference_data


NUMBER_OF_CHAINS_DEFAULT = 1
FORCE_SEQUENTIAL_DEFAULT = False
PROPOSAL_SCALING_DEFAULT = 0.2
GAMMA_DEFAULT = 1.01
ADAPTIVITY_DEFAULT = False
ADAPTIVITY_PERIOD_DEFAULT = 10
NOISE_COV_DEFAULT = 20
IS_PARALLEL_DEFAULT = False
SAMPLE_COUNT_DEFAULT = 10
TUNE_COUNT_DEFAULT = 1
MLDA_DEFAULT = False
MLDA_LEVELS_DEFAULT = 2
PROPOSAL_DEFAULT = tda.GaussianRandomWalk
M0_DEFAULT = 5
DELTA_DEFAULT = 1
NCR_DEFAULT = 1
B_STAR_DEFAULT = 1e-6
ARCHIVE_LIMIT_DEFAULT = 0
ADAPTIVE_LIKELIHOOD_DEFAULT = False

logging.basicConfig(
    level=logging.DEBUG,        # Capture all messages at DEBUG level and above
    stream=sys.stdout
)

@ray.remote
class DataLogger():
    """Ray object to log various data during the sampling process.
    """
    def __init__(self, files_dict: dict) -> None:
        """Setup all files to be written and clean up existing files.

        Args:
            files_dict (dict): Dictionary with keys to identify the
             various files and values being the path to the files.
        """
        #for file in files_dict.values():
        #    with open(file, "w", encoding="utf8") as file:
        #        file.writeline("")

        self.files_dict = files_dict

    def get_file_keys(self) -> list:
        """Get all avaialable file keys for logging.

        Returns:
            list: List of all available keys.
        """
        return list(self.files_dict.keys())

    def write_to_file(self, message: str, file_key: str) -> None:
        """Write to a certain logging file.

        Args:
            message (str): Message to write.
            file_key (str): File key corresponding to the
              file to be written into.
        """
        # If specified key is unknown, return witout writing and log error
        if file_key not in self.files_dict.keys():
            logging.error("No logging file found with key %s", file_key)
            return

        # Create file or append to existing contents
        try:
            with open(self.files_dict[file_key], "a+", encoding="utf8") as file:
                file.write(message)
        except Exception:
            logging.error("Unable to write to file")
            logging.error(traceback.format_exc())

class TinyDAFlowWrapper():
    """
    Wrapper combining a flow123 instance into a tinyDA sampler
    """

    def __init__(self, flow_wrapper):
        # flow wrapper
        self.flow_wrapper = flow_wrapper
        # measured data loader
        self.observed_data = MeasuredData(self.flow_wrapper.sim._config)
        self.observed_data.initialize()
        # better reference to config object
        self.config = self.flow_wrapper.sim._config
        # length of measured data
        self.measured_len = -1
        # time steps of simulation
        self.observe_times = []
        # reference to shared object for logging
        self.logger_ref = None
        # check for sampler config or set default params
        sampler_config_key = "sampler_parameters"
        if sampler_config_key not in self.config:
            logging.warning("Missing sampler parameters, using default values for all")
            self.set_default_sampler_params()
        else:
            try:
                self.load_sampler_params(self.config[sampler_config_key])
            except Exception:
                logging.error("Failed to load sampler params from config, using default values for all")
                logging.error(traceback.format_exc())
                self.set_default_sampler_params()

        # setup priors from config of flow wrapper
        self.setup_priors(self.config)

        # start ray before tinyda
        if not ray.is_initialized():
            ray.init()

    def set_default_sampler_params(self):
        self.is_parallel = IS_PARALLEL_DEFAULT
        self.number_of_chains = NUMBER_OF_CHAINS_DEFAULT
        self.force_sequential = FORCE_SEQUENTIAL_DEFAULT
        self.proposal_scaling = PROPOSAL_SCALING_DEFAULT
        self.gamma = GAMMA_DEFAULT
        self.adaptive = ADAPTIVITY_DEFAULT
        self.adaptivity_period = ADAPTIVITY_PERIOD_DEFAULT
        self.noise_cov = NOISE_COV_DEFAULT
        self.sample_count = SAMPLE_COUNT_DEFAULT
        self.tune_count = TUNE_COUNT_DEFAULT
        self.mlda = MLDA_DEFAULT
        self.levels = MLDA_LEVELS_DEFAULT
        self.proposal = PROPOSAL_DEFAULT
        self.m0 = M0_DEFAULT
        self.delta = DELTA_DEFAULT
        self.ncr = NCR_DEFAULT
        self.b_star = B_STAR_DEFAULT
        self.adaptive_likelihood = ADAPTIVE_LIKELIHOOD_DEFAULT

    def load_sampler_params(self, params):
        # specify number of chains
        chains_key = "chain_count"
        if chains_key not in params:
            logging.warning("Missing number of chains, defaulting to %d", NUMBER_OF_CHAINS_DEFAULT)
            self.number_of_chains = NUMBER_OF_CHAINS_DEFAULT
        else:
            # TODO check if param is actually a valid number, not a string etc.
            self.number_of_chains = params[chains_key]

        # specify whether to force sequential sampling
        force_seq_key = "force_sequential"
        if force_seq_key not in params:
            logging.warning("Force sequential not specified, defaulting to %s", str(FORCE_SEQUENTIAL_DEFAULT))
            self.force_sequential = FORCE_SEQUENTIAL_DEFAULT
        else:
            # TODO check if value is boolean
            self.force_sequential = params[force_seq_key]

        # get number of samples to sample
        sample_count_key = "sample_count"
        if sample_count_key not in params:
            logging.warning("Number of samples not specified, defaulting to %d", SAMPLE_COUNT_DEFAULT)
            self.sample_count = SAMPLE_COUNT_DEFAULT
        else:
            self.sample_count = params[sample_count_key]

        # get length of tune
        tune_count_key = "tune_count"
        if tune_count_key not in params:
            logging.warning("Length of tune not specified, defaulting to %d", TUNE_COUNT_DEFAULT)
            self.tune_count = TUNE_COUNT_DEFAULT
        else:
            self.tune_count = params[tune_count_key]

        # check if using MLDA
        mlda_key = "mlda"
        if mlda_key not in params:
            logging.warning("MLDA not specified, defaulting to %d", MLDA_DEFAULT)
            self.mlda = MLDA_DEFAULT
        else:
            self.mlda = params[mlda_key]

        if self.mlda:
             # check for number of mlda levels
            mlda_levels_key = "mlda_levels"
            if mlda_levels_key not in params:
                logging.warning("Number of MLDA levels not specified, defaulting to %d", MLDA_LEVELS_DEFAULT)
                self.mlda_levels = MLDA_LEVELS_DEFAULT
            else:
                self.mlda_levels = params[mlda_levels_key]

        # proposal selection
        proposal_key = "proposal"
        if proposal_key in params:
            match params[proposal_key]:
                case "DREAM":
                    self.proposal = tda.DREAM
                    logging.info("Using DREAM proposal")
                case "DREAMZ":
                    self.proposal = tda.DREAMZ
                    logging.info("Using DREAMZ proposal")
                case "Metropolis":
                    self.proposal = tda.GaussianRandomWalk
                    logging.info("Using GRW proposal")
                case "MALA":
                    self.proposal = tda.MALA
                    logging.info("Using MALA proposal")
                case "KernelMALA":
                    self.proposal = tda.KernelMALA
                    logging.info("Using Kernel MALA proposal")
                case "IS":
                    self.proposal = tda.IndependenceSampler
                    logging.info("Using Independence Sampler proposal")
                case _:
                    self.proposal = tda.GaussianRandomWalk
                    logging.warning(f"Incorrect sampler specified, defaulting to {PROPOSAL_DEFAULT}")
        else:
            logging.warning(f"No sampler specified, defaulting to {PROPOSAL_DEFAULT}")
            self.sampler = PROPOSAL_DEFAULT

        if self.proposal in [tda.GaussianRandomWalk, tda.MALA, tda.KernelMALA]:
              # check if proposal params are specified
            proposal_scaling_key = "proposal_scaling"
            if proposal_scaling_key in params:
                self.scaling = params[proposal_scaling_key]
            else:
                self.scaling = PROPOSAL_SCALING_DEFAULT
                logging.warning("Unspecified proposal scaling, defaulting to %f", PROPOSAL_SCALING_DEFAULT)


        if self.proposal in [tda.DREAMZ, tda.DREAM]:
            m0_key = "m0"
            if m0_key in params:
                self.m0 = params[m0_key]
            else:
                self.m0 = M0_DEFAULT
                logging.warning("m0 not specified, defaulting to %d", M0_DEFAULT)

            delta_key = "delta"
            if delta_key in params:
                self.delta = params[delta_key]
            else:
                self.delta = DELTA_DEFAULT
                logging.warning("delta not specified, defaulting to %d", DELTA_DEFAULT)

            ncr_key = "ncr"
            if ncr_key in params:
                self.ncr = params[ncr_key]
            else:
                self.ncr = NCR_DEFAULT
                logging.warning("ncr not specified, defaulting to %d", NCR_DEFAULT)

            b_star_key = "b_star"
            if b_star_key in params:
                self.b_star = params[b_star_key]
            else:
                self.b_star = B_STAR_DEFAULT
                logging.warning("b_star not specified, defaulting to %f", B_STAR_DEFAULT)

            archive_limit_key = "archive_limit"
            if archive_limit_key in params:
                self.archive_limit = params[archive_limit_key]
            else:
                self.archive_limit = ARCHIVE_LIMIT_DEFAULT
                logging.warning("archive limit not specified, defaulting to %d", ARCHIVE_LIMIT_DEFAULT)

        # adaptive proposal params
        proposal_adaptive_key = "proposal_adaptive"
        if proposal_adaptive_key in params:
            self.adaptive = params[proposal_adaptive_key]

            global_scaling_key = "proposal_gamma"
            if global_scaling_key not in params:
                logging.warning("Unknown proposal gamma, defaulting to %f", GAMMA_DEFAULT)
                self.gamma = GAMMA_DEFAULT
            else:
                self.gamma = params[global_scaling_key]

            adaptive_period_key = "proposal_adaptivity_period"
            if adaptive_period_key not in params:
                logging.warning("Unknown adaptivity period, defaulting to %d", ADAPTIVITY_PERIOD_DEFAULT)
                self.adaptivity_period = ADAPTIVITY_PERIOD_DEFAULT
            else:
                self.adaptivity_period = params[adaptive_period_key]

        else:
            logging.warning("Unspecified whether to adapt, defaulting to %s", str(ADAPTIVITY_DEFAULT))
            self.adaptive = ADAPTIVITY_DEFAULT

        # check for noise cov
        noise_cov_key = "noise_cov"
        if noise_cov_key not in params:
            logging.info("Noise covariance unspecified, defaulting to %f", NOISE_COV_DEFAULT)
            self.noise_cov = NOISE_COV_DEFAULT
        else:
            self.noise_cov = params[noise_cov_key]

        # check for adaptive likelihood
        adaptive_likelihood_key = "adaptive_likelihood"
        if adaptive_likelihood_key not in params:
            logging.info("Adaptive likelihood unspecified, defaulting to %s", str(ADAPTIVE_LIKELIHOOD_DEFAULT))
            self.adaptive_likelihood = ADAPTIVE_LIKELIHOOD_DEFAULT
        else:
            self.adaptive_likelihood = params[adaptive_likelihood_key]


    def create_proposal_matrix(self):
        dists = [prior["dist"] for prior in self.priors]
        cov_vector = np.empty(len(dists))
        for idx, prior in enumerate(dists):
            if hasattr(prior, "std"):
                cov_vector[idx] = np.power(prior.std(), 2)
            else:
                # add support for uniform and other dists that dont have std attrib
                raise Exception("Unsupported distribution, no 'std' attribute.")
        return np.multiply(np.eye(len(cov_vector)), cov_vector)

    def sample(self) -> list:
        # check whether parallel sampling or not
        self.is_parallel = self.number_of_chains > 1 and ray_is_available and not self.force_sequential

        # setup logging
        # force the logger to exist at the head node
        nodes = ray.nodes()
        head_node = [node for node in nodes if node["NodeManagerAddress"] == ray._private.services.get_node_ip_address()][0]
        head_node_id = head_node["NodeID"]

        logging_files = {
            "observe_times": os.path.join(self.flow_wrapper.sim._config["work_dir"], "observe_times.txt"),
            "chain_delay": os.path.join(self.flow_wrapper.sim._config["work_dir"], "chain_delay.txt"),
            "observe_fails": os.path.join(self.flow_wrapper.sim._config["work_dir"], "observe_fails.txt"),
            "proposal_logs": os.path.join(self.flow_wrapper.sim._config["work_dir"], "proposal_logs.csv")
        }
        # scheduling strategy to add node affinity for head node
        self.logger_ref = DataLogger.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=head_node_id,
                soft=False  # hard constraint
            )
        ).remote(logging_files)
        logging.info("Using following logger files:")
        logging.info(logging_files)

        # setup observed data
        # choose which boreholes to use
        boreholes = self.config["observe_points"]
        # choose which borehole conductivities to use, empty list means none
        cond_boreholes = self.config["conductivity_observe_points"]
        # get actual values and choose synthetic/real data
        if "synthetic_data" in self.config:
            times, values = self.observed_data.generate_synthetic_samples(boreholes, cond_boreholes)
        else:
            times, values = self.observed_data.generate_measured_samples(boreholes, cond_boreholes)
        logging.info("Loading observed values:")
        logging.info(values)
        logging.info("At times:")
        logging.info(times)
        self.observed = values
        self.times = times


        # setup loglike
        #logging.info("Using following noise covariance matrix")
        #logging.info(noise_cov)
        self.config["observed"] = self.observed
        self.measured_len = len(values)
        #self.loglike_object = tda.GaussianLogLike(np.array(self.observed), self.cov)

        # setup logging for proposals
        # format: input params, output pressures for each borehole
        proposal_log_header = [param["name"] for param in self.config["parameters"]]
        proposal_log_header.extend([f"{borehole}_{time_point}" for borehole in boreholes for time_point in np.arange(len(self.times))])
        self.logger_ref.write_to_file.remote(",".join(proposal_log_header) + "\n", "proposal_logs")

        # combine into posterior
        posteriors = []
        subchain_lengths = []

        model_count = len(self.config["models"])

        for level, model in enumerate(self.config["models"]):
            logging.info("Model level: %i", level)
            logging.info("Model name: %s", model["name"])
            logging.info("Model type: %s", model["type"])

            noise_cov = np.multiply(self.noise_cov, np.eye(len(values)))

            # if alternate noise_cov is specified
            if "noise_cov" in model:
                noise_cov = np.multiply(model["noise_cov"], np.eye(len(values)))

            subchain_length = 1
            if "subchain_length" in model:
                subchain_length = model["subchain_length"]

            subchain_lengths.append(subchain_length)


            # if using fine model, use non-adaptive loglike
            if level == len(self.config["models"]) - 1:
                loglike = tda.GaussianLogLike(np.array(self.observed), np.multiply(noise_cov, np.eye(len(values))))
            else:
                loglike = tda.AdaptiveGaussianLogLike(np.array(self.observed), np.multiply(noise_cov, np.eye(len(values))))
            logging.info("Using following noise covariance matrix")
            logging.info(noise_cov)

            if model["type"] == "flow":
                wrapper = deepcopy(self.flow_wrapper)
                wrapper.sim._config["mesh"] = model["file"]
                forward_model = partial(self.flow_model, level=level, wrapper=wrapper)
            elif model["type"] == "nn" or model["type"] == "nn2":
                if model["type"] == "nn2":
                    nn_type = "new"
                else:
                    nn_type = "old"
                
                forward_model = NNModel(model["file"], self.config, self.priors, boreholes, self.measured_len, nn_type=nn_type)

            posterior_level = tda.Posterior(self.prior, loglike, forward_model)
            posteriors.append(posterior_level)

        # remove last subchain length, as it is not needed
        subchain_lengths = subchain_lengths[:-1]

        if len(subchain_lengths) == 1:
            subchain_lengths = subchain_lengths[0]

        # setup proposal covariance matrix (for random gaussian walk & adaptive metropolis)
        proposal_cov = self.create_proposal_matrix()

        # setup proposal
        if self.proposal == tda.GaussianRandomWalk:
            logging.info("Using GRW")
            proposal = tda.GaussianRandomWalk(proposal_cov, self.scaling, self.adaptive, self.gamma, self.adaptivity_period)
        elif self.proposal == tda.DREAMZ:
            logging.info("Using DREAMZ")
            proposal = tda.DREAMZ(self.m0, self.delta, nCR=self.ncr, adaptive=self.adaptive, b_star=self.b_star, archive_limit=self.archive_limit)
            logging.info(proposal.b_star)
        elif self.proposal == tda.DREAM:
            logging.info("Using DREAM")
            proposal = tda.DREAM(self.m0, self.delta, nCR=self.ncr, adaptive=self.adaptive, b_star=self.b_star, archive_limit=self.archive_limit)
        elif self.proposal == tda.MALA:
            logging.info("Using MALA")
            proposal = tda.MALA(self.scaling, self.adaptive, self.gamma, self.adaptivity_period)
        elif self.proposal == tda.KernelMALA:
            logging.info("Using Kernel MALA")
            proposal = tda.KernelMALA(M=5000, t0=2000, scaling=self.scaling, adaptive=self.adaptive, gamma=self.gamma, period=self.adaptivity_period)
        elif self.proposal == tda.IndependenceSampler:
            logging.info("Using Independence Sampler")
            proposal = tda.IndependenceSampler(self.prior)

        initial_parameters = [posteriors[0].prior.rvs() for i in range(self.number_of_chains)]
        error_model = None
        if self.adaptive_likelihood:
            error_model = "state-independent"

        logging.info(error_model)

        # sampling process
        samples = tda.sample(
            posteriors=posteriors,
            proposal=proposal,
            iterations=self.sample_count,
            n_chains=self.number_of_chains,
            initial_parameters=initial_parameters,
            force_sequential=self.force_sequential,
            logger_ref=None,
            adaptive_error_model=error_model,
            subchain_length=subchain_lengths)

        # check and save samples
        parameter_names = [prior["name"] for prior in self.priors]

        levels = [""]
        if model_count == 2:
            levels = ["fine", "coarse"]
        elif model_count > 2:
            levels = [str(i) for i in range(model_count - 1, -1, -1)]

        logging.info("Levels: %s", levels)

        # convert samples to inference data
        # one inference data object per level
        # ordered by levels descending
        idatas = []
        current_subchain_length = 1
        for level in levels:
            idata = tda.to_inference_data(chain=samples, parameter_names=parameter_names, burnin=self.tune_count, level=level)
            # add prior info to idata
            for idx, param in enumerate(idata["posterior"]):
                prior = self.priors[idx]
                bounds = prior["params"]
                match prior["type"]:
                    case "lognorm":
                        mean, std = bounds
                    case "truncnorm":
                        _, _, mean, std = bounds

                idata["posterior"][param].attrs["prior_mean"] = mean
                idata["posterior"][param].attrs["prior_std"] = std

            # add observed data to idata
            idata["sample_stats"].attrs["observed"] = self.observed

            # add observed times to idata
            idata["sample_stats"].attrs["times"] = self.times

            # trim idata to match dims of fine level
            idata = thin_inference_data(idata, current_subchain_length)
            idatas.append(idata)

            # change current subchain length to match next level
            if isinstance(subchain_lengths, list):
                next_level = int(level) - 1
                if next_level >= 0:
                    current_subchain_length = current_subchain_length * subchain_lengths[next_level]
            else:
                current_subchain_length = subchain_lengths

        # return data in reverse order, so that the fine model is last
        idatas.reverse()
        return idatas

    def setup_priors(self, config):
        """
        Prior setup for sampling. All dists are interpreted as normal distributions
        and postprocessed into the proper distribution in the forward model.
        Additional info is saved (type of dist, name) so that forward model knows
        how to transform the individual parameters.
        """
        priors = []
        for param in config["parameters"]:
            prior_name = param["name"]
            bounds = param["bounds"]
            prior_type = param["type"]
            match prior_type:
                case "lognorm":
                    mu, sigma = bounds
                    prior = sps.norm(loc=mu, scale=sigma)
                    logging.info("Prior lognorm, mu=%s, std=%s", prior.mean(), prior.std())
                case "truncnorm":
                    a, b, mu, sigma = bounds
                    prior = sps.norm(loc=mu, scale=sigma)
                    logging.info("Prior truncated norm, a=%s, b=%s, mean=%s, std=%s", a, b, prior.mean(), prior.std())
            priors.append({
                "name": prior_name,
                "type": prior_type,
                "dist": prior,
                "params": bounds
            })

        self.priors = priors
        #self.prior = tda.distributions.JointPrior([prior["dist"] for prior in priors])
        self.prior = sps.multivariate_normal(
            mean = [prior["dist"].mean() for prior in priors],
            cov = np.multiply(
                np.power([prior["dist"].std() for prior in priors], 2),
                np.eye(len(priors))
            ))
        logging.info(self.prior)

    def flow_model(self, params, level, wrapper):
        # log model info
        logging.info("Model level: %i", level)
        
        # transform parameters via info from priors
        logging.info("Raw input:")
        logging.info(params)

        trans_params = np.array(transform_params(params, self.priors))

        # Start time measurement of model
        start = time.time()

        # Pass params to model
        logging.info("Transformed input:")
        logging.info(trans_params)
        wrapper.set_parameters(trans_params)

        try:
            # Get model output
            _, data = wrapper.get_observations()
        except Exception:
            logging.error("Couldn't get observation from wrapper\nSample will be rejected.")
            logging.error(traceback.format_exc())
            data = np.multiply(1e8, np.ones(self.measured_len))

        # Dummy value to force sampler to reject sample
        if data is None:
            data = np.multiply(1e8, np.ones(self.measured_len))

        # Format params for logging purposes
        params_formatted = ",".join(["{:.3f}".format(param) for param in params.tolist()])

        # Get additional data from stdout and stderr of flow
        pattern = r"HM Iteration.*\n"
        param_string = ""
        try:
            with open(wrapper.sim.stdout_path, "r", encoding="utf8") as stdout:
                lines = "".join(stdout.readlines())
                matches = re.findall(pattern, lines)
                iterations = [int(match.split(" ")[2]) for match in matches]
                # example of last line output, split by spaces
                # ['HM', 'Iteration', '3', 'abs.', 'difference:', '8.52479e-05', '', 'rel.', 'difference:', '3.11032e-09\n']
                iterations += [0]
                max_iterations = []
                for idx in np.arange(len(iterations) - 1):
                    # if we find a drop - new time step
                    if iterations[idx] >= iterations[idx + 1]:
                        max_iterations.append(iterations[idx])
                total_max = np.max(max_iterations)
                total_mean = np.mean(max_iterations)

                param_string = ",".join([f"{total_max:.1f}", f"{total_mean:.1f}"])
        except Exception:
            logging.error("Failed to log additional data from flow's output")
            logging.error(traceback.format_exc())
            param_string = ",".join([str(-1), str(-1)])
            try:
                wrapper.sim.copy_sample_dir()
            except Exception:
                logging.error("Failed to copy sample dir")
                logging.error(traceback.format_exc())
            self.logger_ref.write_to_file.remote(params_formatted + "\n", "observe_fails")

        # Clean flow output dir
        wrapper.sim.clean_sample_dir(self.config)

        # End time measurement of model
        end = time.time()
        elapsed = end - start
        # Write time measurement
        # Await confirmation of logging
        elapsed_formatted = f"{elapsed:.2f}"
        logstring = ",".join([elapsed_formatted, param_string, params_formatted]) + "\n"
        #logging.info(logstring)
        self.logger_ref.write_to_file.remote(logstring, "observe_times")

        # log proposal
        observe_formatted = ",".join(["{:.3f}".format(value) for value in data.tolist()])

        if not np.any(data > 1e6):
            proposal_formatted = ",".join([params_formatted, observe_formatted, "\n"])
            self.logger_ref.write_to_file.remote(proposal_formatted, "proposal_logs")

        #if self.config["conductivity_observe_points"]:
        #    num = len(self.config["conductivity_observe_points"])
        #    data = data[:-num]

        #logging.warning("Model output:")
        #logging.warning(data)


        return data
