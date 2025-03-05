import logging
import traceback
import os

import numpy as np
import cma
import scipy.stats as sps

from ..simulation.measured_data import MeasuredData
from ..simulation.flow_wrapper import Wrapper as FlowWrapper

class PyCMAFlowWrapper:

    def __init__(self, flow_wrapper: FlowWrapper):
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

        # setup priors from config of flow wrapper
        self.setup_priors(self.config)

        # choose which boreholes to use
        boreholes = ["H1"]
        # choose which borehole conductivities to use, empty list means none
        cond_boreholes = []
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
        self.measured_len = len(values)
        self.times = times

        self.flow_wrapper.sim._config["mesh"] = self.config["models"][0]["file"]

    def optimize(self, maxevals=1600) -> cma.CMAEvolutionStrategy:
        #os.chdir(self.config["work_dir"])

        initial_means = np.array([param["dist"].mean() for param in self.priors])
        initial_stds = np.array([np.sqrt(param["dist"].std()) for param in self.priors])

        es = cma.CMAEvolutionStrategy(initial_means, 1,
                                    {
                                          "maxfevals": maxevals,
                                          "CMA_stds": initial_stds,
                                          'verb_log': 1
                                    })
        es.optimize(self.model_with_error)

        return es

    def model(self, params):

        trans_params = np.array(self.transform_params(params))

        self.flow_wrapper.set_parameters(trans_params)

        try:
            # Get model output
            _, data = self.flow_wrapper.get_observations()
        except Exception:
            logging.error("Couldn't get observation from wrapper\nSample will be rejected.")
            logging.error(traceback.format_exc())
            data = np.multiply(1e8, np.ones(self.measured_len))

        # Dummy value to force sampler to reject sample
        if data is None:
            data = np.multiply(1e8, np.ones(self.measured_len))

        self.flow_wrapper.sim.clean_sample_dir(self.config)

        logging.warning("Model output:")
        logging.warning(data)
        return data

    def error(self, model_output, observed):
        return np.linalg.norm(model_output - observed)
    
    def model_with_error(self, params):
        model_output = self.model(params)
        return self.error(model_output, self.observed)
    
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

    def transform_params(self, params):
        trans_params = []
        for param, prior in zip(params, self.priors):
            match prior["type"]:
                case "lognorm":
                    trans_param = np.exp(param)
                case "truncnorm":
                    a, b, mu, sigma = prior["params"]
                    lower_bound = (a - mu) / sigma
                    upper_bound = (b - mu) / sigma
                    phi_a = sps.norm.cdf(lower_bound)
                    phi_b = sps.norm.cdf(upper_bound)
                    phi_param = sps.norm.cdf(param, loc=mu, scale=sigma)
                    trans_param = sps.norm.ppf((phi_b - phi_a)*phi_param + phi_a)*sigma + mu

            trans_params.append(trans_param)
        return trans_params

    def save_results_to_file(self, es, file_path):
        best_params = es.result.xbest
        estimated_means = best_params[:len(self.observed)]
        estimated_stds = np.abs(best_params[len(self.observed):])
        covariance_matrix = es.result.C
        with open(file_path, "w") as f:
            f.write(f"Estimated Means: {estimated_means}\n")
            f.write(f"Estimated STDs: {estimated_stds}\n")
            f.write(f"Covariance Matrix: {covariance_matrix}\n")
