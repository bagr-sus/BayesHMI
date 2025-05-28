import time
import os

import numpy as np
import logging

from bp_simunek.samplers.surrogates.flow_torch_wrapper import Wrapper as WrapperOld
from bp_simunek.samplers.surrogates.flow_torch_wrapper_new import Wrapper as WrapperNew
from bp_simunek.samplers.transforms import new_to_old_model, transform_params


class NNModel():

    def __init__(self, filepath, config, priors, boreholes=["V1"], measured_len=0, nn_type="old"):
        if nn_type == "new":
            self.wrapper = WrapperNew(os.path.join(config["work_dir"], filepath), boreholes=boreholes)
        else:
            self.wrapper = WrapperOld(os.path.join(config["work_dir"], filepath), boreholes=boreholes)
        self.measured_len = measured_len
        self.priors = priors
        self.grad = np.zeros((len(priors), 1))


    def __call__(self, params):
        # log model info
        #logging.info("Model level: %i", level)
        
        # transform parameters via info from priors
        logging.info("Raw input:")
        logging.info(params)

        trans_params = np.array(transform_params(params, self.priors))
        #logging.info("Transformed input:")
        #logging.info(trans_params)

        old_perms = new_to_old_model(*trans_params[2:])
        #logging.info("Old perms:")
        #logging.info(old_perms)

        final_params = np.concatenate([trans_params[0:4], old_perms])
        #logging.info("Final params:")
        #logging.info(final_params)

        # Start time measurement of model
        start = time.time()

        # Pass params to model
        self.wrapper.set_parameters(final_params)

        data, gradient = self.wrapper.get_observations()

        if gradient is not None:
            self.grad = np.squeeze(gradient.numpy())
            logging.info("Gradient:")
            logging.info(self.grad)

        # Dummy value to force sampler to reject sample
        if data is None:
            data = np.multiply(1e8, np.ones(self.measured_len))

        if np.any(np.isnan(data)):
            logging.warning("NaN values in model output, sample will be rejected.")
            data = np.multiply(1e8, np.ones(self.measured_len))

        if np.any(np.isinf(data)):
            logging.warning("Inf values in model output, sample will be rejected.")
            data = np.multiply(1e8, np.ones(self.measured_len))

        if np.issubdtype(data.dtype, np.str_):
            logging.warning("String values in model output, sample will be rejected.")
            data = np.multiply(1e8, np.ones(self.measured_len))

        # Format params for logging purposes
        params_formatted = ",".join([str(param) for param in params.tolist()])

        # End time measurement of model
        end = time.time()
        elapsed = end - start
        # Write time measurement
        # Await confirmation of logging
        elapsed_formatted = f"{elapsed:.2f}"
        logstring = ",".join([elapsed_formatted, params_formatted]) + "\n"
        #logging.info(logstring)
        #self.logger_ref.write_to_file.remote(logstring, "observe_times")

        #if self.config["conductivity_observe_points"]:
        #    num = len(self.config["conductivity_observe_points"])
        #    data = data[:-num]
        #logging.info(data)
        #logging.info(data.shape)
        return data

    def gradient(self, params, sensitivity):

        return self.grad