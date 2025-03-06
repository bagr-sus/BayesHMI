import os

import numpy as np

from ..samplers.tinyda_flow import TinyDAFlowWrapper
from ..simulation.flow_wrapper import Wrapper
from definitions import ROOT_DIR


new_config_path = os.path.join(ROOT_DIR, "tests", "simulation", "templates", "test_workdir11")
old_config_path = os.path.join(ROOT_DIR, "tests", "simulation", "templates", "test_workdir10")

observe_path = os.path.join(ROOT_DIR, "tests", "measured_data")

print(new_config_path)
print(old_config_path)

# according to new config!
test_params = np.array([
    -16.4340685618576,
    24.8176103991685,
    17.6221730477346,
    16.2134058307626,
    -48.8651125766410,
    33,
    -36.8413614879047,
    1.79175946922806
])

wrapper_new = Wrapper(new_config_path)
wrapper_new.set_observe_path(observe_path)
wrapper_old = Wrapper(old_config_path)
wrapper_old.set_observe_path(observe_path)

wrapper = TinyDAFlowWrapper(wrapper_new)

new_params = wrapper.transform_params(test_params)
old_perms = wrapper.new_to_old_model(*new_params[2:])
old_params = np.concatenate([new_params[0:4], old_perms])

wrapper_new.set_parameters(new_params)
new_observe = wrapper_new.get_observations()

wrapper_old.set_parameters(old_params)
old_observe = wrapper_old.get_observations()

print("New model observe:")
print(new_observe)
print("Old model observe:")
print(old_observe)
