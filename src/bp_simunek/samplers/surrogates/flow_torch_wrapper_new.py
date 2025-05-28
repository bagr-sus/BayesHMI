import yaml
import os
import logging
from pathlib import Path

import numpy as np
import torch


OUTPUT_NORMALIZATION_FACTOR = 275
OUTPUT_LENGTH_PER_BOREHOLE = 26

def load_mlp_from_yaml(yaml_file):

    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)
    class MLP(torch.nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size):
            super().__init__()
            layers = []
            current_size = input_size
            for hidden_size in hidden_sizes:
                layers.append(torch.nn.Linear(current_size, hidden_size))
                layers.append(torch.nn.Tanh())
                current_size = hidden_size
            layers.append(torch.nn.Linear(current_size, output_size))
            self.network = torch.nn.Sequential(*layers)
        def forward(self, x):
            return self.network(x)
    model = MLP(params['input_size'], params['hidden_sizes'], params['output_size'])

    state_dict_path = os.path.join(Path(yaml_file).parent, params['state_dict_path'])

    model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
    model.eval()
    return model



class Wrapper():

    def __init__(self, file_path, boreholes=["H1"]):
        self.model = load_mlp_from_yaml(file_path)
        self.boreholes = boreholes
        self.device = "cpu"

        self.means = np.array([-17.808857, 22.572035, 19.07825, 14.988299, -44.922424, 27.877882, -36.001244, 1.4077234])
        self.stds = np.array([6.0780525, 2.9205055, 1.3847401, 1.4956254, 6.0630407, 24.937956, 0.99940753, 1.7116407])

    def set_parameters(self, params):
        self.params = (np.log(params) - self.means) / self.stds

    def get_observations(self):
        # Convert parameters to tensor
        params_tensor = torch.tensor(self.params, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Get model output
        with torch.no_grad():
            output_normalized = self.model(params_tensor)

        output = np.transpose(output_normalized.cpu().numpy() * OUTPUT_NORMALIZATION_FACTOR)

        # Select data for specified boreholes
        selected_data = np.empty((0, 1))
        for bh in self.boreholes:
            match bh:
                case "V1":
                    i = 0
                case "H1":
                    i = 1
                case _:
                    raise ValueError(f"Unknown borehole: {bh}")
            selected_data = np.append(selected_data, output[OUTPUT_LENGTH_PER_BOREHOLE*i:OUTPUT_LENGTH_PER_BOREHOLE*(i+1)])

        return selected_data, None 