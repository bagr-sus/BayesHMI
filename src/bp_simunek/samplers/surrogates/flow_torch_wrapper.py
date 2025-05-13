import logging
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
device = "cpu"


class FlowSurrogate(nn.Module):
    def __init__(self, hidden_size: int = 200, sequence_length: int = 26,
                 input_size: int = 8, preprocessor_multiplier: int = 5):
        super(FlowSurrogate, self).__init__()

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.input_size = input_size

        self.activation = nn.Tanh()

        self.data_processor = nn.ModuleList(
            [nn.Linear(i * 4, sequence_length * 4 * preprocessor_multiplier, device=device) for i in range(26)])
        self.data_processor1 = nn.Linear(
            sequence_length * preprocessor_multiplier * 4,
            sequence_length * preprocessor_multiplier * 4)
        self.data_processor2 = nn.Linear(sequence_length * preprocessor_multiplier * 4, sequence_length * 4)

        self.L1 = nn.Linear(input_size, hidden_size)
        self.L1_retention = nn.Linear(sequence_length * 4, hidden_size)

        self.L2 = nn.Linear(hidden_size, hidden_size)
        self.L2_retention = nn.Linear(sequence_length * 4, hidden_size)

        self.L3 = nn.Linear(hidden_size, hidden_size)
        self.L3_retention = nn.Linear(sequence_length * 4, hidden_size)

        self.L4 = nn.Linear(hidden_size, 4)
        self.L4_retention = nn.Linear(sequence_length * 4, 4)

    def forward(self, x):
        # Create a tensor of ones with the same device and dtype as x, avoiding direct multiplication for initialization
        data = torch.ones((x.size(0), self.sequence_length * 4), dtype=torch.float32, device=x.device)
        # Concatenate the ones and x at the beginning to avoid repeated concatenation
        y = torch.ones((x.size(0), self.sequence_length * 4), dtype=torch.float32, device=x.device)

        # Assuming new_encoder can work with partial views or slices of y for efficiency
        for i in range(1, 26):
            data1 = self.activation(self.data_processor[i](data[:, :(i * 4)].clone()))
            data1 = self.activation(self.data_processor1(data1))
            data1 = self.activation(self.data_processor2(data1))
            x1 = self.activation(self.L1(x) + self.L1_retention(data1))
            x1 = self.activation(self.L2(x1) + self.L2_retention(data1))
            x1 = self.activation(self.L3(x1) + self.L3_retention(data1))
            y[:, (i * 4):((i + 1) * 4)] = self.L4(x1) + self.L4_retention(data1)
            data[:, (i * 4):((i + 1) * 4)] = y[:, (i * 4):((i + 1) * 4)]

        return y


class Wrapper:
    def __init__(self, file_path, device='cpu', boreholes=["H1"]):

        # get path of this file:
        # path = os.path.dirname(__file__)

        # self.model = FlowSurrogate(hidden_size=600)
        # file_path = os.path.join(path, "model_TSX_small.pth")
        # self.model = FlowSurrogate(hidden_size=2000)
        # file_path = os.path.join(path, "model_TSX_large.pth")

        self.device = device
        nn_data = torch.load(file_path, map_location=torch.device('cpu'))
        hidden_size = nn_data["L1.bias"].shape[0]
        self.model = FlowSurrogate(hidden_size=hidden_size)
        self.model.load_state_dict(nn_data)
        self.model.to(self.device)
        self.means = np.array([-16, 26, 17, 16, -48, -41, -14, -16], dtype=np.float32)[None, :]
        self.std = np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)[None, :]
        self.parameters = self.means.copy()
        self.boreholes = boreholes

    def set_parameters(self, parameters: npt.NDArray) -> None:
        parameters = parameters.reshape(-1, 8)
        #logging.info("params before set")
        #logging.info(parameters)
        self.parameters = (np.log(parameters) - self.means) / self.std
        #logging.info("params after set")
        #logging.info(self.parameters)

    def get_observations(self) -> npt.NDArray:
        #with torch.no_grad():
        parameters = torch.tensor(self.parameters, dtype=torch.float32, device=self.device, requires_grad=True)
        y = self.model(parameters)
        loss = y.sum()
        loss.backward()

        gradient = parameters.grad

        y = y.cpu().detach().numpy() * 275.0
        #return 1, np.concatenate([y[:, i::4] for i in range(4)], axis=1)
        data = [y[:, i::4] for i in range(4)]
        #logging.info(f"Surrogate data: {data}")
        #logging.info(data)
        select_data = np.empty((0, 1))
        for bh in self.boreholes:
            match bh:
                case "V1":
                    i = 0
                case "V2":
                    i = 1
                case "H1":
                    i = 2
                case "H2":
                    i = 3
            select_data = np.append(select_data, data[i].flatten())

        return select_data.flatten(), gradient
