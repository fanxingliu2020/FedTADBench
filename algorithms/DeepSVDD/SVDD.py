import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from options import features_dict, args
from algorithms.DeepSVDD.DeepSVDD import BaseNet

class Model_first_stage(BaseNet):
    def __init__(self):
        super(Model_first_stage, self).__init__()

        self.rep_dim = 8

        # Encoder (must match the Deep SVDD network above)
        self.linear1 = nn.Linear(features_dict[args.dataset], 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, self.rep_dim)

        # Decoder
        self.linear4 = nn.Linear(self.rep_dim, 16)
        self.linear5 = nn.Linear(16, 32)
        self.linear6 = nn.Linear(32, features_dict[args.dataset])

        self.mse_loss = nn.MSELoss()

        self.device = torch.device('cuda:0')

    def forward(self, x):
        x_hat = self.linear1(x)
        x_hat = F.relu(x_hat)
        x_hat = self.linear2(x_hat)
        x_hat = F.relu(x_hat)
        h = self.linear3(x_hat)
        x_hat = self.linear4(h)
        x_hat = F.relu(x_hat)
        x_hat = self.linear5(x_hat)
        x_hat = F.relu(x_hat)
        x_hat = self.linear6(x_hat)

        scores = []
        for i in range(x.shape[0]):
            scores.append(self.mse_loss(x[i], x_hat[i]))
        if len(x_hat.size()) == 3:
            x_hat = x_hat[:, -1, :]
        scores = torch.tensor(scores)
        others = {'output': x_hat}

        return h, scores, others

class Model_second_stage(nn.Module):
    def __init__(self):
        super(Model_second_stage, self).__init__()

        # self.device = torch.device('cuda:0')

        self.first_stage_model = None

        self.rep_dim = 8

        # Encoder (must match the Deep SVDD network above)
        self.linear1 = nn.Linear(features_dict[args.dataset], 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, self.rep_dim)

        self.R = torch.tensor(0.0)  # radius R initialized with 0 by default.
        self.nu = 0.1

        self.device = torch.device('cuda:0')

        self.c = None

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def load_first_stage_model(self, first_stage_model):
        net_dict = self.state_dict()
        ae_net_dict = first_stage_model

        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.load_state_dict(net_dict)

    def forward(self, x):
        x_hat = self.linear1(x)
        x_hat = F.relu(x_hat)
        x_hat = self.linear2(x_hat)
        x_hat = F.relu(x_hat)
        h = self.linear3(x_hat)
        return h, h, {}
