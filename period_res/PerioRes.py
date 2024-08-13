import torch
import numpy as np
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


class PerioRes(nn.Module):
    def __init__(self, input_dim, periods, hidden_dim=10, spectral_radius=(0.8, 0.8), regular=0.9, fusion=False):
        super(PerioRes, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.periods = periods
        self.radius = spectral_radius
        self.fusion = fusion
        self.connectivity = min(1., 10 / hidden_dim)
        self.regular = regular
        self.W_in = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim), requires_grad=False)
        self.W_s = nn.Parameter(torch.from_numpy(self.get_res(0).astype(np.float32)), requires_grad=False)
        # self.W_s = nn.ParameterList(
        #     [nn.Parameter(torch.from_numpy(self.get_res(0).astype(np.float32)), requires_grad=False) for _ in periods])
        self.W_p = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(self.get_res(1).astype(np.float32)), requires_grad=False) for _ in periods])

    def get_res(self, i):
        w = np.random.randn(self.hidden_dim, self.hidden_dim)
        mask = np.random.choice([0, 1], size=(self.hidden_dim, self.hidden_dim),
                                p=[1 - self.connectivity, self.connectivity])
        w = w * mask
        max_lambda = max(abs(np.linalg.eig(w)[0]))
        return self.radius[i] * w / max_lambda

    def forward(self, x):
        batch_size, length, _ = x.size()
        x_transformed = torch.matmul(x, self.W_in)
        regular1 = self.regular * torch.eye(self.hidden_dim).unsqueeze(0).repeat(x.size(0), 1, 1)
        regular2 = self.regular * torch.eye(2 * self.hidden_dim).unsqueeze(0).repeat(x.size(0), 1, 1)
        feature_vectors = []

        for idx, period in enumerate(self.periods):
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            h_history = torch.zeros(batch_size, length, self.hidden_dim, device=x.device)

            if period < length:
                for t in range(period):
                    h_t = x_transformed[:, t, :] + h @ self.W_s
                    h = torch.tanh(h_t)
                    h_history[:, t, :] = h
                for t in range(period, length):
                    h_t = x_transformed[:, t, :] + h @ self.W_s + h_history[:, t - period, :] @ self.W_p[idx]
                    h = torch.tanh(h_t)
                    h_history[:, t, :] = h
                H = torch.cat([h_history[:, period - 1:length - 1, :], h_history[:, :length - period, :]], dim=2)
                X = x[:, period:, :]
                regular = regular2

                # H = h_history[:, :length - 1, :]
                # X = x[:, 1:, :]
                # regular = regular1
            else:
                for t in range(length):
                    h_t = x_transformed[:, t, :] + h @ self.W_p[idx]
                    h = torch.tanh(h_t)
                    h_history[:, t, :] = h

                    H = h_history[:, :length - 1, :]
                    X = x[:, 1:, :]
                    regular = regular1

            # Ridge regression
            Ht = H.transpose(1, 2)  # [batch_size, hidden_num, length]
            HtH = torch.bmm(Ht, H)
            HtX = torch.bmm(Ht, X)
            W_out = torch.linalg.solve(HtH + regular, HtX)

            feature_vectors.append(W_out.flatten(start_dim=1).numpy())
        return np.concatenate(feature_vectors, axis=1)
