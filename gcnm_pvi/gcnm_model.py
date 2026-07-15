# gcnm_model.py
"""GCN learning components (GCNBlock, training loop, LM feature updates)."""

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

torch.set_default_dtype(torch.float64)


def initializeDataset(sigma_array, V_array, edge_index, initial_sigma=1.0):
    n_samples, n_elements = sigma_array.shape
    data_true = torch.tensor(sigma_array, dtype=torch.float64)
    data_in = torch.full(
        (n_samples, n_elements), float(initial_sigma), dtype=torch.float64
    )
    data_V = torch.tensor(V_array, dtype=torch.float64)

    dataset = []
    for i in range(n_samples):
        dataset.append(
            Data(
                edge_index=edge_index,
                x=data_in[i, :].unsqueeze(dim=1),
                y=data_true[i, :].unsqueeze(dim=1),
                V=data_V[i, :].unsqueeze(dim=1),
            )
        )
    return dataset


def computeLMUpdates(dataset, physics, lambda_lm=0.1, hyper_pvi=0.0, differential=False):
    for i in range(len(dataset)):
        data = dataset[i]
        sigma_current = data.x[:, 0]
        vmeas = data.V.squeeze().cpu().numpy()

        if differential and hasattr(physics, "lm_update_differential"):
            if physics.sigma_init is None:
                physics.calibrate(vmeas)
            delta, _ = physics.lm_update_differential(
                vmeas, lambda_lm=lambda_lm, hyper_pvi=hyper_pvi
            )
        else:
            delta, _ = physics.lm_update(
                sigma_current.cpu().numpy(),
                vmeas,
                lambda_lm=lambda_lm,
                hyper_pvi=hyper_pvi,
            )

        delta_sigma = torch.tensor(delta).unsqueeze(1)
        H = torch.cat(
            (sigma_current.unsqueeze(1).to(delta_sigma.device), delta_sigma), dim=1
        )
        dataset[i].x = H
    return dataset


class GCNBlock(torch.nn.Module):
    def __init__(self, channels, in_channels=2):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.conv_layers = torch.nn.ModuleList()
        for out_channels in channels:
            self.conv_layers.append(GCNConv(in_channels, out_channels))
            in_channels = out_channels
        self.conv_layers.append(GCNConv(in_channels, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.conv_layers:
            layer.reset_parameters()

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, ei)
            if i < (len(self.conv_layers) - 1):
                x = torch.nn.functional.relu(x)
        return x


class ResidualGCNBlock(GCNBlock):
    """Predict a correction to the Newton feature stored in channel one."""

    def __init__(self, channels, in_channels=5, newton_channel=1):
        super().__init__(channels, in_channels=in_channels)
        self.newton_channel = int(newton_channel)
        last = self.conv_layers[-1]
        torch.nn.init.zeros_(last.lin.weight)
        if last.bias is not None:
            torch.nn.init.zeros_(last.bias)

    def forward(self, data):
        correction = super().forward(data)
        return data.x[:, self.newton_channel : self.newton_channel + 1] + correction


class PhysicsProposalResidualGCNBlock(GCNBlock):
    """Predict a correction around ``current + per-stage LM direction``.

    Channels zero and one must contain the normalized current conductivity
    change and the newly recomputed normalized LM direction, respectively.
    Zero initialization of the final graph layer makes the initial network
    exactly equal to the nonlinear physics proposal.
    """

    def __init__(self, channels, in_channels=2, state_channel=0, update_channel=1):
        super().__init__(channels, in_channels=in_channels)
        self.state_channel = int(state_channel)
        self.update_channel = int(update_channel)
        last = self.conv_layers[-1]
        torch.nn.init.zeros_(last.lin.weight)
        if last.bias is not None:
            torch.nn.init.zeros_(last.bias)

    def forward(self, data):
        correction = super().forward(data)
        proposal = (
            data.x[:, self.state_channel : self.state_channel + 1]
            + data.x[:, self.update_channel : self.update_channel + 1]
        )
        return proposal + correction


def trainModel(model, dataset, optimizer, split, batch_size, max_epochs, patience, start_time):
    split = int(split * len(dataset))
    loader_tr = DataLoader(dataset[:split], batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(dataset[split:], batch_size=batch_size, shuffle=False)

    loss_tr = torch.zeros(max_epochs)
    loss_va = torch.zeros(max_epochs)
    pat = patience
    loss_va_min = 1e7
    best_model = copy.deepcopy(model)
    best_epoch = 0

    for epoch in range(max_epochs):
        loss_tr[epoch] = train(model, loader_tr, optimizer, max(split, 1))
        loss_va[epoch] = valid(model, loader_va, max(len(dataset) - split, 1))

        elapsed = time.time() - start_time
        hrs, rem = divmod(int(elapsed), 3600)
        mins, sec = divmod(rem, 60)
        print(
            "# ({:02d}:{:02d}:{:02d}) Epoch {:3d} | Training Loss {:.4f} | Validation Loss {:.4f}".format(
                hrs, mins, sec, epoch, loss_tr[epoch] * 1000, loss_va[epoch] * 1000
            )
        )

        if loss_va[epoch] <= loss_va_min:
            loss_va_min = loss_va[epoch]
            best_model = copy.deepcopy(model)
            pat = patience
            best_epoch = epoch
        else:
            pat -= 1
            if pat == 0:
                for model_w, best_w in zip(model.parameters(), best_model.parameters()):
                    model_w.data = best_w.data
                print("# Training stopped early on epoch", best_epoch)
                break

    return model, loss_tr, loss_va


def train(model, loader_tr, optimizer, n):
    model.train()
    loss_all = 0
    for data in loader_tr:
        optimizer.zero_grad()
        sigma_k = model(data)
        loss = F.mse_loss(sigma_k, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / n


def valid(model, loader_va, n):
    model.eval()
    loss_all = 0
    with torch.no_grad():
        for data in loader_va:
            loss = F.mse_loss(model(data), data.y)
            loss_all += data.num_graphs * loss.item()
    return loss_all / n


def applyModel(model, dataset):
    model.eval()
    predictions = torch.zeros((len(dataset), dataset[0].x.shape[0]))
    with torch.no_grad():
        for i in range(len(dataset)):
            predictions[i, :] = model(dataset[i]).squeeze()
            dataset[i].x = predictions[i, :].unsqueeze(dim=1)
    return dataset, predictions
