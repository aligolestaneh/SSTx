import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP model with dropout and uncertainty prediction options.

    in -> hidden -> 2*hidden -> 4*hidden -> 2*hidden -> hidden -> out/logvar
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=32,
        dropout=0,
        pred_var=False,
        pred_cov=False,
    ):
        """Initialize the MLP model with given dimensions"""
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pred_var = pred_var
        self.pred_cov = pred_cov

        self.in_net = self.block(in_dim, hidden_dim, dropout=0)
        self.block1 = self.block(hidden_dim, 2 * hidden_dim, dropout)
        self.block2 = self.block(2 * hidden_dim, 4 * hidden_dim, dropout)
        self.block3 = self.block(4 * hidden_dim, 2 * hidden_dim, dropout)
        self.block4 = self.block(2 * hidden_dim, hidden_dim, dropout=0)
        self.out_net = nn.Linear(hidden_dim, out_dim)
        if pred_var:
            self.uncertainty_net = nn.Linear(hidden_dim, out_dim)
        elif pred_cov:
            self.uncertainty_net = nn.Linear(
                hidden_dim, out_dim * (out_dim + 1) // 2
            )

    def block(self, in_dim, out_dim, dropout=0):
        """Simple block with linear layer, leaky relu, and dropout."""
        if not dropout:
            fc = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(),
            )
        else:
            fc = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            )
        return fc

    def layers(self, x):
        """Forward pass through the layers."""
        x = self.in_net(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x):
        """Forward pass, return pred. Include logvar if include uncertainty."""
        x = self.layers(x)
        pred = self.out_net(x)

        # if include logvar
        if self.pred_var or self.pred_cov:
            uncertainty = self.uncertainty_net(x)
            pred = torch.cat([pred, uncertainty], dim=-1)

        return pred


class ResidualPhysics(MLP):
    """Residual Physics model.
            ----------------
            |              ↓
    x -> physics -> MLP -> + -> pred/logvar
    |                ↑
    ------------------
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        equation,
        hidden_dim=32,
        dropout=0,
        pred_var=False,
        pred_cov=False,
    ):
        """Initialize the MLP model with given dimensions"""
        super(ResidualPhysics, self).__init__(
            in_dim + out_dim,
            out_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_var=pred_var,
            pred_cov=pred_cov,
        )
        self.equation = equation

    def forward(self, x):
        """Residual physics forward pass"""
        # Physics
        eq = self.equation(x)
        # MLP
        x_p = torch.cat([x, eq], dim=-1)
        x = self.layers(x_p)
        pred = self.out_net(x)
        # Residual
        pred = pred + eq

        # if include logvar
        if self.pred_var or self.pred_cov:
            uncertainty = self.uncertainty_net(x)
            pred = torch.cat([pred, uncertainty], dim=-1)

        return pred
