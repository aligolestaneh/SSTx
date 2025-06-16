import torch
import torch.nn.functional as F


# SE2 Pose functions
def log_se2(transforms):
    """Logarithm map from SE(2) -> se(2).

    transforms is a Nx3x3 batch homogeneous transformation matrix.
    Returns a Nx3 batch of vectors [delta_x, delta_y, omage] in tangent space.

    omega = arctan2(rot[1, 0], rot[0, 0])
    [dx, dy] = V(w)^-1 * t
    """
    device = transforms.device
    batch_size = transforms.shape[0]

    rot = transforms[:, 0:2, 0:2]  # (N, 2, 2)
    t = transforms[:, 0:2, 2]  # (N, 2)

    # Compute the omega
    omega = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])  # (N,)
    c = torch.cos(omega)
    s = torch.sin(omega)

    # Compute dx and dy
    # V_inv = w / (2 * (1 - cos(w))) *
    #         [[    sin(w)   , 1 - cos(w)]
    #          [-(1 - cos(w)),   sin(w)  ]]
    # Avoid division by zero in V_inv
    eps = 1e-3
    mask = torch.abs(omega) >= eps
    # only compute for elements where omega is not too small
    V_inv = torch.zeros(
        batch_size, 2, 2, device=device, dtype=transforms.dtype
    )  # (N, 2, 2)
    denom = 2 * (1 - c)
    V_inv[mask] = (
        omega[mask].unsqueeze(-1).unsqueeze(-1)
        / denom[mask].unsqueeze(-1).unsqueeze(-1)
    ) * torch.stack(
        [
            torch.stack([s[mask], 1 - c[mask]], dim=1),
            torch.stack([c[mask] - 1, s[mask]], dim=1),
        ],
        dim=1,
    )
    V_inv[~mask] = torch.eye(2, device=device, dtype=transforms.dtype)

    dt = torch.bmm(V_inv, t.unsqueeze(-1)).squeeze(-1)  # (B, 2)
    dx, dy = dt[:, 0], dt[:, 1]

    return torch.stack([dx, dy, omega], dim=1)


def get_transform_se2(params):
    """
    Convert SE2 parameters to SE2 transform matrices.

    params is a Nx3 batch of vectors [x, y, theta].
    This is different from the tangent space vector [dx, dy, omega].
    Returns a Nx3x3 batch of SE2 transform matrices.
    """
    x, y, theta = params[:, 0], params[:, 1], params[:, 2]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    return torch.stack(
        [
            torch.stack([cos_theta, -sin_theta, x], dim=1),
            torch.stack([sin_theta, cos_theta, y], dim=1),
            torch.stack([zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )


def inv_transform_se2(transforms):
    """Invert a batch of SE2 transform matrices."""
    rot = transforms[:, :2, :2]  # (N, 2, 2)
    t = transforms[:, :2, 2:]  # (N, 2, 1)

    rot_transposed = rot.transpose(1, 2)  # (N, 2, 2)
    t_new = -torch.bmm(rot_transposed, t)  # (N, 2, 1)

    transforms_inv = torch.eye(
        3, device=transforms.device, dtype=transforms.dtype
    ).repeat(transforms.size(0), 1, 1)
    transforms_inv[:, :2, :2] = rot_transposed
    transforms_inv[:, :2, 2] = t_new.squeeze(-1)

    return transforms_inv


# SE2 Pose Loss functions
def mse_se2_loss(y_pred, y_true):
    """MSE Loss for SE2 Pose.
    Also handle the case of having logvar in output by ignoring it.
    """
    # If y_pred includes log-variance or extra dims, trim it
    if y_pred.shape[-1] > y_true.shape[-1]:
        y_pred = y_pred[:, : y_true.shape[-1]]

    y_pred_mat = get_transform_se2(y_pred)  # (N, 3, 3)
    y_true_mat = get_transform_se2(y_true)  # (N, 3, 3)
    err_mat = torch.matmul(inv_transform_se2(y_true_mat), y_pred_mat)
    delta_err = log_se2(err_mat)  # (N, 3)

    # Compute MSE loss in tangent space
    loss = torch.mean(delta_err**2)
    # # add weights to rotation error
    # loss = torch.mean(
    #     delta_err[:, 0] ** 2
    #     + delta_err[:, 1] ** 2
    #     + 0.1 * delta_err[:, 2] ** 2
    # )
    return loss


def nll_se2_loss(y_pred, y_true):
    """NLL Loss for SE2 Pose."""
    dim = y_true.shape[-1]
    mu, logvar = y_pred[:, :dim], y_pred[:, dim:]

    y_pred_mat = get_transform_se2(mu)  # (N, 3, 3)
    y_true_mat = get_transform_se2(y_true)  # (N, 3, 3)
    err_mat = torch.matmul(inv_transform_se2(y_true_mat), y_pred_mat)
    delta_err = log_se2(err_mat)  # (N, 3)

    nll = 0.5 * (
        logvar + (delta_err**2 / (torch.exp(logvar) + 1e-8))
    )  # (N, 3)
    # loss = torch.mean(torch.sum(nll, dim=-1))
    loss = torch.mean(nll)
    return loss


# Loss functions for general use but deprecated
# Since the prediction is a SE2 Pose
def mse_loss(y_pred, y_true):
    """A simple wrapper of mse loss.
    Also handle the case of having logvar in output by ignoring it.
    """
    # If y_pred includes log-variance or extra dims, trim it
    if y_pred.shape[-1] > y_true.shape[-1]:
        y_pred = y_pred[:, : y_true.shape[-1]]

    return F.mse_loss(y_pred, y_true)


def nll_loss(y_pred, y_true):
    """NLL Loss function, which includes uncertainty."""
    dim = y_true.shape[-1]
    mu, logvar = y_pred[:, :dim], y_pred[:, dim:]

    # Negative log likelihood loss
    # L = 0.5 * log(var) + 0.5 * ((mu - y)^2 / var) + constant
    nll = 0.5 * (logvar + ((y_true - mu) ** 2) / (torch.exp(logvar) + 1e-8))
    loss = torch.mean(nll)
    return loss


def multi_nll_loss(y_pred, y_true):
    """Multivariate NLL Loss function."""
    dim = y_true.shape[-1]
    mu, ls = y_pred[:, :dim], y_pred[:, dim:]
    assert ls.shape[-1] == dim * (dim + 1) // 2, "Incorrect Cov shape."

    # Cholesky factors L (shape: batch_size x dim x dim)
    batch_size = y_pred.shape[0]
    mu, ls = y_pred[:, :dim], y_pred[:, dim:]

    # Cholesky factors L (shape: batch_size x dim x dim)
    l_cholesky = torch.zeros((batch_size, dim, dim), device=y_pred.device)
    # Fill in L's lower triangular part with the predicted parameters
    tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)
    l_cholesky[:, tril_indices[0], tril_indices[1]] = ls
    # Ensure the diagonal entries are positive (softplus).
    diag_indices = torch.arange(dim, device=y_pred.device)
    l_cholesky[:, diag_indices, diag_indices] = F.softplus(
        l_cholesky[:, diag_indices, diag_indices]
    )

    # Compute the difference between the true target and predicted mean.
    diff = (y_true - mu).unsqueeze(2)  # (batch_size, dim, 1)
    # Solve L * z = diff for z (i.e., compute z = L^{-1} diff) for each sample.
    # This avoids explicit computation of the covariance inverse.
    z = torch.linalg.solve_triangular(l_cholesky, diff, upper=False)
    # Mahalanobis distance term: (y - μ)ᵀ Σ⁻¹ (y - μ) = sum(z^2)
    mahalanobis = torch.sum(z**2, dim=[1, 2])  # (batch_size,)

    # The log-determinant of Σ can be computed from L:
    # log(det(Σ)) = 2 * sum(log(diag(L)))
    diag_indices = torch.arange(dim, device=y_pred.device)
    log_det_sigma = 2 * torch.sum(
        torch.log(l_cholesky[:, diag_indices, diag_indices] + 1e-6), dim=1
    )

    # Compute the per-sample negative log-likelihood (remove constant term).
    nll = 0.5 * (mahalanobis + log_det_sigma)
    # Return the mean loss over the batch.
    return torch.mean(nll)
