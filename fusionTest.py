# ------------------------------------------------------------
# Batch car control matching via gradient descent (PyTorch)
# ------------------------------------------------------------
from typing import Callable, Dict, Any, Optional, Tuple
import math
import torch

torch.set_default_dtype(torch.float64)
DEVICE = "cuda"  # change to "cuda" if available


# --------------------- 1) Car dynamics ----------------------
def car_dynamics(start: torch.Tensor, control: torch.Tensor, *, duration: float) -> torch.Tensor:
    """
    start:   (..., 3) = [x, y, yaw]
    control: (..., 2) = [v, omega]
    duration: scalar step
    returns: next state (..., 3)
    """
    x, y, yaw = start.unbind(-1)
    v, w = control.unbind(-1)

    dx = v * torch.cos(yaw) * duration
    dy = v * torch.sin(yaw) * duration
    dyaw = w * duration

    return torch.stack([x + dx, y + dy, yaw + dyaw], dim=-1)


# -------- 2) Wrapper: explicit Jacobian wrt control u -------
def control_jacobian(
    dynamics: Callable[..., torch.Tensor],
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    dynamics_kwargs: Optional[Dict[str, Any]] = None,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (y, df/du) with shapes:
      y:    (..., state_dim)
      dfdu: (..., state_dim, control_dim)
    """
    if dynamics_kwargs is None:
        dynamics_kwargs = {}

    u = u.clone().requires_grad_(True)

    with torch.enable_grad():
        y = dynamics(x.detach(), u, **dynamics_kwargs)

    # flatten leading batch dims for per-item jacobian clarity
    def _flatten(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, t.shape[-1]) if t.ndim > 1 else t.unsqueeze(0)

    x_flat = _flatten(x.detach())
    u_flat = _flatten(u)
    y_flat = _flatten(y)

    B = x_flat.shape[0]
    state_dim = y_flat.shape[-1]
    from torch.autograd.functional import jacobian

    def per_item(i: int) -> torch.Tensor:
        xi = x_flat[i]
        ui = u_flat[i]

        def g(u_local: torch.Tensor) -> torch.Tensor:
            return dynamics(xi, u_local, **dynamics_kwargs)

        J = jacobian(g, ui, vectorize=True, create_graph=create_graph, strict=False)
        return J.reshape(state_dim, -1)

    Js = [per_item(i) for i in range(B)]
    J_flat = torch.stack(Js, dim=0)  # (B, state_dim, control_dim)

    batch_shape = y.shape[:-1]
    control_dim = J_flat.shape[-1]
    dfdu = J_flat.reshape(*batch_shape, state_dim, control_dim)
    return y, dfdu


# ------------------------ Helpers ---------------------------
def uniform(low: float, high: float, shape, device=DEVICE):
    return (low - high) * torch.rand(shape, device=device) + high


def sample_states(N: int, bounds: Dict[str, tuple], device=DEVICE):
    # bounds keys: x, y, yaw
    x = uniform(*bounds["x"], (N, 1), device)
    y = uniform(*bounds["y"], (N, 1), device)
    yaw = uniform(*bounds["yaw"], (N, 1), device)
    return torch.cat([x, y, yaw], dim=-1)


def sample_controls(N: int, bounds: Dict[str, tuple], device=DEVICE):
    # bounds keys: v, omega
    v = uniform(*bounds["v"], (N, 1), device)
    w = uniform(*bounds["omega"], (N, 1), device)
    return torch.cat([v, w], dim=-1)


def clamp_controls(u: torch.Tensor, bounds: Dict[str, tuple]):
    u[..., 0].clamp_(min=bounds["v"][0], max=bounds["v"][1])
    u[..., 1].clamp_(min=bounds["omega"][0], max=bounds["omega"][1])


# -------------------------- Main ----------------------------
if __name__ == "__main__":
    torch.manual_seed(1611)

    # 8) Sampling bounds (tweak as you like)
    bounds_state = {
        "x": (-10.0, 10.0),
        "y": (-10.0, 10.0),
        "yaw": (-math.pi, math.pi),
    }
    bounds_control = {
        "v": (0.0, 5.0),  # forward speed only, 0..5 m/s
        "omega": (-1.0, 1.0),  # turn rate in rad/s
    }
    dt = 0.3

    N = 1000

    # 3) Sample 1000 random start states and 1000 random controls
    x0 = sample_states(N, bounds_state).to(DEVICE)  # (N, 3)
    u_true = sample_controls(N, bounds_control).to(DEVICE)  # (N, 2)

    # 4) Get target states from these samples
    with torch.no_grad():
        x_target = car_dynamics(x0, u_true, duration=dt)  # (N, 3)

    # 5) Sample another 1000 random controls (initial guess)
    u_hat = sample_controls(N, bounds_control).to(DEVICE)
    u_hat.requires_grad_(True)

    # (Optional) sanity: Jacobian shape check on a small subset via wrapper
    _ = control_jacobian(car_dynamics, x0[:5], u_hat[:5], dynamics_kwargs={"duration": dt})

    # 6) Optimize the new controls to match targets
    optimizer = torch.optim.Adam([u_hat], lr=0.2)
    steps = 200

    def loss_fn(pred, target):
        return torch.nn.functional.mse_loss(pred, target)

    # Track before/after for comparison
    with torch.no_grad():
        init_state_err = loss_fn(car_dynamics(x0, u_hat, duration=dt), x_target).item()
        init_ctrl_rmse = torch.sqrt(torch.mean((u_hat - u_true) ** 2)).item()

    for it in range(steps):
        optimizer.zero_grad(set_to_none=True)
        x_pred = car_dynamics(x0, u_hat, duration=dt)
        loss = loss_fn(x_pred, x_target)
        loss.backward()
        optimizer.step()

        # project back into bounds
        with torch.no_grad():
            clamp_controls(u_hat, bounds_control)

        if (it + 1) % 25 == 0:
            print(f"iter {it+1:3d}  loss={loss.item():.6e}")

    with torch.no_grad():
        final_state_err = loss_fn(car_dynamics(x0, u_hat, duration=dt), x_target).item()
        final_ctrl_rmse = torch.sqrt(torch.mean((u_hat - u_true) ** 2)).item()

    # 7) Compare sampled vs original controls
    print("\n--- Results ---")
    print(f"State MSE before: {init_state_err:.6e}")
    print(f"State MSE after : {final_state_err:.6e}")
    print(f"Control RMSE before: {init_ctrl_rmse:.6e}")
    print(f"Control RMSE after : {final_ctrl_rmse:.6e}")

    # You can also inspect a few rows:
    with torch.no_grad():
        idx = torch.randint(0, N, (5,))
        print("\nSample comparisons [v, omega]:")
        for i in idx.tolist():
            print(f"true={u_true[i].tolist()}  hat={u_hat[i].tolist()}")
