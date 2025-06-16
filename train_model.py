import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.utils import DataLoader
from models.torch_model import MLP, ResidualPhysics
from models.torch_loss_se2 import nll_se2_loss, mse_se2_loss
from models.model import TorchModel, OptTorchModel, OptTorchModel_2
from models.physics import push_physics


def load_model(
    model_class,
    in_dim,
    out_dim,
    hidden=32,
    dropout=0,
    pred_var=False,
    lr=1e-3,
    batch_size=16,
    epochs=1000,
    device=None,
):
    """
    Load model function
    Return a model with Sklearn style API for PyTorch
    """
    if device is not None and isinstance(device, str):
        device = torch.device(device)
    if model_class == "residual":
        model_class = lambda: ResidualPhysics(
            in_dim, out_dim, push_physics, hidden, dropout, pred_var
        )
    elif model_class == "mlp":
        model_class = lambda: MLP(in_dim, out_dim, hidden, dropout, pred_var)
    else:
        raise ValueError(f"Model class {model_class} not supported")

    # Use NLL loss if variance is predicted
    loss_fn = nll_se2_loss if pred_var else mse_se2_loss

    model = TorchModel(
        model_class,
        optimizer=torch.optim.Adam,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        loss_fn=loss_fn,
        score_fn=mse_se2_loss,
        verbose=1,
        device=device,
    )
    return model


def load_opt_model(model, start_model, epochs=1000, device=None):
    model = OptTorchModel(
        model,
        start_model,
        lr=1e-4,
        epochs=epochs,
        device=device,
    )
    return model


def load_opt_model_2(model, lr=1e-4, epochs=1000, device=None):
    model = OptTorchModel_2(
        model,
        lr=lr,
        epochs=epochs,
        device=device,
    )
    return model


def main(model_type, object_name):
    # Seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data
    data_loader = DataLoader(object_name, "data", val_size=1000)
    datasets = data_loader.load_data()

    # Model
    pred_var = False
    model = load_model(model_type, 3, 3, pred_var=pred_var)

    # Train
    tr_losses, val_losses = model.fit(
        datasets["x_pool"],  # [:200],
        datasets["y_pool"],  # [:200],
        datasets["x_val"],
        datasets["y_val"],
    )

    # Evaluate
    y_pred = model.predict(datasets["x_val"])
    y = datasets["y_val"]
    # Absolute loss in position and rotation
    mu = y_pred[:, :3]
    position_error = np.mean(np.linalg.norm(y[:, :2] - mu[:, :2], axis=1))
    rotation_error = np.mean(np.abs(y[:, 2] - mu[:, 2]))
    print(
        f"Mean Absolute Error - Position: {position_error}"
        + f" - Rotation: {rotation_error}"
    )
    plt.plot(np.arange(len(tr_losses)), tr_losses)
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.show()

    model.save(f"saved_models/{model_type}_{object_name}.pt")


if __name__ == "__main__":
    model_type = "residual"
    object_name = "cracker_box"

    main(model_type, object_name)
