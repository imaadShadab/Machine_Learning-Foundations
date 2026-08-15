import numpy as np


def mse_loss(prediction, y):
    return 0.5 * np.mean((prediction - y) ** 2)


def loss_gradient(prediction, y): # Cost Gradient
    """
    dC/dŷ
    How Cost changes as output changes 
    Gradient of the loss with respect to the prediction.
    """
    return prediction - y


def weight_gradient(x, prediction, y):
    """
    dC/dw

    Uses the chain rule:

    dC/dw = dC/dŷ * dŷ/dz * dz/dw

    For our current linear output:

    dC/dŷ = prediction - y
    dŷ/dz = 1
    dz/dw = x
    """
    dC_dyhat = loss_gradient(prediction, y)
    dyhat_dz = 1
    dz_dw = x

    return dC_dyhat * dyhat_dz * dz_dw


def bias_gradient(prediction, y):
    """
    dC/db

    dC/db = dC/dŷ * dŷ/dz * dz/db
    """
    dC_dyhat = loss_gradient(prediction, y)
    dyhat_dz = 1
    dz_db = 1

    return dC_dyhat * dyhat_dz * dz_db