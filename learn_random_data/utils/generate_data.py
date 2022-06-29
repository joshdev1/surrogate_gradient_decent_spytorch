import torch
from learn_random_data.utils.model_parameters import BATCH_SIZE, DEVICE, DTYPE, NB_INPUTS, NB_STEPS, TIME_STEP


def generate_random_data(freq):
    prob = freq * TIME_STEP
    mask = torch.rand((BATCH_SIZE, NB_STEPS, NB_INPUTS),
                      device=DEVICE, dtype=DTYPE)
    x_data = torch.zeros((BATCH_SIZE, NB_STEPS,
                          NB_INPUTS), device=DEVICE, dtype=DTYPE,
                         requires_grad=False)
    x_data[mask < prob] = 1.0
    return x_data
