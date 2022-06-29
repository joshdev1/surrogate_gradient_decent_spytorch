import torch
from learn_random_data.utils.model_parameters import batch_size, device, dtype, nb_inputs, nb_steps, time_step


def generate_random_data(freq):
    prob = freq * time_step
    mask = torch.rand((batch_size, nb_steps, nb_inputs),
                      device=device, dtype=dtype)
    x_data = torch.zeros((batch_size, nb_steps,
                          nb_inputs), device=device, dtype=dtype,
                         requires_grad=False)
    x_data[mask < prob] = 1.0
    return x_data
