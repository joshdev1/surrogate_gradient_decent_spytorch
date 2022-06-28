import torch
import model_parameters


def generate_random_data(freq):
    prob = freq*model_parameters.time_step
    mask = torch.rand((model_parameters.batch_size, model_parameters.nb_steps, model_parameters.nb_inputs),
                      device=model_parameters.device, dtype=model_parameters.dtype)
    x_data = torch.zeros((model_parameters.batch_size, model_parameters.nb_steps,
                          model_parameters.nb_inputs), device=model_parameters.device, dtype=model_parameters.dtype,
                         requires_grad=False)
    x_data[mask < prob] = 1.0
    return x_data
