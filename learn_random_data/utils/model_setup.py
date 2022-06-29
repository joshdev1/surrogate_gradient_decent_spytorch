import numpy as np
import torch
from learn_random_data.utils.model_parameters import BATCH_SIZE, DEVICE, DTYPE, NB_INPUTS, TIME_STEP,\
    NB_HIDDEN, NB_OUTPUTS, TAU_MEM, TAU_SYN


def alpha():
    return float(np.exp(-TIME_STEP/TAU_SYN))


def beta():
    return float(np.exp(-TIME_STEP/TAU_MEM))


def y_data():  # random labels for the random data
    return torch.tensor(1*(np.random.rand(BATCH_SIZE) < 0.5), device=DEVICE)


def _weight_scale():
    return 7*(1.0-beta())  # this should give us some spikes to begin with


def _initialize_weight_tensor(layer1, layer2):
    return torch.empty((layer1, layer2),  device=DEVICE, dtype=DTYPE, requires_grad=True)


def input_to_hidden_synaptic_weights():
    w1 = _initialize_weight_tensor(NB_INPUTS, NB_HIDDEN)
    torch.nn.init.normal_(w1, mean=0.0, std=_weight_scale() / np.sqrt(NB_INPUTS))
    return w1


def hidden_to_output_synaptic_weights():
    w2 = _initialize_weight_tensor(NB_HIDDEN, NB_OUTPUTS)
    torch.nn.init.normal_(w2, mean=0.0, std=_weight_scale() / np.sqrt(NB_HIDDEN))
    return w2
