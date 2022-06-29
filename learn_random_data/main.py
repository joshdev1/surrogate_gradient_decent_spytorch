import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
import torch.nn as nn
from learn_random_data.utils.generate_data import generate_random_data
from learn_random_data.utils.model_parameters import BATCH_SIZE, DEVICE, DTYPE, NB_INPUTS, NB_STEPS, TIME_STEP,\
    NB_HIDDEN, NB_OUTPUTS, TAU_MEM, TAU_SYN


def alpha():
    return float(np.exp(-TIME_STEP/TAU_SYN))


def beta():
    return float(np.exp(-TIME_STEP/TAU_MEM))


def y_data():  # random labels for the random data
    return torch.tensor(1*(np.random.rand(BATCH_SIZE) < 0.5), device=DEVICE)


def weight_scale():
    return 7*(1.0-beta())  # this should give us some spikes to begin with


def initialize_weight_tensor(layer1, layer2):
    return torch.empty((layer1, layer2),  device=DEVICE, dtype=DTYPE, requires_grad=True)


def input_to_hidden_synaptic_weights():
    w1 = initialize_weight_tensor(NB_INPUTS, NB_HIDDEN)
    torch.nn.init.normal_(w1, mean=0.0, std=weight_scale()/np.sqrt(NB_INPUTS))
    return w1


def hidden_to_output_synaptic_weights():
    w2 = initialize_weight_tensor(NB_HIDDEN, NB_OUTPUTS)
    torch.nn.init.normal_(w2, mean=0.0, std=weight_scale()/np.sqrt(NB_HIDDEN))
    return w2


print("init done")


def multiply_input_spikes_with_weight_matrix():
    return torch.einsum("abc,cd->abd", (generate_random_data(5),  input_to_hidden_synaptic_weights()))


# this is the heaviside function
def spike_fn(x):
    out = torch.zeros_like(x)
    out[x > 0] = 1.0
    return out


syn = torch.zeros((BATCH_SIZE, NB_HIDDEN), device=DEVICE, dtype=DTYPE)
mem = torch.zeros((BATCH_SIZE, NB_HIDDEN), device=DEVICE, dtype=DTYPE)

# Here we define two lists which we use to record the membrane potentials and output spikes
mem_rec = []
spk_rec = []

# Here we loop over time
for t in range(NB_STEPS):
    mthr = mem - 1.0
    out = spike_fn(mthr)
    rst = out.detach()  # We do not want to backprop through the reset

    new_syn = alpha() * syn + multiply_input_spikes_with_weight_matrix()[:, t]
    new_mem = (beta() * mem + syn) * (1.0 - rst)

    mem_rec.append(mem)
    spk_rec.append(out)

    mem = new_mem
    syn = new_syn

# Now we merge the recorded membrane potentials into a single tensor
mem_rec = torch.stack(mem_rec, dim=1)
spk_rec = torch.stack(spk_rec, dim=1)


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
    gs=GridSpec(*dim)
    if spk is not None:
        dat = 1.0*mem
        dat[spk > 0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i == 0: a0 = ax = plt.subplot(gs[i])
        else: ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[i])
        ax.axis("off")


fig = plt.figure(dpi=100)
plot_voltage_traces(mem_rec, spk_rec)
plt.show()





