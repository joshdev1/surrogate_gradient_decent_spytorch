import torch

nb_inputs = 100
nb_hidden = 4
nb_outputs = 2
time_step = 1e-3
nb_steps = 200
batch_size = 256

dtype = torch.float
device = torch.device("cpu")

# TODO: Absolute and relative imports for more succinct access to model parameters
