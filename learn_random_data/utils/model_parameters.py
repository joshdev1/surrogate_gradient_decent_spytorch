import torch

NB_INPUTS = 100
NB_HIDDEN = 4
NB_OUTPUTS = 2
TIME_STEP = 1e-3
NB_STEPS = 200
BATCH_SIZE = 256

TAU_MEM = 10e-3
TAU_SYN = 5e-3

DTYPE = torch.float
DEVICE = torch.device("cpu")

# TODO: Absolute and relative imports for more succinct access to model parameters
