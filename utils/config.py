import torch
import random
import numpy as np

# MAX_SEQ = 200
# MIN_SEQ = 3
# SEED = 2021

def set_random_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
