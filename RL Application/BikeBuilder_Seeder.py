'''
This script contains the seeding for the training.
The functions between line 23 till 29 produce internal states.
These are needed to keep the seed generation consistent throughout
the different libraries that need it. Otherwise, if for example
np.random.rand() was called a different number would be assigned each time.
np.random.seed() ensures that each np.random operation uses the same seed consistently
without having to specify it. 

'''

# =============================================================================
# IMPORTS
# =============================================================================
import random
import numpy as np
import torch as th

# =============================================================================
# SEED GENERATOR
# =============================================================================

def seed_everything(seed: int | None = None) -> int:
    # If no seed is specified, generate one
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
        print(f"No seed specified - generated: {seed} as seed")

    else:
        print(f"SEED SPECIFIED AS {seed}")

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    return seed

