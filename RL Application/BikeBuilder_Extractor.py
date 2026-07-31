# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch as th
import torch.nn as nn
 
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def flat_dim(shape: tuple[int, ...] | None) -> int:
    # Returns how many single numbers an observation contains once flattened.
    # e.g. shape (50, 5, 2) -> 500
    assert shape is not None
    return int(np.prod(shape))
 
 
def mlp_branch(input_dim: int, output_dim: int) -> nn.Sequential:
    # A small two-layer network: input -> hidden -> output.
    # hidden_dim is a simple heuristic (roughly half the input, but never smaller
    # than the output) - a reasonable starting point, not a hard rule.
    hidden_dim = max(output_dim, input_dim // 2)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )

# =============================================================================
# CUSTOM FEATURES EXTRACTOR
# =============================================================================
 
class BikeBuilder_Extractor(BaseFeaturesExtractor):
    '''
    Groups the observation Dict into five branches:
      1. guide_curve      - the resampled curve points
      2. stock_geometry   - the full frame stock geometry
      3. stock_mask       - which frames are still available
      4. current          - current_frame
      5. progress         - progress + max_t (combined)
 
    Each branch is a small MLP. Their outputs are concatenated and passed
    through one final layer to produce the feature vector the policy and
    value networks actually see.
    '''
 
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
 
        # ---- Branch output sizes ----
        # How many numbers each branch boils its observation down to.
        # Larger / more complex observations get more room to work with.
        guide_curve_out    = 32
        stock_geometry_out = 64
        stock_mask_out     = 16
        current_out        = 16
        progress_out       = 8
 
        # ---- Branch 1: guide_curve ----
        guide_curve_dim = flat_dim(observation_space["guide_curve"].shape)
        self.guide_curve_net = mlp_branch(guide_curve_dim, guide_curve_out)
 
        # ---- Branch 2: stock_geometry ----
        stock_geometry_dim = flat_dim(observation_space["stock_geometry"].shape)
        self.stock_geometry_net = mlp_branch(stock_geometry_dim, stock_geometry_out)
 
        # ---- Branch 3: stock_mask ----
        stock_mask_dim = flat_dim(observation_space["stock_mask"].shape)
        self.stock_mask_net = mlp_branch(stock_mask_dim, stock_mask_out)
 
        # ---- Branch 4: current_frame  ----
        current_dim = flat_dim(observation_space["current_frame"].shape)
        self.current_net = mlp_branch(current_dim, current_out)
 
        # ---- Branch 5: progress + max_t (combined) ----
        progress_dim = (
            flat_dim(observation_space["progress"].shape)
            + flat_dim(observation_space["max_t"].shape)
        )
        self.progress_net = mlp_branch(progress_dim, progress_out)
 
        # ---- Combining layer ----
        # Takes every branch's output, stitched together, and projects it down
        # to features_dim - the size the rest of PPO (policy + value heads) expects.
        combined_dim = (
            guide_curve_out
            + stock_geometry_out
            + stock_mask_out
            + current_out
            + progress_out
        )
        self.combine_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )
 
    def forward(self, observations: dict) -> th.Tensor:
        batch_size = observations["guide_curve"].shape[0]
 
        # Flatten each observation from its natural shape (e.g. (50, 5, 2))
        # down to a single row per item in the batch (e.g. (batch, 500)).
        guide_curve_flat    = observations["guide_curve"].reshape(batch_size, -1)
        stock_geometry_flat = observations["stock_geometry"].reshape(batch_size, -1)
        stock_mask_flat     = observations["stock_mask"].reshape(batch_size, -1)
 
        current_flat = observations["current_frame"].reshape(batch_size, -1)
 
        progress_flat = th.cat([
            observations["progress"].reshape(batch_size, -1),
            observations["max_t"].reshape(batch_size, -1),
        ], dim=1)
 
        # Run each branch on its own slice of the observation.
        guide_curve_feat    = self.guide_curve_net(guide_curve_flat)
        stock_geometry_feat = self.stock_geometry_net(stock_geometry_flat)
        stock_mask_feat     = self.stock_mask_net(stock_mask_flat)
        current_feat        = self.current_net(current_flat)
        progress_feat       = self.progress_net(progress_flat)
 
        # Stitch every branch's output together, then combine into one vector.
        combined = th.cat([
            guide_curve_feat,
            stock_geometry_feat,
            stock_mask_feat,
            current_feat,
            progress_feat,
        ], dim=1)
 
        return self.combine_net(combined)