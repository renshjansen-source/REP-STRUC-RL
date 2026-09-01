'''A Custom Extractor using a stock-invariant PointNet style encoder for the
Geometry observations'''

# Things to still investigate here:
# The use of global maxing
# The lack of a hidden layer in the global + local combined NN of the geometry
# Use of a seperate MLP for stock masking (believe I did test it at one point)
# Is stock mask obs really needed with action masking? 
# Old implementation of current_frame (with sweep) had hidden layers on the 
# Combination stage. I have changed this. Test.
# With current setup of the progress / max_t NN the hidden layer is the same
# Size as the output layer. Not wrong, but remember this. 
# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch as th
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from internal_variables import IV

# =============================================================================
# ASSISTS
# =============================================================================
def flat_shape(shape: tuple[int, ...] | None) -> int:
    assert shape is not None
    return int(np.prod(shape))

def hidden_dim_size(input_dim: int, output_dim: int) -> int:
    hidden_dim = max(output_dim, input_dim // 2)
    return hidden_dim

# =============================================================================
# EXTRACTOR
# =============================================================================
class Custom_PointNet_Extractor(BaseFeaturesExtractor):
    '''
    Groups the observation Dict into five branches:
      1. guide_curve      - the resampled curve points
      2. stock_geometry   - the full frame stock geometry
      3. stock_mask       - which frames are still available (optional)
      4. current          - current_frame
      5. progress         - progress + max_t (combined)
      6. stock_areas      - the area obs, currently not used (optional)
    '''

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Flags for optional observations
        self.use_stock_mask  = "stock_mask"  in observation_space.spaces
        self.use_stock_areas = "stock_areas" in observation_space.spaces

        # Flags for fusion into stock_encoder (only meaningful if the obs exists at all)
        self.fuse_mask_in_stock  = self.use_stock_mask  and IV.fuse_mask_in_stock
        self.fuse_areas_in_stock = self.use_stock_areas and IV.fuse_areas_in_stock

        # ---------------------------------------------------------------------
        # GUIDE CURVE
        # ---------------------------------------------------------------------
        guide_curve_in    = flat_shape(observation_space["guide_curve"].shape)
        guide_curve_hidden    = hidden_dim_size(guide_curve_in, IV.guide_curve_out)
        self.guide_curve_net = nn.Sequential(
            nn.Linear(guide_curve_in, guide_curve_hidden),
            nn.ReLU(),
            nn.Linear(guide_curve_hidden, IV.guide_curve_out),
            nn.ReLU(),
        )
        
        # ---------------------------------------------------------------------
        # STOCK GEOMETRY - PointNet Style
        # ---------------------------------------------------------------------
        # Retrieve observation space shapes
        stock_shape = observation_space["stock_geometry"].shape
        assert stock_shape is not None
        if self.fuse_areas_in_stock:
            area_space = observation_space["stock_areas"]
            assert area_space.shape is not None
            area_shape = area_space.shape
        else:
            area_shape = None

        # Set frame shape
        n_frames, points_per_frame, coords_per_point = stock_shape

        # Set shape of area and mask depending on toggle
        area_in = area_shape[-1] if self.fuse_areas_in_stock else 0
        mask_in = 1 if self.fuse_mask_in_stock else 0

        # Input of stock encoder
        per_frame_in = points_per_frame * coords_per_point
        stock_encoder_in = per_frame_in + mask_in + area_in

        # Output of stock encoder
        frame_embed_out = 16

        # Hidden layer of stock encoder
        stock_encoder_hidden = hidden_dim_size(stock_encoder_in, frame_embed_out)

        # Stock-invariant MLP
        self.stock_encoder = nn.Sequential(
            nn.Linear(stock_encoder_in, stock_encoder_hidden),
            nn.ReLU(),
            nn.Linear(stock_encoder_hidden, frame_embed_out),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # STOCK GEOMETRY - Local and global outputs
        # ---------------------------------------------------------------------
        # After each frame has been passed seperately to through the invariant MLP
        # We use both a local output which is per frame, and global output which uses
        # element-wise maxing across the frames. 
        local_dim  = n_frames * frame_embed_out
        global_dim = frame_embed_out

        self.stock_combine = nn.Sequential(
            nn.Linear(local_dim + global_dim, IV.stock_geometry_out),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # STOCK MASK (optional)
        # ---------------------------------------------------------------------
        if self.use_stock_mask:
            stock_mask_in     = flat_shape(observation_space["stock_mask"].shape)
            stock_mask_hidden = hidden_dim_size(stock_mask_in, IV.stock_mask_out)
            self.stock_mask_net = nn.Sequential(
                nn.Linear(stock_mask_in, stock_mask_hidden),
                nn.ReLU(),
                nn.Linear(stock_mask_hidden, IV.stock_mask_out),
                nn.ReLU(),
            )

        # ---------------------------------------------------------------------
        # CURRENT FRAME - PointNet style if sweeped
        # ---------------------------------------------------------------------
        current_shape = observation_space["current_frame"].shape
        assert current_shape is not None
        # Either just coordinates [shape = 2] or multiple frames [shape = 3]
        self.current_frame_is_sweep = len(current_shape) == 3 

        if self.current_frame_is_sweep:
            n_cf, cf_points_per_frame, cf_coords = current_shape # cf = current_frame
            cf_embed_out = 16
            per_cf_in    = cf_points_per_frame * cf_coords

            cf_encoder_hidden = hidden_dim_size(per_cf_in, cf_embed_out)
            self.current_frame_encoder = nn.Sequential(
                nn.Linear(per_cf_in, cf_encoder_hidden),
                nn.ReLU(),
                nn.Linear(cf_encoder_hidden, cf_embed_out),
                nn.ReLU(),
            )
            cf_local_dim  = n_cf * cf_embed_out
            cf_global_dim = cf_embed_out

            self.current_net = nn.Sequential(
                nn.Linear(cf_local_dim + cf_global_dim, IV.current_out),
                nn.ReLU(),
            )

        else:
            current_in     = flat_shape(current_shape)
            current_hidden = hidden_dim_size(current_in, IV.current_out)
            self.current_net = nn.Sequential(
                nn.Linear(current_in, current_hidden),
                nn.ReLU(),
                nn.Linear(current_hidden, IV.current_out),
                nn.ReLU(),
            )

        # ---------------------------------------------------------------------
        # STOCK AREAS (optional)
        # ---------------------------------------------------------------------
        if self.use_stock_areas:
            stock_areas_in     = flat_shape(observation_space["stock_areas"].shape)
            stock_areas_hidden = hidden_dim_size(stock_areas_in, IV.stock_areas_out)
            self.stock_areas_net = nn.Sequential(
                nn.Linear(stock_areas_in, stock_areas_hidden),
                nn.ReLU(),
                nn.Linear(stock_areas_hidden, IV.stock_areas_out),
                nn.ReLU(),
            )

        # ---------------------------------------------------------------------
        # PROGRESS + MAX_T (combined)
        # ---------------------------------------------------------------------
        progress_in = (
            flat_shape(observation_space["progress"].shape)
            + flat_shape(observation_space["max_t"].shape)
        )
        progress_hidden = hidden_dim_size(progress_in, IV.progress_out)
        self.progress_net = nn.Sequential(
            nn.Linear(progress_in, progress_hidden),
            nn.ReLU(),
            nn.Linear(progress_hidden, IV.progress_out),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # FINAL COMBINE
        # ---------------------------------------------------------------------
        combined_dim = (
            IV.guide_curve_out
            + IV.stock_geometry_out
            + (IV.stock_mask_out  if self.use_stock_mask  else 0)
            + (IV.stock_areas_out if self.use_stock_areas else 0)
            + IV.current_out
            + IV.progress_out
        )
        self.combine_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    # -------------------------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------------------------
    def forward(self, observations: dict) -> th.Tensor:
        batch_size = observations["guide_curve"].shape[0] # Guide curve is an arbitrary selection - just need the actual batch size (on rollout it would be 128, e.g.)

        # ---------------------------------------------------------------------
        # Flat Observations
        # ---------------------------------------------------------------------
        guide_curve_flat = observations["guide_curve"].reshape(batch_size, -1)
        progress_flat = th.cat([
            observations["progress"].reshape(batch_size, -1),
            observations["max_t"].reshape(batch_size, -1),
        ], dim=1)

        guide_curve_feat  = self.guide_curve_net(guide_curve_flat)
        progress_feat     = self.progress_net(progress_flat)

        # Flatting for stock with area and mask toggles
        n_frames = observations["stock_geometry"].shape[1]
        stock_flat_per_frame = observations["stock_geometry"].reshape(batch_size * n_frames, -1)

        if self.fuse_mask_in_stock:
            mask_per_frame = observations["stock_mask"].reshape(batch_size * n_frames, 1)
            stock_flat_per_frame = th.cat([stock_flat_per_frame, mask_per_frame], dim=1)

        if self.fuse_areas_in_stock:
            area_per_frame = observations["stock_areas"].reshape(batch_size * n_frames, -1)
            stock_flat_per_frame = th.cat([stock_flat_per_frame, area_per_frame], dim=1)

        frame_embeddings = self.stock_encoder(stock_flat_per_frame)
        frame_embeddings = frame_embeddings.reshape(batch_size, n_frames, -1)

        local_feat  = frame_embeddings.reshape(batch_size, -1)
        global_feat = frame_embeddings.max(dim=1).values

        stock_geometry_feat = self.stock_combine(th.cat([local_feat, global_feat], dim=1))

        # Seperate stock and area observation
        if self.use_stock_mask:
            stock_mask_flat = observations["stock_mask"].reshape(batch_size, -1)
            stock_mask_feat = self.stock_mask_net(stock_mask_flat)

        if self.use_stock_areas:
            stock_areas_flat = observations["stock_areas"].reshape(batch_size, -1)
            stock_areas_feat = self.stock_areas_net(stock_areas_flat)

        if self.current_frame_is_sweep:
            n_cf = observations["current_frame"].shape[1]
            current_flat_per_frame = observations["current_frame"].reshape(batch_size * n_cf, -1)
            cf_embeddings = self.current_frame_encoder(current_flat_per_frame)
            cf_embeddings = cf_embeddings.reshape(batch_size, n_cf, -1)

            cf_local  = cf_embeddings.reshape(batch_size, -1)
            cf_global = cf_embeddings.max(dim=1).values
            current_feat = self.current_net(th.cat([cf_local, cf_global], dim=1))
        else:
            current_flat = observations["current_frame"].reshape(batch_size, -1)
            current_feat = self.current_net(current_flat)

        parts = [guide_curve_feat, stock_geometry_feat]

        if self.use_stock_mask:
            parts.append(stock_mask_feat)

        if self.use_stock_areas:
            parts.append(stock_areas_feat)

        parts += [current_feat, progress_feat]

        return self.combine_net(th.cat(parts, dim=1))