import torch as th
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution


class MaskablePolicy(MultiInputActorCriticPolicy):
    """
    forward() and evaluate_actions() are inherited untouched — both already
    reach extract_features() and _get_action_dist_from_latent() via `self.`
    calls, so they're routed through the two overrides below for free.
    """

    _current_obs: dict[str, th.Tensor]

    def __init__(self, *args, use_masking: bool = True, **kwargs):
        super().__init__(*args, **kwargs)      # --- unchanged: identical RNG draws to vanilla
        self.use_masking = use_masking          # +++ NEW: bookkeeping only, no RNG involved

    def extract_features(self, obs, features_extractor=None):
        self._current_obs = obs          # type: ignore       # +++ NEW: only line added
        return super().extract_features(obs, features_extractor)   # --- unchanged

    def get_distribution(self, obs):
        self._current_obs = obs          # type: ignore        # +++ NEW: needed separately — see the super()
        return super().get_distribution(obs)     #     bypass in the real source above

    def _get_action_dist_from_latent(self, latent_pi):
        if not self.use_masking:
            return super()._get_action_dist_from_latent(latent_pi)   # --- unchanged fallback

        if not isinstance(self.action_dist, MultiCategoricalDistribution):
            raise TypeError("Masking only implemented for MultiDiscrete action spaces.")

        mean_actions = self.action_net(latent_pi)                    # +++ NEW below this line
        mask = self._mask_from_obs(self._current_obs)
        mean_actions = mean_actions.masked_fill(~mask, -1e8)
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def _mask_from_obs(self, obs: dict) -> th.Tensor:
        frame_available = obs["stock_mask"].bool()   # <-- confirm this key against your _get_obs()
        batch  = frame_available.shape[0]
        device = frame_available.device
        open_heads = th.ones(batch, 12, dtype=th.bool, device=device)  # 5 tar + 5 cand + 2 mirror
        return th.cat([frame_available, open_heads], dim=1)