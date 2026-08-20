import torch
from stable_baselines3 import PPO
from MaskedPolicy import MaskedMultiInputActorCriticPolicy
from PointNet_Extractor import PointNet_Extractor

policy_kwargs = dict(
    features_extractor_class  = PointNet_Extractor,
    features_extractor_kwargs = dict(features_dim=256),
)

model_a = PPO("MultiInputPolicy", train_env, seed=seed, policy_kwargs=policy_kwargs)
model_b = PPO(MaskedMultiInputActorCriticPolicy, train_env, seed=seed,
              policy_kwargs=dict(**policy_kwargs, use_masking=False))

for (na, pa), (nb, pb) in zip(model_a.policy.named_parameters(), model_b.policy.named_parameters()):
    if not torch.equal(pa, pb):
        print(f"First divergence: {na}")
        break
else:
    print("Initial weights identical.")