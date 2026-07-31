from gymnasium.envs.registration import register

register(
    id="environment/BikeBuilder-v0",
    entry_point="environment.envs:BikeBuilder_Env",
)