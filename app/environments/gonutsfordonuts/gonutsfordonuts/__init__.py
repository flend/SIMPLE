from gym.envs.registration import register

register(
    id='GoNutsForDonuts-v0',
    entry_point='gonutsfordonuts.envs:GoNutsForDonutsEnv',
)

