from myosuite.utils import gym

register = gym.register

register(
    id="walkLeg26Imitation-v0",
    entry_point="walk.envs.leg26_imitation:MyoAssistLeg26Imitation",
    max_episode_steps=1000,
    kwargs={},
)
