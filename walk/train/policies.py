from typing import Callable, Optional, Tuple

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn

from walk.train.config import TrainSessionConfigBase, ImitationTrainSessionConfig
from walk.utils.data_types import DictionableDataclass


class HumanPPOCustomNetwork(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        custom_policy_params,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.net_arch = custom_policy_params.net_arch
        self.net_indexing_info = custom_policy_params.net_indexing_info

        self.reset_policy_networks()
        self.reset_value_network()

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(obs), self.forward_critic(obs)

    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        return self.policy_net(obs)

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        return self.value_net(obs)

    def reset_policy_networks(self):
        layers = []
        last_dim = self.observation_space.shape[0]
        for dim in self.net_arch.get("human_actor", [64, 64]):
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim
        layers.append(nn.Linear(last_dim, self.action_space.shape[0]))
        layers.append(nn.Tanh())
        self.policy_net = nn.Sequential(*layers)

    def reset_value_network(self):
        layers = []
        last_dim = self.observation_space.shape[0]
        for dim in self.net_arch.get("common_critic", [128, 128]):
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim
        layers.append(nn.Linear(last_dim, 1))
        self.value_net = nn.Sequential(*layers)


class HumanActorCriticPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        use_sde: bool = False,
        *args,
        **kwargs,
    ):
        custom_policy_params_dict = kwargs.pop("custom_policy_params", None)
        custom_policy_params = DictionableDataclass.create(
            ImitationTrainSessionConfig.PolicyParams.CustomPolicyParams,
            custom_policy_params_dict,
        )

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.policy_network = HumanPPOCustomNetwork(
            observation_space, action_space, custom_policy_params
        )

        self.action_dist = DiagGaussianDistribution(action_space)
        self.log_std = nn.Parameter(
            th.ones(self.action_space.shape[0], device=self.device)
            * custom_policy_params.log_std_init,
            requires_grad=True,
        )
        self.apply(self.init_weights)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        mean_actions = self.policy_network.forward_actor(obs)
        value = self.policy_network.forward_critic(obs)

        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, value, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        mean_actions, value = self.policy_network(obs)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return value, log_prob, entropy

    def get_distribution(self, obs: th.Tensor) -> DiagGaussianDistribution:
        mean_actions = self.policy_network.forward_actor(obs)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        return self.policy_network.forward_critic(obs)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def reset_network(
        self,
        reset_shared_net: bool = False,
        reset_policy_net: bool = False,
        reset_value_net: bool = False,
    ):
        if reset_policy_net:
            self.policy_network.reset_policy_networks()
        if reset_value_net:
            self.policy_network.reset_value_network()
