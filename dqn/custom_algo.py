"""
This module contains the ModifiedDQN class, which subclasses the DQN algorithm.
"""

from typing import Optional, Type

from ray.rllib.algorithms.algorithm import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQN as OriginalDQN
from ray.rllib.algorithms.dqn.dqn import DQNConfig as OriginalDQNCOnfig
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override

from dqn.custom_policy import DQNTorchPolicy


class DQNConfig(OriginalDQNCOnfig):
    """
    Modified DQNConfig class that points to the ModifiedDQN algo class.
    """

    @override(OriginalDQNCOnfig)
    def __init__(self):
        super().__init__()
        self.algo_class = DQN


class DQN(OriginalDQN):
    """
    Modified DQN class that returns the custom DQNTorchPolicy as default.
    """

    @classmethod
    @override(OriginalDQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return DQNConfig()

    @classmethod
    @override(OriginalDQN)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        return DQNTorchPolicy
