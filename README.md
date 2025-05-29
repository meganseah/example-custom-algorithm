# Custom Algorithm for RLSuite

This is an example custom algorithm repository, adapted from
https://github.com/ray-project/ray/tree/master/rllib/algorithms/dqn. This
repository is used as a test case for RLSuite to verify that RLSuite can run
custom algorithms.

## How to use

1. Install the custom algorithm by running `poetry install` in the
   directory containing this repository.
2. Run the custom algorithm with RLSuite by specifying the algorithm name
   `dqn.custom_algo.DQN` in your configuration file.

   
