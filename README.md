# STAR: Enhancing Visual Reinforcement Learning with State-Action Representation

This is an original PyTorch implementation of STAR from [Enhancing Visual Reinforcement Learning with State-Action Representation]

## Method

STAR is a simple visual RL framework which introduces a state-action joint representation considering the interaction between states and actions,
and uses it to enhance the input of the 𝑄-functions. The representation learning process includes two encoders. The state encoder extracts
features from the raw images, while the state-action encoder takes these features and the action as input to output state-action embedding. During the training of the value functions, the parameters of the state-action encoder remain static, and the output embedding serves as an additional input for the value function.

<p align="center">
  <img src='fig/overview.png' width="750"/>
</p>
