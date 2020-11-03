# SoftActorCritic-Pendulum

Brief, quick and dirty SAC implementation using Tensorflow `2.x`. The entropy coeficient `alpha = 0.2` is fixed and not learned. A solution for the `Pendulum-v0` should be found after about 30 episodes / 6000 steps. 

## Dependencies

```
gym
numpy
tensorflow (2.x)
tensorflow_probability
```

## Tensorboard

The episode returns are also logged into a `log` directory inside the working directory for usage with tensorboard. 
```
tensorboard --logdir logs
```
