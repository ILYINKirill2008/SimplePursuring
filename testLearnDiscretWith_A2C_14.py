
import numpy as np
import torch

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, A2C, PPO, DDPG, TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn.policies import DQNPolicy, CnnPolicy, MultiInputPolicy

import SimplePursuringDiscreteEnv7 as spEnv

#-----------------------------------------------------------------------------

discount_factor = 0.95

envS = spEnv.SimplePursuringDiscreteEn(True)
envS.verbose = 1

envS.ShappingGamma = discount_factor
envS.VTgt = 0.004
#MaxAbsAction = float(0.3);
envS.FireReward = float(1)

env = Monitor(envS)
check_env(env, warn=True)

model = A2C("MlpPolicy", env, verbose=1, gamma=discount_factor, 
            learning_rate=0.001,
            max_grad_norm=3,
            tensorboard_log="TensorboardLog"
            )

model.learn(total_timesteps=200000000, log_interval=1,)

for q in range(10):
    obs = env.reset()
    while True:
         action, _states = model.predict(obs, deterministic=True)
         obs, reward, done, info = env.step(action)
         if done:
             break