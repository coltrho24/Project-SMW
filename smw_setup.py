import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import stable_retro
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

def compute_reward(prev_info, curr_info):
    reward = 0.0

    # --- Progress ---
    curr_pos = curr_info['x']
    prev_pos = prev_info['x']
    delta_x = curr_pos - prev_pos
    if 0 < delta_x < 200:
        reward += delta_x * 0.01

    # --- Timer penalty ---
    # curr_time = curr_info.get('timer_h', 0) * 100 + curr_info.get('timer_t', 0) * 10 + curr_info.get('timer_o', 0)
    # prev_time = prev_info.get('timer_h', 0) * 100 + prev_info.get('timer_t', 0) * 10 + prev_info.get('timer_o', 0)

    # --- Milestones ---
    if curr_info.get('midway_flag', 0) > prev_info.get('midway_flag', 0):
        reward += 100.0
    if curr_info.get('game_mode', 20) == 12 and prev_info.get('game_mode', 20) == 20:
        reward += 500.0

    # --- Powerups ---
    if curr_info.get('powerup_status', 0) > prev_info.get('powerup_status', 0):
        reward += 1.0

    # --- Penalize jumping near koopa wall ---
    # if 350 < curr_info.get('x', 0) < 500:
    #     if curr_info.get('y', 0) < prev_info.get('y', 0):
    #         reward -= 5

    # --- Penalize jumping ---
    # if curr_info.get('y', 0) < prev_info.get('y', 0):
    #     reward -= 0.05

    return reward

ACTIONS = [
    [0,0,0,0,0,0,0,0,0,0,0,0],  # nothing
    [0,0,0,0,0,0,0,1,0,0,0,0],  # right
    [1,0,0,0,0,0,0,1,0,0,0,0],  # right + B (jump)
    [0,1,0,0,0,0,0,1,0,0,0,0],  # right + Y (run)
    [1,1,0,0,0,0,0,1,0,0,0,0],  # right + B + Y (run jump)
    [1,0,0,0,0,0,0,0,0,0,0,0],  # B (jump in place)
    [0,0,0,0,0,0,1,0,0,0,0,0],  # left
    [1,0,0,0,0,0,1,0,0,0,0,0],  # left + B (jump left)
    [0,0,0,0,1,0,0,0,0,0,0,0],  # up
    [0,0,0,0,0,1,0,0,0,0,0,0],  # down
    [0,0,0,0,0,0,0,0,0,0,0,0],  # nothing
    [0,0,0,0,0,0,0,0,0,0,0,0],  # nothing
]

class SMWEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = ACTIONS
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1,  1,  1], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.prev_info = {}
        self.stuck_counter = 0
        self.last_x = 0

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.prev_info = {'x': 0, 'screen': 0, 'powerup_status': 0, 'riding_yoshi': 0, 'midpoint_flag': 0, 'lives': 4, 'score': 0, 'coins': 0}
        self.stuck_counter = 0
        self.last_x = 0
        obs = self.get_obs(info)
        return obs, info

    def step(self, action):
        actual_action = self.actions[action]
        _, _, terminated, truncated, info = self.env.step(actual_action)
        obs = self.get_obs(info)
        reward = compute_reward(self.prev_info, info)
        curr_x = info.get('x', 0)

        curr_time = info.get('timer_h', 0) * 100 + info.get('timer_t', 0) * 10 + info.get('timer_o', 0)
        if curr_time == 0:
            terminated = True
            reward -= 5.0
        if info.get('game_mode', 20) == 11 and self.prev_info.get('game_mode', 20) == 20:
            terminated = True
            reward -= 5.0
        if info.get('game_mode', 20) == 12 and self.prev_info.get('game_mode', 20) == 20:
            terminated = True
            reward += curr_time * 0.1  # time bonus only on level completion

        if curr_x == self.last_x:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_x = curr_x

        if self.stuck_counter > 200:
            terminated = True
            reward -= 5.0
            self.stuck_counter = 0

        self.prev_info = info
        return obs, reward, terminated, truncated, info

    def get_obs(self, info):
        curr_time = info.get('timer_h', 0) * 100 + info.get('timer_t', 0) * 10 + info.get('timer_o', 0)
        mario_x = info.get('x', 0)

        sprite0_dist = (info.get('sprite0_x', 0) - mario_x) / 255.0
        sprite1_dist = (info.get('sprite1_x', 0) - mario_x) / 255.0

        return np.array([
            info.get('screen', 0) / 255.0,
            mario_x / 5300.0,
            info.get('powerup_status', 0) / 3.0,
            info.get('riding_yoshi', 0),
            info.get('midpoint_flag', 0),
            info.get('y', 0) / 255.0,
            curr_time / 300.0,
            sprite0_dist,
            sprite1_dist,
        ], dtype=np.float32)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.fc3(x), dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)