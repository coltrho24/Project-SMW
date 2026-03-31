import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import stable_retro
import cv2
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

def compute_reward(prev_info, curr_info):
    reward = 0.0

    # --- Progress ---
    curr_pos = curr_info['x']
    prev_pos = prev_info['x']
    delta_x = curr_pos - prev_pos
    if delta_x > 0:
        reward += delta_x * 0.01

    # --- Milestones ---
    if curr_info.get('x', 0) > 2565 and prev_info.get('x', 0) <= 2565:
        reward += 250.0
    if curr_info.get('x', 0) > 3400 and prev_info.get('x', 0) <= 3400:
        reward += 250.0
    if curr_info.get('x', 0) > 3900 and prev_info.get('x', 0) <= 3900:
        reward += 500.0
    if curr_info.get('game_mode', 20) == 12 and prev_info.get('game_mode', 20) == 20:
        reward += 1000.0

    reward -= 0.0001

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
    [0,1,0,0,0,0,1,0,0,0,0,0],  # left + Y (run)
    # [1,1,0,0,0,0,1,0,0,0,0,0],  # left + B + Y (run jump)
    [0,0,0,0,1,0,0,0,0,0,0,0],  # up
    [0,0,0,0,0,1,0,0,0,0,0,0],  # down
]

NUM_STACK = 4  # number of frames to stack

class SMWEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = ACTIONS
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, NUM_STACK),
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.prev_info = {}
        self.stuck_counter = 0
        self.last_x = 0
        self.frame_stack = None  # will hold (84, 84, NUM_STACK)

    def preprocess(self, frame):
        # Grayscale + resize using cv2 (much faster than PIL)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_LINEAR)
        return resized  # (84, 84)if curr_info.get('midway_flag', 0) > prev_info.get('midway_flag', 0):

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        self.prev_info = {'x': 0, 'midway_flag': 0, 'lives': 4, 'game_mode': 20}
        self.stuck_counter = 0
        self.last_x = 0
        # Fill stack with copies of the first frame
        processed = self.preprocess(frame)
        self.frame_stack = np.stack([processed] * NUM_STACK, axis=-1)  # (84, 84, 4)
        return self.frame_stack, info

    def step(self, action):
        actual_action = self.actions[action]

        for _ in range(4):
            frame, _, terminated, truncated, info = self.env.step(actual_action)
            if terminated or truncated:
                break

        # Shift stack left and add new frame on the right
        processed = self.preprocess(frame)
        self.frame_stack = np.concatenate([self.frame_stack[:, :, 1:], processed[:, :, np.newaxis]], axis=-1)
        obs = self.frame_stack
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


class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # Shared CNN backbone (input: 84x84x4)
        self.cnn = nn.Sequential(
            nn.Conv2d(NUM_STACK, 32, kernel_size=8, stride=4),  # -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),         # -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),         # -> (64, 7, 7)
            nn.ReLU(),
            nn.Flatten()                                         # -> 3136
        )
        cnn_out = 64 * 7 * 7  # = 3136
        self.actor_head  = nn.Sequential(nn.Linear(cnn_out, 512), nn.ReLU(), nn.Linear(512, action_dim))
        self.critic_head = nn.Sequential(nn.Linear(cnn_out, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        # x: (batch, H, W, C) -> (batch, C, H, W), normalized to [0, 1]
        x = x.permute(0, 3, 1, 2).float() / 255.0
        features = self.cnn(x)
        probs = F.softmax(self.actor_head(features), dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic_head(features)
        return action, log_prob, dist, value