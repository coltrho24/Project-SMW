# -*- coding: utf-8 -*-
from collections import deque
import random
import cv2
import torch
import stable_retro


class Env():
  def __init__(self, args):
    self.device = args.device
    self.env = stable_retro.make(
      game='SuperMarioWorld-Snes-v0',
      state=getattr(args, 'smw_state', stable_retro.State.DEFAULT),
      render_mode='human'
    )
    # Map integer actions to MultiBinary button combos
    # Button order: [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]
    #                 0  1    2      3    4    5     6     7    8  9  10 11
    self._actions = [
      [0,0,0,0,0,0,0,0,0,0,0,0],  # no-op
      [0,0,0,0,0,0,0,1,0,0,0,0],  # right
      [0,0,0,0,0,0,0,1,1,0,0,0],  # right + A (jump)
      [0,0,0,0,0,0,0,1,0,0,0,0],  # right + Y (run)
      [0,0,0,0,0,0,0,1,1,0,0,0],  # right + A + Y (run-jump)
      [0,0,0,0,0,0,0,0,1,0,0,0],  # A only (jump in place)
      [0,0,0,0,0,0,1,0,0,0,0,0],  # left
      [0,0,0,0,0,0,1,0,1,0,0,0],  # left + A (jump left)
      [0,0,0,0,0,0,1,0,0,1,0,0],  # left + Y (run left)
    ]
    self.lives = 0
    self.life_termination = False
    self.window = args.history_length
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True
    self._max_episode_length = args.max_episode_length
    self._step_count = 0
    self._x_pos = 0
    self._max_x_pos = 0  # Furthest x reached this episode
    self._game_mode = 20  # Normal gameplay mode

  def _get_state(self):
    # Use the observation directly — calling render() with render_mode=None can block
    gray = cv2.cvtColor(self._last_obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(resized, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def _get_lives(self):
    info = self.env.data.lookup_all()
    return int(info.get('lives', 4))

  def _get_x(self, info):
    return int(info.get('screen', 0)) * 256 + int(info.get('x', 0))

  def reset(self):
    self.life_termination = False
    self._reset_buffer()
    obs, info = self.env.reset()
    self._last_obs = obs
    self._step_count = 0
    for _ in range(random.randrange(30)):
      obs, _, terminated, truncated, info = self.env.step(self._actions[0])
      self._last_obs = obs
      if terminated or truncated:
        obs, info = self.env.reset()
        self._last_obs = obs
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self._get_lives()
    self._x_pos = self._get_x(info)
    self._max_x_pos = self._x_pos
    self._game_mode = 20
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
    buttons = self._actions[action]
    prev_x = self._x_pos
    last_info = {}
    for t in range(4):
      obs, _, terminated, truncated, last_info = self.env.step(buttons)
      self._last_obs = obs
      self._step_count += 1
      new_game_mode = int(self.env.get_ram()[0x100])
      if new_game_mode in (18, 19) and self._game_mode == 20:
        terminated = True
      self._game_mode = new_game_mode
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = terminated or truncated or (self._step_count >= self._max_episode_length)
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Reward shaping: new-ground bonus + small rightward movement bonus - time penalty
    new_x = self._get_x(last_info)
    new_ground = max(0, new_x - self._max_x_pos)          # reward for reaching new furthest-right position
    move_right = max(0, new_x - self._x_pos) * 0.1        # small bonus for any rightward movement
    time_penalty = -0.05                                    # per-step cost to discourage oscillation
    reward = new_ground + move_right + time_penalty
    self._x_pos = new_x
    self._max_x_pos = max(self._max_x_pos, new_x)
    # Penalize overworld entry (detected inside the frame loop above)
    if self._game_mode in (18, 19):
      done = True
      reward -= 3.0
    # Always check lives — in eval mode too — to avoid game over animation hanging the emulator
    lives = self._get_lives()
    if lives < self.lives:
      reward -= 5.0  # Death penalty
      if lives <= 0:
        # Game over imminent: force terminal before the game over screen can hang
        done = True
        self.life_termination = False
      elif self.training:
        self.life_termination = not done
        done = True
    self.lives = lives
    return torch.stack(list(self.state_buffer), 0), reward, done

  def train(self):
    self.training = True

  def eval(self):
    self.training = False

  def action_space(self):
    return len(self._actions)

  def render(self):
    frame = self.env.render()
    if frame is not None:
      cv2.imshow('screen', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
      cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
    self.env.close()
