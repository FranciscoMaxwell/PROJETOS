# cart_dqn.py
# Carrinho que aprende sozinho com DQN (PyTorch + Pygame)
# Requer: torch, pygame, numpy

import os
import random
import math
import argparse
from collections import deque, namedtuple
import time

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Hyperparâmetros
# -----------------------
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 20000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000  # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20000  # steps
MAX_EPISODES = 2000
MAX_STEPS = 800
SAVE_EVERY = 5000  # steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Utils / Replay Buffer
# -----------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -----------------------
# DQN Network (MLP)
# -----------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# Environment (Pygame simple track)
# -----------------------
class SimpleTrackEnv:
    def __init__(self, screen_size=(640,480), render=True):
        self.width, self.height = screen_size
        self.render = render
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption("Carrinho RL")
            self.clock = pygame.time.Clock()
        # Track: rectangular with inner obstacles example
        self.track_rect = pygame.Rect(50, 50, self.width-100, self.height-100)
        # goal zone at far right
        self.goal_x = self.track_rect.right - 30
        self.reset()

    def reset(self):
        # start near left center inside track
        self.car_pos = np.array([self.track_rect.left + 40.0, self.track_rect.centery], dtype=float)
        self.car_angle = 0.0  # in degrees, 0 pointing right
        self.speed = 0.0
        self.max_speed = 5.0
        self.radius = 8  # for collision
        self.time = 0
        self.done = False
        # obstacles: list of rects
        self.obstacles = [
            pygame.Rect(self.width//2 - 60, self.height//2 - 100, 40, 200),
            pygame.Rect(self.width//2 + 40, self.height//2 - 200, 40, 120)
        ]
        state = self._get_state()
        return state

    def step(self, action):
        """
        Actions (discrete, 5):
        0: turn left + accelerate
        1: turn left
        2: straight
        3: turn right
        4: turn right + accelerate
        """
        # translate action into controls
        turn_speed = 6.0  # degrees per step
        accel = 0.4
        brake = 0.3

        if action == 0:
            self.car_angle -= turn_speed
            self.speed += accel
        elif action == 1:
            self.car_angle -= turn_speed
            self.speed += 0.0
        elif action == 2:
            self.speed += 0.05  # small forward bias
        elif action == 3:
            self.car_angle += turn_speed
            self.speed += 0.0
        elif action == 4:
            self.car_angle += turn_speed
            self.speed += accel

        # clamp speed
        self.speed = max(-1.0, min(self.speed, self.max_speed))
        # friction
        self.speed *= 0.995

        # update position
        rad = math.radians(self.car_angle)
        vx = math.cos(rad) * self.speed
        vy = math.sin(rad) * self.speed
        self.car_pos[0] += vx
        self.car_pos[1] += vy

        self.time += 1

        # compute reward
        reward = 0.0
        done = False

        # collision with boundaries
        if not self.track_rect.collidepoint(self.car_pos[0], self.car_pos[1]):
            reward = -100.0
            done = True
        # collision with obstacles
        car_rect = pygame.Rect(self.car_pos[0]-self.radius, self.car_pos[1]-self.radius, self.radius*2, self.radius*2)
        for ob in self.obstacles:
            if ob.colliderect(car_rect):
                reward = -100.0
                done = True
                break

        # success: reach goal area
        if self.car_pos[0] >= self.goal_x:
            reward = 200.0
            done = True

        # survival reward small
        if not done:
            reward += 1.0  # encourage longer survival / progress

        # optionally limit steps
        if self.time >= MAX_STEPS:
            done = True

        self.done = done
        next_state = self._get_state()
        return next_state, reward, done, {}

    def _get_state(self):
        # sensors: cast N rays and return normalized distances, plus speed and angle
        rays = 7
        max_dist = 120.0
        angles = np.linspace(-90, 90, rays) + self.car_angle  # relative to car angle
        dists = []
        for a in angles:
            rad = math.radians(a)
            dist = self._ray_distance(self.car_pos, rad, max_dist)
            dists.append(dist / max_dist)  # normalize 0-1
        # include speed normalized and angle normalized (-180,180 -> -1..1)
        speed_norm = (self.speed / self.max_speed)  # -1..1
        ang_norm = math.sin(math.radians(self.car_angle))  # encode angle as sin
        state = np.array(dists + [speed_norm, ang_norm], dtype=np.float32)
        return state

    def _ray_distance(self, pos, rad, max_d):
        # step sampling along ray, checking collisions with walls and obstacles
        x, y = pos
        step = 2.0
        dist = 0.0
        while dist < max_d:
            nx = x + math.cos(rad) * dist
            ny = y + math.sin(rad) * dist
            # outside bounds -> stop
            if not self.track_rect.collidepoint(nx, ny):
                return dist
            # obstacles
            p = (nx, ny)
            for ob in self.obstacles:
                if ob.collidepoint(p):
                    return dist
            dist += step
        return max_d

    def render_env(self):
        if not self.render:
            return
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit()

        self.screen.fill((30,30,30))
        # draw track area
        pygame.draw.rect(self.screen, (50,50,50), self.track_rect)
        # draw goal zone
        pygame.draw.rect(self.screen, (30,120,30), (self.goal_x, self.track_rect.top, self.track_rect.right-self.goal_x, self.track_rect.height))
        # draw obstacles
        for ob in self.obstacles:
            pygame.draw.rect(self.screen, (150,50,50), ob)
        # draw car
        x, y = int(self.car_pos[0]), int(self.car_pos[1])
        pygame.draw.circle(self.screen, (200,200,50), (x,y), self.radius)
        # draw heading line
        rad = math.radians(self.car_angle)
        hx = int(x + math.cos(rad) * self.radius * 2.5)
        hy = int(y + math.sin(rad) * self.radius * 2.5)
        pygame.draw.line(self.screen, (255,0,0), (x,y), (hx,hy), 2)
        # draw rays
        rays = 7
        max_dist = 120.0
        angles = np.linspace(-90, 90, rays) + self.car_angle
        for a in angles:
            rad = math.radians(a)
            dist = self._ray_distance(self.car_pos, rad, max_dist)
            ex = int(x + math.cos(rad) * dist)
            ey = int(y + math.sin(rad) * dist)
            pygame.draw.line(self.screen, (100,200,200), (x,y), (ex,ey), 1)

        pygame.display.flip()
        self.clock.tick(60)

# -----------------------
# Training loop
# -----------------------
def train(render=True, resume=False, model_path="models/best_model.pth"):
    env = SimpleTrackEnv(render=render)
    state_dim = len(env._get_state())
    n_actions = 5

    policy_net = DQN(state_dim, n_actions).to(DEVICE)
    target_net = DQN(state_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(BUFFER_SIZE)

    # prepare replay with random transitions
    print("[INFO] Populating replay buffer with random actions...")
    while len(replay) < MIN_REPLAY_SIZE:
        s = env.reset()
        done = False
        while not done and len(replay) < MIN_REPLAY_SIZE:
            a = random.randrange(n_actions)
            ns, r, done, _ = env.step(a)
            replay.push(s, a, r, ns, done)
            s = ns
    print(f"[INFO] Replay buffer size: {len(replay)}")

    steps_done = 0
    best_reward = -1e9
    total_steps = 0

    # epsilon schedule
    def get_epsilon(step):
        return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)

    for ep in range(1, MAX_EPISODES+1):
        state = env.reset()
        ep_reward = 0.0
        for t in range(MAX_STEPS):
            steps_done += 1
            total_steps += 1
            eps = get_epsilon(steps_done)
            if random.random() < eps:
                action = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
                    qvals = policy_net(s_t)
                    action = int(qvals.argmax().item())

            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # sample and train
            if len(replay) >= BATCH_SIZE:
                batch = replay.sample(BATCH_SIZE)
                states = torch.tensor(np.array(batch.state), device=DEVICE, dtype=torch.float32)
                actions = torch.tensor(batch.action, device=DEVICE, dtype=torch.long).unsqueeze(1)
                rewards = torch.tensor(batch.reward, device=DEVICE, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(np.array(batch.next_state), device=DEVICE, dtype=torch.float32)
                dones = torch.tensor(batch.done, device=DEVICE, dtype=torch.float32).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + (1 - dones) * GAMMA * next_q

                loss = nn.functional.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # save periodically
            if total_steps % SAVE_EVERY == 0:
                os.makedirs("models", exist_ok=True)
                torch.save(policy_net.state_dict(), "models/last_model.pth")
                print(f"[INFO] Saved checkpoint at step {total_steps}")

            if render:
                try:
                    env.render_env()
                except SystemExit:
                    pygame.quit()
                    return

            if done:
                break

        print(f"[EP {ep}] Reward: {ep_reward:.1f} | Eps: {eps:.3f} | Steps: {t+1}")
        # save best
        if ep_reward > best_reward:
            best_reward = ep_reward
            os.makedirs("models", exist_ok=True)
            torch.save(policy_net.state_dict(), model_path)
            print(f"[INFO] New best model saved with reward {best_reward:.1f}")

    # final save
    torch.save(policy_net.state_dict(), "models/last_model.pth")
    print("[INFO] Training finished.")

# -----------------------
# Play (load model and run)
# -----------------------
def play(model_path="models/best_model.pth"):
    env = SimpleTrackEnv(render=True)
    state_dim = len(env._get_state())
    n_actions = 5
    model = DQN(state_dim, n_actions).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    while True:
        s = env.reset()
        done = False
        total = 0
        while not done:
            with torch.no_grad():
                s_t = torch.tensor(s, device=DEVICE, dtype=torch.float32).unsqueeze(0)
                a = int(model(s_t).argmax().item())
            s, r, done, _ = env.step(a)
            total += r
            try:
                env.render_env()
            except SystemExit:
                pygame.quit()
                return
        print(f"[PLAY] Episode finished, total reward: {total}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train agent")
    parser.add_argument("--play", action="store_true", help="Play using saved model")
    parser.add_argument("--no-render", action="store_true", help="Don't render (headless training faster)")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Model path")
    args = parser.parse_args()

    if args.train:
        train(render=(not args.no_render), model_path=args.model)
    elif args.play:
        if not os.path.exists(args.model):
            print("Model not found. Train first.")
        else:
            play(model_path=args.model)
    else:
        print("Use --train or --play")
