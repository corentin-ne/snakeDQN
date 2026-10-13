"""Pygame viewer: watch the trained agent or play yourself.

    python play.py                 # watch snake_dqn.pth
    python play.py --human         # arrow keys
"""
import argparse

import numpy as np
import pygame
import torch

from dqn import Agent
from snake_env import N_ACTIONS, N_FEATURES, SnakeEnv

CELL, HEADER = 32, 48
LCD, INK, FOOD, HOT = (117, 162, 89), (34, 49, 29), (194, 51, 43), (242, 193, 78)
KEYS = {pygame.K_UP: 0, pygame.K_RIGHT: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--human", action="store_true")
    p.add_argument("--model", default="snake_dqn.pth")
    p.add_argument("--size", type=int, default=12)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    agent = None
    if not args.human:
        agent = Agent(N_FEATURES, N_ACTIONS)
        agent.load(args.model)
        agent.net.eval()

    pygame.init()
    env = SnakeEnv(args.size, args.size)
    side = args.size * CELL
    screen = pygame.display.set_mode((side, side + HEADER))
    pygame.display.set_caption("Snake DQN")
    font = pygame.font.Font("assets/PressStart2P-Regular.ttf", 14)
    clock = pygame.time.Clock()
    state, best, want = env.reset(), 0, None

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
            if e.type == pygame.KEYDOWN and e.key in KEYS:
                want = KEYS[e.key]

        if agent:
            with torch.no_grad():
                q = agent.net(torch.as_tensor(state[None]))[0].numpy()
            action = int(q.argmax())
        else:
            q = np.zeros(N_ACTIONS)
            diff = (want - env.dir) % 4 if want is not None else 0
            action = {0: 0, 1: 1, 3: 2}.get(diff, 0)

        state, _, done = env.step(action)
        if done:
            best = max(best, env.score)
            state = env.reset()
            want = None

        screen.fill(INK)
        pygame.draw.rect(screen, LCD, (0, HEADER, side, side))
        if env.food is not None:
            fx, fy = env.food
            pygame.draw.circle(screen, FOOD, (fx * CELL + CELL // 2, HEADER + fy * CELL + CELL // 2), CELL // 3)
        for i, (x, y) in enumerate(env.body):
            pad = 2 if i == 0 else 4
            pygame.draw.rect(screen, INK, (x * CELL + pad, HEADER + y * CELL + pad, CELL - 2 * pad, CELL - 2 * pad))
        screen.blit(font.render(f"SCORE {env.score}  BEST {best}", True, LCD), (12, 16))
        if agent:
            for i, v in enumerate(q):
                c = HOT if i == action else LCD
                screen.blit(pygame.font.Font(None, 18).render(f"{'SRL'[i]}{v:5.1f}", True, c), (side - 150 + i * 50, 30))
        pygame.display.flip()
        clock.tick(args.fps)


if __name__ == "__main__":
    main()
