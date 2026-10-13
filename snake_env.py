"""Headless, allocation-light Snake environment.

The state is a compact 20-feature vector (relative danger, free run length,
flood-fill reachable area, heading, food and tail direction, length). The web
version in web/index.html mirrors this file exactly so trained weights transfer.
"""
from collections import deque

import numpy as np

# Clockwise: up, right, down, left
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
N_ACTIONS = 3  # straight, turn right, turn left
N_FEATURES = 20
TURN = (0, 1, 3)  # relative action -> direction offset

R_FOOD, R_DEATH, R_CLOSER, R_FARTHER = 10.0, -10.0, 0.05, -0.07


class SnakeEnv:
    def __init__(self, width=12, height=12, seed=None):
        self.w, self.h = width, height
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((height, width), dtype=np.int8)  # 1 = body
        self.reset()

    def reset(self):
        self.grid.fill(0)
        cx, cy = self.w // 2, self.h // 2
        self.body = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])  # head first
        for x, y in self.body:
            self.grid[y, x] = 1
        self.dir = 1
        self.score = 0
        self.steps = 0
        self.hunger = 0
        self._place_food()
        return self.features()

    def _place_food(self):
        free = np.flatnonzero(self.grid.ravel() == 0)
        if free.size == 0:
            self.food = None
            return
        i = free[self.rng.integers(free.size)]
        self.food = (int(i % self.w), int(i // self.w))

    def _blocked(self, x, y, tail):
        return not (0 <= x < self.w and 0 <= y < self.h) or (self.grid[y, x] and (x, y) != tail)

    def _flood(self, sx, sy, tail):
        """Cells reachable from (sx, sy) once the head has moved there."""
        g, w, h = self.grid, self.w, self.h
        seen = np.zeros_like(g, dtype=bool)
        seen[sy, sx] = True
        if tail is not None:
            g[tail[1], tail[0]] = 0
        q, n = [(sx, sy)], 1
        while q:
            x, y = q.pop()
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] and not g[ny, nx]:
                    seen[ny, nx] = True
                    n += 1
                    q.append((nx, ny))
        if tail is not None:
            g[tail[1], tail[0]] = 1
        return n

    def features(self):
        hx, hy = self.body[0]
        tail = self.body[-1]
        f = np.zeros(N_FEATURES, dtype=np.float32)
        size = self.w * self.h
        span = max(self.w, self.h)
        for i, t in enumerate(TURN):
            dx, dy = DIRS[(self.dir + t) % 4]
            nx, ny = hx + dx, hy + dy
            if self._blocked(nx, ny, tail):
                f[i] = 1.0
            else:
                f[6 + i] = self._flood(nx, ny, tail) / size
            run, x, y = 0, nx, ny
            while not self._blocked(x, y, tail):
                run += 1
                x += dx
                y += dy
            f[3 + i] = run / span
        f[9 + self.dir] = 1.0
        if self.food is not None:
            fx, fy = self.food
            f[13], f[14], f[15], f[16] = fy < hy, fx > hx, fy > hy, fx < hx
        f[17] = (tail[0] - hx) / self.w
        f[18] = (tail[1] - hy) / self.h
        f[19] = len(self.body) / size
        return f

    def step(self, action):
        """Returns (features, reward, done)."""
        self.dir = (self.dir + TURN[action]) % 4
        dx, dy = DIRS[self.dir]
        hx, hy = self.body[0]
        nx, ny = hx + dx, hy + dy
        tail = self.body[-1]
        self.steps += 1
        self.hunger += 1

        eat = (nx, ny) == self.food
        if self._blocked(nx, ny, None if eat else tail):
            return self.features(), R_DEATH, True

        if not eat:
            self.body.pop()
            self.grid[tail[1], tail[0]] = 0
        self.body.appendleft((nx, ny))
        self.grid[ny, nx] = 1

        if eat:
            self.score += 1
            self.hunger = 0
            self._place_food()
            if self.food is None:  # board filled: perfect game
                return self.features(), R_FOOD, True
            return self.features(), R_FOOD, False

        if self.hunger > self.w * self.h:
            return self.features(), R_DEATH, True

        fx, fy = self.food
        closer = abs(nx - fx) + abs(ny - fy) < abs(hx - fx) + abs(hy - fy)
        return self.features(), R_CLOSER if closer else R_FARTHER, False
