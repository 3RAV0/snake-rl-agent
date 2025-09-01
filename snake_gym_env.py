import math
import random
from collections import deque
from typing import Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

Moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    def __init__(self, board: int = 16, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.board = int(board)
        self.render_mode = render_mode
        self._rnd = random.Random()
        if seed is not None:
            self._rnd.seed(seed)

        self.observation_space = spaces.Discrete(96)
        self.action_space = spaces.Discrete(3)

        self.body: deque = deque()
        self.heading: int = 1
        self.food: Tuple[int, int] = (0, 0)
        self.points = 0
        self.ticks = 0
        self._max_ticks = self.board * self.board * 4

        self._pg = None
        self._surf = None
        self._clk = None
        self._cell = 40
        self._bg = (30, 30, 30)
        self._clr_body = (0, 255, 255)
        self._clr_head = (0, 120, 120)
        self._clr_food = (250, 0, 0)
        self._clr_grid = (50, 50, 50)

        self._stuck = 0
        self._last_gap = None
        self._stall = 20

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rnd.seed(seed)
        self.body.clear()
        center = (self.board // 2, self.board // 2)
        self.heading = 1
        self.body.append(center)
        self.body.append((center[0], center[1] - 1))
        self.body.append((center[0], center[1] - 2))
        self.points = 0
        self.ticks = 0
        self._spawn_food()
        obs = self._make_state()
        info = {"score": self.points}
        if self.render_mode == "human":
            self._init_pg()
            self.render()

        head = self.body[0]
        self._last_gap = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        self._stuck = 0

        return obs, info

    def step(self, move: int):
        assert self.action_space.contains(move)
        self._turn(move)
        head = self.body[0]
        dr, dc = Moves[self.heading]
        nxt = (head[0] + dr, head[1] + dc)

        prev_gap = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        reward = -0.02
        end = False
        cut = False

        grow = (nxt == self.food)
        if self._hit(nxt, will_grow=grow):
            reward += -10.0
            end = True
            obs = self._make_state()
            info = {"score": self.points, "reason": "collision"}
            if self.render_mode == "human":
                self.render()
            return obs, reward, end, cut, info

        self.body.appendleft(nxt)
        if nxt == self.food:
            self.points += 1
            reward += 10.0
            self._spawn_food()
        else:
            self.body.pop()

        new_head = self.body[0]
        new_gap = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        if new_gap < prev_gap:
            reward += 0.15
            self._stuck = 0
        elif new_gap > prev_gap:
            reward += -0.15
            self._stuck = 0
        else:
            self._stuck += 1

        fwd = Moves[self.heading]
        vec = (self.food[0] - new_head[0], self.food[1] - new_head[1])
        align = 0
        if abs(vec[0]) > abs(vec[1]):
            if vec[0] < 0 and fwd == (-1, 0): align = 1
            elif vec[0] > 0 and fwd == (1, 0): align = 1
            elif fwd in [(0, 1), (0, -1)]: align = -1
        else:
            if vec[1] < 0 and fwd == (0, -1): align = 1
            elif vec[1] > 0 and fwd == (0, 1): align = 1
            elif fwd in [(-1, 0), (1, 0)]: align = -1
        reward += 0.03 * align

        if self._stuck >= self._stall:
            reward += -1.0
            end = True

        self.ticks += 1
        if self.ticks >= self._max_ticks:
            cut = True

        obs = self._make_state()
        info = {"score": self.points}
        if self.render_mode == "human":
            self.render()
        return obs, reward, end, cut, info

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode == "rgb_array":
            img = np.zeros((self.board, self.board, 3), dtype=np.uint8)
            img[:, :] = np.array(self._bg, dtype=np.uint8)
            fr, fc = self.food
            img[fr, fc] = self._clr_food
            for idx, (r, c) in enumerate(self.body):
                img[r, c] = self._clr_head if idx == 0 else self._clr_body
            return img
        if self.render_mode == "human":
            self._init_pg()
            import pygame
            self._surf.fill(self._bg)
            for i in range(self.board + 1):
                pygame.draw.line(self._surf, self._clr_grid, (0, i * self._cell),
                                 (self.board * self._cell, i * self._cell), 1)
                pygame.draw.line(self._surf, self._clr_grid, (i * self._cell, 0),
                                 (i * self._cell, self.board * self._cell), 1)
            self._draw(self.food, self._clr_food)
            for idx, cell in enumerate(self.body):
                self._draw(cell, self._clr_head if idx == 0 else self._clr_body)
            pygame.display.flip()
            self._clk.tick(self.metadata.get("render_fps", 10))

    def close(self):
        if self._pg:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._pg = None
            self._surf = None
            self._clk = None

    def _spawn_food(self):
        while True:
            r = self._rnd.randrange(self.board)
            c = self._rnd.randrange(self.board)
            if (r, c) not in self.body:
                self.food = (r, c)
                break

    def _inside(self, pos):
        return 0 <= pos[0] < self.board and 0 <= pos[1] < self.board

    def _hit(self, pos, will_grow=False):
        if not self._inside(pos):
            return True
        body = list(self.body) if will_grow else list(self.body)[:-1]
        return pos in body

    def _turn(self, mv: int):
        if mv == 1:
            nd = (self.heading - 1) % 4
        elif mv == 2:
            nd = (self.heading + 1) % 4
        else:
            nd = self.heading
        if len(self.body) > 1:
            head = self.body[0]
            neck = self.body[1]
            dr, dc = Moves[nd]
            nxt = (head[0] + dr, head[1] + dc)
            if nxt == neck:
                nd = self.heading
        self.heading = nd

    def _make_state(self) -> int:
        ldir = (self.heading - 1) % 4
        rdir = (self.heading + 1) % 4
        dirs = [ldir, self.heading, rdir]
        head = self.body[0]
        dangs: List[int] = []
        for d in dirs:
            dr, dc = Moves[d]
            nxt = (head[0] + dr, head[1] + dc)
            dangs.append(1 if self._hit(nxt, will_grow=False) else 0)
        dL, dF, dR = dangs
        fr, fc = self.food
        vec = (fr - head[0], fc - head[1])
        fwd = Moves[self.heading]
        lv = Moves[(self.heading - 1) % 4]
        rv = Moves[(self.heading + 1) % 4]
        def dot(u, v): return u[0]*v[0] + u[1]*v[1]
        vals = [dot(vec, lv), dot(vec, fwd), dot(vec, rv)]
        fdir = int(np.argmax(vals))
        return self._encode(dL, dF, dR, fdir, self.heading)

    @staticmethod
    def _encode(dL, dF, dR, fdir, head):
        idx = dL
        idx = idx * 2 + dF
        idx = idx * 2 + dR
        idx = idx * 3 + fdir
        idx = idx * 4 + head
        return idx

    def _init_pg(self):
        if self._pg is not None:
            return
        import pygame
        pygame.init()
        w = self.board * self._cell
        h = self.board * self._cell
        self._surf = pygame.display.set_mode((w, h))
        pygame.display.set_caption("SnakeRL")
        self._clk = pygame.time.Clock()
        self._pg = pygame

    def _draw(self, rc, color):
        import pygame
        r, c = rc
        x = c * self._cell
        y = r * self._cell
        rect = pygame.Rect(x + 1, y + 1, self._cell - 2, self._cell - 2)
        pygame.draw.rect(self._surf, color, rect)
