import sys
import pygame
from snake_gym_env import SnakeEnv

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

def key_to_dir(key):
    if key in (pygame.K_w, pygame.K_UP): return UP
    if key in (pygame.K_d, pygame.K_RIGHT): return RIGHT
    if key in (pygame.K_s, pygame.K_DOWN): return DOWN
    if key in (pygame.K_a, pygame.K_LEFT): return LEFT
    return None

def to_relative(cur, target):
    if target is None: return 0
    diff = (target - cur) % 4
    if diff == 0: return 0
    if diff == 1: return 2
    if diff == 3: return 1
    return 0

def main():
    env = SnakeEnv(board=16, render_mode="human")
    obs, info = env.reset()

    pygame.init()
    pygame.key.set_repeat(0)
    clk = pygame.time.Clock()
    spd = 10

    target_dir = None
    can_turn = True
    alive = True
    hold = False
    moves = 0

    while alive:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                alive = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    alive = False
                elif k == pygame.K_r:
                    obs, info = env.reset()
                    target_dir = None
                    can_turn = True
                    moves = 0
                elif k == pygame.K_p:
                    hold = not hold
                elif k in (pygame.K_PLUS, pygame.K_EQUALS):
                    spd = min(60, spd + 1)
                elif k == pygame.K_MINUS:
                    spd = max(1, spd - 1)
                else:
                    if can_turn:
                        d = key_to_dir(k)
                        if d is not None:
                            target_dir = d
                            can_turn = False

        if hold:
            clk.tick(15)
            continue

        act = to_relative(env.heading, target_dir)
        obs, reward, term, trunc, info = env.step(act)
        moves += 1

        target_dir = None
        can_turn = True

        if term or trunc:
            pygame.display.set_caption(f"SNAKE | Score: {info.get('score', 0)}")
            print(f"Over! Score: {info.get('score', 0)}, Steps: {moves}")
            pygame.time.wait(300)
            obs, info = env.reset()
            target_dir = None
            can_turn = True
            moves = 0

        clk.tick(spd)

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
