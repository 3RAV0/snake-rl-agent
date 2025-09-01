import argparse
import os
import numpy as np
from snake_gym_env import SnakeEnv

def show(qfile="q_table.npy", rounds=5, board=16, speed=10):
    assert os.path.exists(qfile), f"Q-table not found at {qfile}"
    Q = np.load(qfile)

    env = SnakeEnv(board=board, render_mode="human")
    env.metadata["render_fps"] = speed

    for ep in range(1, rounds + 1):
        st, info = env.reset()
        done = False
        gain = 0.0

        while not done:
            act = int(np.argmax(Q[st]))
            st, rew, term, trunc, info = env.step(act)
            done = term or trunc
            gain += rew

        print(f"[Show] Ep {ep} | Score={info.get('score',0)} | Return={gain:.2f}")

    env.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q_path", type=str, default="q_table.npy")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--board", type=int, default=16)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()
    show(args.q_path, args.episodes, args.board, args.fps)

if __name__ == "__main__":
    main()
