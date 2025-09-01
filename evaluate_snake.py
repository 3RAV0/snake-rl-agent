import argparse
import os
import numpy as np
from snake_gym_env import SnakeEnv

def play_round(environment, qtable, show=False):
    environment.render_mode = "human" if show else None
    obs, info = environment.reset()
    finished = False
    gain = 0.0
    step_count = 0
    while not finished:
        move = int(np.argmax(qtable[obs]))
        obs, reward, term, trunc, info = environment.step(move)
        finished = term or trunc
        gain += reward
        step_count += 1
    if show:
        environment.close()
    return gain, info.get("score", 0), step_count

def test_agent(rounds=100, q_file="q_table.npy", board=16, show_every=0):
    assert os.path.exists(q_file), f"Q-table not found at {q_file}"
    qtable = np.load(q_file)
    environment = SnakeEnv(board=board, render_mode=None)

    gains, scores, steps_log = [], [], []
    for ep in range(1, rounds + 1):
        render_flag = (show_every and ep % show_every == 0)
        g, s, st = play_round(environment, qtable, show=render_flag)
        gains.append(g); scores.append(s); steps_log.append(st)

        if ep % max(1, rounds // 10) == 0 or render_flag:
            print(f"[Eval] Ep {ep}/{rounds} | R={g:.2f} | score={s} | steps={st}")

    print(f"[Eval DONE] AvgR={np.mean(gains):.2f} | AvgScore={np.mean(scores):.2f} | AvgSteps={np.mean(steps_log):.1f}")
    return np.array(gains), np.array(scores), np.array(steps_log)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--q_path", type=str, default="q_table.npy")
    parser.add_argument("--board", type=int, default=16)
    parser.add_argument("--render_sample_every", type=int, default=0)
    args = parser.parse_args()
    test_agent(args.episodes, args.q_path, args.board, args.render_sample_every)

if __name__ == "__main__":
    main()
