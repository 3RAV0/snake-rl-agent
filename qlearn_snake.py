import argparse
import os
import random
import numpy as np
from snake_gym_env import SnakeEnv

def pick_action(Q, st, eps):
    if random.random() < eps:
        return random.randrange(Q.shape[1])
    return int(np.argmax(Q[st]))

def learn(
    rounds=5000,
    lr=0.1,
    disc=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=4000,
    show_every=0,
    seed=42,
    save_file=None,
    load_file=None,
    board=16
):
    random.seed(seed)
    np.random.seed(seed)
    env = SnakeEnv(board=board, render_mode=None, seed=seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)

    if load_file and os.path.exists(load_file):
        Q_in = np.load(load_file)
        assert Q_in.shape == Q.shape
        Q[:] = Q_in
        print(f"Loaded Q from {load_file}")

    scores = []
    avgs = []
    best = -1

    def eps_sched(ep):
        if eps_decay <= 0:
            return eps_end
        frac = min(1.0, ep / float(eps_decay))
        return eps_start + frac * (eps_end - eps_start)

    for ep in range(1, rounds + 1):
        env.render_mode = "human" if (show_every and ep % show_every == 0) else None
        st, info = env.reset()
        done = False
        gain = 0.0
        cnt = 0

        while not done:
            e = eps_sched(ep)
            act = pick_action(Q, st, e)
            nxt, rew, term, trunc, info = env.step(act)
            done = term or trunc
            mx = np.max(Q[nxt])
            tgt = rew + (0.0 if done else disc * mx)
            err = tgt - Q[st, act]
            Q[st, act] += lr * err
            st = nxt
            gain += rew
            cnt += 1

        scores.append(gain)
        if len(scores) >= 100:
            avgs.append(float(np.mean(scores[-100:])))
        else:
            avgs.append(float(np.mean(scores)))

        if info.get("score", -1) > best:
            best = int(info["score"])

        if ep % max(1, (rounds // 20)) == 0 or env.render_mode == "human":
            print(f"Ep {ep:5d}/{rounds} | eps={e:.3f} | R={gain:7.2f} | avg100={avgs[-1]:7.2f} | score={info.get('score',0)} | steps={cnt}")

        if env.render_mode == "human":
            env.close()

    print(f"Done. Best score: {best}")
    if save_file:
        np.save(save_file, Q)
        print(f"Saved Q to {save_file}")

    return Q, np.array(scores, dtype=np.float32), np.array(avgs, dtype=np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_end", type=float, default=0.05)
    p.add_argument("--epsilon_decay_episodes", type=int, default=4000)
    p.add_argument("--render_every", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", type=str, default="q_table.npy")
    p.add_argument("--load_path", type=str, default=None)
    p.add_argument("--board", type=int, default=16)
    args = p.parse_args()

    learn(
        rounds=args.episodes,
        lr=args.alpha,
        disc=args.gamma,
        eps_start=args.epsilon_start,
        eps_end=args.epsilon_end,
        eps_decay=args.epsilon_decay_episodes,
        show_every=args.render_every,
        seed=args.seed,
        save_file=args.save_path,
        load_file=args.load_path,
        board=args.board
    )

if __name__ == "__main__":
    main()
