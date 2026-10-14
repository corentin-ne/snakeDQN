"""Train the Snake DQN with vectorised environments.

    python train.py --steps 300000            # train, save snake_dqn.pth, export web/weights.json
    python train.py --eval 200 --load snake_dqn.pth
"""
import argparse
import time
from collections import deque

import numpy as np
import torch

from dqn import Agent, ReplayBuffer
from snake_env import N_ACTIONS, N_FEATURES, SnakeEnv


def evaluate(agent, episodes, size, seed=0):
    rng = np.random.default_rng(seed)
    scores = []
    for e in range(episodes):
        env = SnakeEnv(size, size, seed=seed + e)
        s, done = env.reset(), False
        while not done:
            s, _, done = env.step(int(agent.act(s[None], 0.0, rng)[0]))
        scores.append(env.score)
    return np.array(scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000, help="total env steps")
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--size", type=int, default=12)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--nstep", type=int, default=3)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--eps-decay", type=float, default=0.4, help="fraction of steps to anneal epsilon over")
    p.add_argument("--load")
    p.add_argument("--save", default="snake_dqn.pth")
    p.add_argument("--web", default="web/weights.json")
    p.add_argument("--eval", type=int, default=0, help="only evaluate N episodes")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(args.seed)
    agent = Agent(N_FEATURES, N_ACTIONS, hidden=args.hidden, lr=args.lr, gamma=args.gamma, n_step=args.nstep)
    if args.load:
        agent.load(args.load)
    if args.eval:
        sc = evaluate(agent, args.eval, args.size)
        print(f"eval {args.eval} games: mean {sc.mean():.1f}  max {sc.max()}  median {np.median(sc):.0f}")
        return

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="snake-dqn", config=vars(args))

    envs = [SnakeEnv(args.size, args.size, seed=args.seed + i) for i in range(args.envs)]
    states = np.stack([e.reset() for e in envs])
    buf = ReplayBuffer(args.buffer, N_FEATURES)
    nxt = np.empty_like(states)
    rew = np.empty(args.envs, np.float32)
    done = np.empty(args.envs, np.float32)
    recent, games, best, t0 = [], 0, 0, time.time()
    iters = args.steps // args.envs

    n, g = args.nstep, args.gamma
    pending = [deque() for _ in envs]  # per-env (s, a, r) awaiting their n-step return
    best_eval, eval_every = -1.0, max(1, 25_000 // args.envs)

    for it in range(1, iters + 1):
        eps = max(0.01, 1.0 - it / (iters * args.eps_decay))
        acts = agent.act(states, eps, rng)
        out = []
        for k, env in enumerate(envs):
            nxt[k], rew[k], done[k] = env.step(int(acts[k]))
            q = pending[k]
            q.append((states[k].copy(), acts[k], rew[k]))
            if done[k]:
                while q:  # flush with truncated returns
                    out.append((q[0][0], q[0][1], sum(g ** i * t[2] for i, t in enumerate(q)), nxt[k], 1.0))
                    q.popleft()
            elif len(q) == n:
                out.append((q[0][0], q[0][1], sum(g ** i * t[2] for i, t in enumerate(q)), nxt[k].copy(), 0.0))
                q.popleft()
        if out:
            buf.add_batch(*(np.array(c) for c in zip(*out)))
        for k in np.flatnonzero(done):
            env = envs[k]
            games += 1
            best = max(best, env.score)
            recent.append(env.score)
            if run:
                run.log({"score": env.score, "epsilon": eps}, step=it * args.envs)
            nxt[k] = env.reset()
        states, nxt = nxt, states

        for pg in agent.opt.param_groups:  # linear LR decay to 10%
            pg["lr"] = args.lr * (1 - 0.9 * it / iters)
        if buf.n >= args.batch * 4:
            agent.learn(buf.sample(args.batch, rng))

        if it % 1000 == 0:
            avg = np.mean(recent[-200:]) if recent else 0.0
            print(f"step {it * args.envs:>8}  games {games:>6}  avg200 {avg:5.1f}  best {best:>3}  "
                  f"eps {eps:.2f}  {it * args.envs / (time.time() - t0):,.0f} st/s", flush=True)
        if it % eval_every == 0 and eps < 0.2:  # keep the best greedy checkpoint
            sc = evaluate(agent, 30, args.size, seed=10_000).mean()
            if sc > best_eval:
                best_eval = sc
                agent.save(args.save)
            print(f"  eval30 {sc:.1f}  (best {best_eval:.1f})", flush=True)

    if best_eval < 0:
        agent.save(args.save)
    agent.load(args.save)
    agent.export_web(args.web)
    sc = evaluate(agent, 100, args.size)
    print(f"final greedy eval (100 games): mean {sc.mean():.1f}  max {sc.max()}")
    if run:
        run.summary["eval_mean"] = float(sc.mean())
        run.finish()


if __name__ == "__main__":
    main()
