"""Train the Snake DQN with vectorised environments.

    python train.py --steps 300000            # train, save snake_dqn.pth, export web/weights.json
    python train.py --eval 200 --load snake_dqn.pth
"""
import argparse
import time

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
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.95)
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
    agent = Agent(N_FEATURES, N_ACTIONS, lr=args.lr, gamma=args.gamma)
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

    for it in range(1, iters + 1):
        eps = max(0.01, 1.0 - it / (iters * args.eps_decay))
        acts = agent.act(states, eps, rng)
        for k, env in enumerate(envs):
            nxt[k], rew[k], done[k] = env.step(int(acts[k]))
        buf.add_batch(states, acts, rew, nxt, done)
        for k in np.flatnonzero(done):
            env = envs[k]
            games += 1
            best = max(best, env.score)
            recent.append(env.score)
            if run:
                run.log({"score": env.score, "epsilon": eps}, step=it * args.envs)
            nxt[k] = env.reset()
        states, nxt = nxt, states

        if buf.n >= args.batch * 4:
            agent.learn(buf.sample(args.batch, rng))

        if it % 2000 == 0:
            avg = np.mean(recent[-200:]) if recent else 0.0
            print(f"step {it * args.envs:>8}  games {games:>6}  avg200 {avg:5.1f}  best {best:>3}  "
                  f"eps {eps:.2f}  {it * args.envs / (time.time() - t0):,.0f} st/s", flush=True)
            agent.save(args.save)

    agent.save(args.save)
    agent.export_web(args.web)
    sc = evaluate(agent, 100, args.size)
    print(f"final greedy eval (100 games): mean {sc.mean():.1f}  max {sc.max()}")
    if run:
        run.summary["eval_mean"] = float(sc.mean())
        run.finish()


if __name__ == "__main__":
    main()
