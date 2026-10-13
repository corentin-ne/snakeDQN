# snakeDQN
Project for the Computational Intelligence class at POSTECH University.

A Double + Dueling DQN that learns Snake on a 12×12 board, plus a phone-friendly web visualizer.

## Web demo
Open `web/index.html` in any browser (single file, no server needed):
- **Watch AI** – the pretrained agent, with live Q-values, sensor inputs and hidden-layer activations.
- **Train live** – a fresh network trains in the browser (replay buffer, Double DQN, Polyak target, Adam).
- **Play** – swipe / arrow keys.

## Python
```bash
pip install torch numpy pygame        # wandb optional
python train.py --steps 400000         # saves snake_dqn.pth and embeds weights into web/index.html
python train.py --eval 200 --load snake_dqn.pth
python play.py                         # watch the agent (add --human to play)
```

## Design
- `snake_env.py` – headless env, O(1) moves, 20-feature state: danger / clear run / flood-fill free space
  for straight-right-left, heading, food direction, tail offset, length. Relative actions (no suicide reversals).
- `dqn.py` – dueling MLP (20→128→128→V+A), Double DQN targets, Huber loss, grad clipping,
  Polyak target updates, preallocated numpy replay buffer.
- `train.py` – 16 vectorised envs with batched action selection, optional `--wandb` logging.
