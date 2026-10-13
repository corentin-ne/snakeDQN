"""Double + Dueling DQN with a preallocated numpy replay buffer."""
import base64
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)
        self.a = nn.Linear(hidden, n_out)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        a = self.a(x)
        return self.v(x) + a - a.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity, n_features):
        self.cap, self.i, self.n = capacity, 0, 0
        self.s = np.zeros((capacity, n_features), np.float32)
        self.s2 = np.zeros((capacity, n_features), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add_batch(self, s, a, r, s2, d):
        idx = (self.i + np.arange(len(a))) % self.cap
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx] = s, a, r, s2, d
        self.i = (self.i + len(a)) % self.cap
        self.n = min(self.n + len(a), self.cap)

    def sample(self, k, rng):
        j = rng.integers(0, self.n, k)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j]


class Agent:
    def __init__(self, n_in, n_out, hidden=128, lr=5e-4, gamma=0.95, tau=0.01, device="cpu"):
        self.n_out, self.gamma, self.tau, self.device = n_out, gamma, tau, device
        self.net = DuelingMLP(n_in, n_out, hidden).to(device)
        self.target = DuelingMLP(n_in, n_out, hidden).to(device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, states, epsilon, rng):
        q = self.net(torch.as_tensor(states, device=self.device))
        actions = q.argmax(1).cpu().numpy()
        explore = rng.random(len(actions)) < epsilon
        actions[explore] = rng.integers(0, self.n_out, explore.sum())
        return actions

    def learn(self, batch):
        s, a, r, s2, d = (torch.as_tensor(x, device=self.device) for x in batch)
        with torch.no_grad():  # Double DQN: online net picks, target net evaluates
            a2 = self.net(s2).argmax(1, keepdim=True)
            y = r + self.gamma * (1 - d) * self.target(s2).gather(1, a2).squeeze(1)
        q = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():  # Polyak target update
            for p, tp in zip(self.net.parameters(), self.target.parameters()):
                tp.lerp_(p, self.tau)
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.net.state_dict())

    def export_web(self, path):
        """Write weights as base64 float32 (row-major [out, in]) for web/index.html."""
        sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.net.state_dict().items()}
        blob = np.concatenate([sd[k].ravel() for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                                                         "v.weight", "v.bias", "a.weight", "a.bias")])
        meta = {"in": sd["fc1.weight"].shape[1], "hidden": sd["fc1.weight"].shape[0],
                "out": sd["a.weight"].shape[0], "w": base64.b64encode(blob.tobytes()).decode()}
        with open(path, "w") as f:
            json.dump(meta, f)
        page = os.path.join(os.path.dirname(path), "index.html")
        if os.path.exists(page):  # embed so the page works as a single file
            with open(page) as f:
                html = f.read()
            html = re.sub(r'(<script id="weights" type="application/json">).*?(</script>)',
                          lambda m: m.group(1) + json.dumps(meta) + m.group(2), html, count=1, flags=re.S)
            with open(page, "w") as f:
                f.write(html)
