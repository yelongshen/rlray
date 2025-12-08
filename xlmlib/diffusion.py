import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from tqdm import tqdm
import matplotlib.pyplot as plt

class MLP(nn.Sequential):
    def __init__(self, in_features, out_features, hidden_features=[64, 64]):
        layers = []
        for a, b in zip((in_features, *hidden_features), (*hidden_features, out_features)):
            layers.extend([nn.Linear(a, b), nn.ELU()])
        super().__init__(*layers[:-1])

class EpsNet(nn.Module):
    def __init__(self, features=2, freqs=16, hidden=[128, 128, 128]):
        super().__init__()
        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)
        self.net = MLP(2 * freqs + features, features, hidden_features=hidden)

    def time_embed(self, t):
        t = self.freqs * t[..., None]
        return torch.cat([t.cos(), t.sin()], dim=-1)

    def forward(self, x_t, t):
        emb = self.time_embed(t).expand(*x_t.shape[:-1], -1)
        return self.net(torch.cat([x_t, emb], dim=-1))

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.device = device
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_acp = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - alphas_cumprod)
        self.one_over_sqrt_alpha = 1.0 / torch.sqrt(alphas)
        self.posterior_var = betas * (1.0 - alphas_cumprod.roll(1, 0)) / (1.0 - alphas_cumprod)
        self.posterior_var[0] = betas[0]

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_acp[t].unsqueeze(-1)
        s2 = self.sqrt_one_minus_acp[t].unsqueeze(-1)
        return s1 * x0 + s2 * noise

    def p_sample(self, model, x_t, t):
        beta_t = self.betas[t].unsqueeze(-1)
        alpha_t = self.alphas[t].unsqueeze(-1)
        acp_t = self.alphas_cumprod[t].unsqueeze(-1)
        eps_pred = model(x_t, t.float() / (self.T - 1))
        mean = self.one_over_sqrt_alpha[t].unsqueeze(-1) * (x_t - beta_t / torch.sqrt(1.0 - acp_t) * eps_pred)
        if (t == 0).all():
            return mean
        var = self.posterior_var[t].unsqueeze(-1)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    def sample(self, model, n, features=2):
        x = torch.randn(n, features, device=self.device)
        for i in tqdm(reversed(range(self.T)), total=self.T, ncols=88, leave=False):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_np, _ = make_moons(16384, noise=0.05)
    data = torch.from_numpy(data_np).float().to(device)

    model = EpsNet(features=2, freqs=16, hidden=[128, 128, 128]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 1000 steps. 
    ddpm = DDPM(T=1000, beta_start=1e-4, beta_end=2e-2, device=device)

    bs = 256
    steps = 16384
    for step in tqdm(range(steps), ncols=88):
        idx = torch.randint(0, len(data), (bs,), device=device)
        x0 = data[idx]
        t = torch.randint(0, ddpm.T, (bs,), device=device)
        noise = torch.randn_like(x0)
        x_t = ddpm.q_sample(x0, t, noise) #forward process.
        t_norm = t.float() / (ddpm.T - 1)
        pred = model(x_t, t_norm)
        loss = F.mse_loss(pred, noise)
        if step % 100 == 0:
            print(loss.detach().item())
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        samples = ddpm.sample(model, n=16384).cpu()

    plt.figure(figsize=(4.8, 4.8), dpi=150)
    plt.hist2d(samples[:, 0].numpy(), samples[:, 1].numpy(), bins=64)
    plt.savefig('moons_ddpm.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
