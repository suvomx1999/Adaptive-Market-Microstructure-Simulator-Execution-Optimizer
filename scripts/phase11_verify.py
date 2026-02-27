import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator.env import TradingEnv
from simulator.rl_agent import PPOAgent
from simulator.models import Side
import os

def verify_distributional_rl():
    print("Verifying Phase 11: Distributional RL (QR-PPO)...")
    
    env = TradingEnv(total_quantity=1000, horizon=20, target_side=Side.BUY)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    n_quantiles = 32
    agent = PPOAgent(state_dim, action_dim, n_quantiles=n_quantiles)
    
    # 1. Train for a few episodes
    print("Training QR-PPO agent to learn cost distribution...")
    rewards = []
    for episode in range(100):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store(state, action, log_prob, reward, done)
            state = next_state
            ep_reward += reward
        agent.update()
        rewards.append(ep_reward)
        if episode % 20 == 0:
            print(f"Episode {episode}: Reward {ep_reward:.4f}")

    # 2. Visualize Learned Distribution (Quantiles)
    state, _ = env.reset()
    state_t = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        _, quantiles = agent.policy(state_t)
    
    quantiles = quantiles.squeeze().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_quantiles), quantiles, color='skyblue', alpha=0.7)
    plt.axhline(y=np.mean(quantiles), color='red', linestyle='--', label='Mean Value (Standard Critic)')
    plt.title("Phase 11: Learned Value Distribution (Quantile Regression)")
    plt.xlabel("Quantile Index")
    plt.ylabel("Estimated Return (Value)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "phase11_dist_rl_verification.png"
    plt.savefig(plot_path)
    print(f"Verification plot saved as {plot_path}")
    
    # 3. Cost Statistics
    print(f"\nDistributional Stats:")
    print(f"  Mean Value: {np.mean(quantiles):.4f}")
    print(f"  5th Percentile (VaR proxy): {quantiles[int(0.05*n_quantiles)]:.4f}")
    print(f"  CVaR Proxy: {np.mean(quantiles[:int(0.05*n_quantiles)]):.4f}")

if __name__ == "__main__":
    verify_distributional_rl()
