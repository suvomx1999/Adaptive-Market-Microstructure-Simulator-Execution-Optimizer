import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator.env import TradingEnv
from simulator.rl_agent import PPOAgent
from simulator.strategies import TWAPStrategy
from simulator.models import Side
import os

def train_agent(n_episodes=100):
    env = TradingEnv(total_quantity=1000, horizon=20, target_side=Side.BUY)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    
    rewards_history = []
    
    print(f"--- Training PPO Execution Agent for {n_episodes} episodes ---")
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store(state, action, log_prob, reward, done)
            state = next_state
            episode_reward += reward
            
        agent.update()
        
        if episode % 10 == 0:
            print(f"Episode {episode:3}: Reward = {episode_reward:8.4f}")
            
        rewards_history.append(episode_reward)
        
    return agent, rewards_history

def compare_rl_vs_twap(agent, n_tests=50):
    env = TradingEnv(total_quantity=1000, horizon=20, target_side=Side.BUY)
    
    rl_costs = []
    twap_costs = []
    
    print(f"\n--- Comparing PPO vs TWAP over {n_tests} runs ---")
    
    for _ in range(n_tests):
        # 1. Test RL
        state, _ = env.reset()
        rl_total_reward = 0
        done = False
        while not done:
            # Greedy action
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs, _ = agent.policy(state_t)
            action = torch.argmax(probs).item()
            state, reward, done, _, _ = env.step(action)
            rl_total_reward += reward
        rl_costs.append(-rl_total_reward) # Cost is negative reward

        # 2. Test TWAP
        env.reset()
        twap_total_reward = 0
        done = False
        while not done:
            _, reward, done, _, _ = env.step(action=1) # Action 1 is 5% fill (TWAP for 20 steps)
            twap_total_reward += reward
        twap_costs.append(-twap_total_reward)
        
    return rl_costs, twap_costs

def verify_rl_agent():
    # 1. Train
    agent, rewards = train_agent(n_episodes=100)
    
    # 2. Compare
    rl_costs, twap_costs = compare_rl_vs_twap(agent, n_tests=50)
    
    # 3. Stats
    print(f"\nResults:")
    print(f"  PPO Mean Cost : {np.mean(rl_costs):.4f}")
    print(f"  TWAP Mean Cost: {np.mean(twap_costs):.4f}")
    
    # 4. Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Learning Curve
    smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
    ax1.plot(smoothed_rewards)
    ax1.set_title("PPO Learning Curve (Smoothed Rewards)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    
    # Cost Comparison
    ax2.boxplot([rl_costs, twap_costs], labels=['PPO Agent', 'TWAP'])
    ax2.set_title("Execution Cost Comparison")
    ax2.set_ylabel("IS Cost (Absolute)")
    
    plt.tight_layout()
    plot_path = "phase5_rl_verification.png"
    plt.savefig(plot_path)
    print(f"\nVerification plots saved to: {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    verify_rl_agent()
