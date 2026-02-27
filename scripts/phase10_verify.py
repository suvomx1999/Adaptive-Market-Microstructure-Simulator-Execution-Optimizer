import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator.env import TradingEnv
from simulator.rl_agent import PPOAgent
from simulator.models import Side
import os

def verify_sentiment_execution():
    print("Verifying Phase 10: Sentiment-Driven Execution...")
    
    env = TradingEnv(total_quantity=1000, horizon=20, target_side=Side.BUY)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    
    # 1. Train for a few episodes
    print("Training agent with sentiment awareness...")
    for episode in range(50):
        state, _ = env.reset()
        done = False
        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store(state, action, log_prob, reward, done)
            state = next_state
        agent.update()

    # 2. Test: Bullish vs Bearish Sentiment
    print("Testing Sentiment Impact on Execution...")
    
    def run_test_with_sentiment(shock_value):
        env.reset()
        env.sentiment_gen.set_shock(shock_value)
        state, _ = env.reset() # Reset again to apply shock in the env state if needed
        # Manually force the first few steps of sentiment
        env.current_sentiment = shock_value
        
        inventory_path = []
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent.policy(state_t)
            action = torch.argmax(logits).item()
            state, reward, done, _, _ = env.step(action)
            # Override sentiment to maintain the shock for testing
            env.current_sentiment = shock_value 
            inventory_path.append(env.remaining_qty)
        return inventory_path

    bullish_path = run_test_with_sentiment(0.8)
    bearish_path = run_test_with_sentiment(-0.8)
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(bullish_path, label='Bullish Sentiment (+0.8)', marker='o', color='green')
    plt.plot(bearish_path, label='Bearish Sentiment (-0.8)', marker='x', color='red')
    plt.title("Phase 10: Impact of Market Sentiment on BUY Execution Speed")
    plt.xlabel("Time Step")
    plt.ylabel("Remaining Inventory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "phase10_sentiment_verification.png"
    plt.savefig(plot_path)
    print(f"Verification plot saved as {plot_path}")

if __name__ == "__main__":
    verify_sentiment_execution()
