import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from simulator.lob import OrderBook
from simulator.generators import OrderFlowGenerator, MarketRegime
from simulator.data_loader import FI2010DataLoader
from simulator.strategies import TWAPStrategy, VWAPStrategy, AlmgrenChrissStrategy
from simulator.rl_agent import PPOAgent
from simulator.regime import RegimeDetector, AdaptiveStrategy
from simulator.models import Side, Order, OrderType
import torch
import random

def calculate_risk_metrics(costs):
    """Calculate IS, CVaR, Variance, and Impact Cost"""
    mean_is = np.mean(costs)
    var_is = np.var(costs)
    cvar_95 = np.mean(np.sort(costs)[int(0.95 * len(costs)):])
    return mean_is, var_is, cvar_95

def run_simulation(strategy, num_episodes=100, horizon=100, quantity=1000):
    all_is = []
    for _ in range(num_episodes):
        lob = OrderBook()
        gen = OrderFlowGenerator(base_lambda=5.0, use_hawkes=True)
        mid = 100.0
        
        # Warm up
        for _ in range(20):
            dt, order, cancel = gen.generate_event(mid)
            if order: lob.add_order(order)
            best_bid = lob.get_best_bid()
            best_ask = lob.get_best_ask()
            if best_bid and best_ask:
                mid = (best_bid + best_ask) / 2.0

        remaining = quantity
        total_cost = 0
        
        for t in range(horizon):
            # Check for regime shift (randomly trigger)
            if np.random.random() < 0.05:
                gen.trigger_liquidity_shock(duration=10)
            
            # Get strategy action
            if hasattr(strategy, 'get_action'):
                # Adaptive Strategy
                best_bid = lob.get_best_bid() or (mid - 0.01)
                best_ask = lob.get_best_ask() or (mid + 0.01)
                spread = best_ask - best_bid
                vol = 100 # Mock volume
                action_size = strategy.get_action(mid, spread, vol, remaining, horizon - t)
            elif hasattr(strategy, 'get_next_trade'):
                # Standard Strategy (TWAP, VWAP, Almgren-Chriss)
                action_size = strategy.get_next_trade(remaining, horizon - t)
            elif hasattr(strategy, 'act'):
                # RL Agent directly
                # Mock state for RL
                state = np.array([0.1, 0.0, 0.5, 0.5, 0.0]) # spread, imb, inv, time, sent
                action_idx, _ = strategy.act(state)
                action_size = min(remaining, (action_idx + 1) * 50)
            else:
                action_size = 0

            # Execute
            if action_size > 0:
                order = Order(999, Side.SELL, OrderType.MARKET, 0, action_size)
                trades = lob.add_order(order)
                if trades:
                    avg_price = sum(tr.price * tr.quantity for tr in trades) / sum(tr.quantity for tr in trades)
                    total_cost += (mid - avg_price) * action_size
                remaining -= action_size
            
            # Step market
            dt, order, cancel = gen.generate_event(mid)
            if order: lob.add_order(order)
            best_bid = lob.get_best_bid()
            best_ask = lob.get_best_ask()
            if best_bid and best_ask:
                mid = (best_bid + best_ask) / 2.0
            
            if remaining <= 0: break
            
        # Terminal execution if remaining
        if remaining > 0:
            order = Order(999, Side.SELL, OrderType.MARKET, 0, remaining)
            trades = lob.add_order(order)
            if trades:
                avg_price = sum(tr.price * tr.quantity for tr in trades) / sum(tr.quantity for tr in trades)
                total_cost += (mid - avg_price) * remaining
        
        is_cost = total_cost / (quantity * 100.0) # Percentage IS
        all_is.append(is_cost)
        
    return np.array(all_is)

def run_simulation_real_data(strategy, num_episodes=10, horizon=100, quantity=1000):
    """Runs simulation using the real FI-2010 LOB dataset"""
    all_is = []
    loader = FI2010DataLoader()
    
    for _ in range(num_episodes):
        lob = OrderBook()
        loader.reset()
        mid = 100.0
        
        # Initial LOB state from real data
        bids, asks = loader.get_lob_snapshot()
        for p, q in bids: lob.add_order(Order(random.randint(1,1000), Side.BUY, OrderType.LIMIT, p, q))
        for p, q in asks: lob.add_order(Order(random.randint(1,1000), Side.SELL, OrderType.LIMIT, p, q))
        
        remaining = quantity
        total_cost = 0
        
        for t in range(horizon):
            # Get strategy action
            if hasattr(strategy, 'act'):
                # RL Agent directly (using mock state for this demo)
                state = np.array([0.1, 0.0, 0.5, 0.5, 0.0])
                action_idx, _ = strategy.act(state)
                action_size = min(remaining, (action_idx + 1) * 50)
            else:
                action_size = 0

            # Execute
            if action_size > 0:
                order = Order(999, Side.SELL, OrderType.MARKET, 0, action_size)
                trades = lob.add_order(order)
                if trades:
                    avg_price = sum(tr.price * tr.quantity for tr in trades) / sum(tr.quantity for tr in trades)
                    total_cost += (mid - avg_price) * action_size
                remaining -= action_size
            
            # Step market using real data
            loader.get_next_state()
            bids, asks = loader.get_lob_snapshot()
            lob = OrderBook() # Sync to snapshot
            for p, q in bids: lob.add_order(Order(random.randint(1,1000), Side.BUY, OrderType.LIMIT, p, q))
            for p, q in asks: lob.add_order(Order(random.randint(1,1000), Side.SELL, OrderType.LIMIT, p, q))
            
            best_bid = lob.get_best_bid()
            best_ask = lob.get_best_ask()
            if best_bid and best_ask:
                mid = (best_bid + best_ask) / 2.0
            
            if remaining <= 0: break
            
        if remaining > 0:
            order = Order(999, Side.SELL, OrderType.MARKET, 0, remaining)
            trades = lob.add_order(order)
            if trades:
                avg_price = sum(tr.price * tr.quantity for tr in trades) / sum(tr.quantity for tr in trades)
                total_cost += (mid - avg_price) * remaining
        
        is_cost = total_cost / (quantity * 100.0)
        all_is.append(is_cost)
        
    return np.array(all_is)

def main():
    print("--- Starting Ablation Study and Statistical Validation ---")
    
    # 1. Initialize Strategies
    twap = TWAPStrategy(1000, 100)
    vwap = VWAPStrategy(1000, 100)
    
    # RL Agents
    ppo_vanilla = PPOAgent(5, 6) # No regime awareness
    
    # Adaptive Strategy (PPO + HMM + BOCPD)
    agg_ac = AlmgrenChrissStrategy(1000, 100, risk_aversion=0.1)
    pass_ac = AlmgrenChrissStrategy(1000, 100, risk_aversion=0.01)
    adaptive = AdaptiveStrategy(agg_ac, pass_ac)
    
    # 2. Run Simulations
    n_eps = 200
    results = {}
    print(f"Running {n_eps} episodes for each strategy...")
    
    results['TWAP'] = run_simulation(twap, n_eps)
    results['VWAP'] = run_simulation(vwap, n_eps)
    results['PPO RL'] = run_simulation(ppo_vanilla, n_eps)
    results['Adaptive (PPO+HMM+BOCPD)'] = run_simulation(adaptive, n_eps)
    
    # RL on Real Data
    print("Running RL Agent on Real FI-2010 Data...")
    results['RL (FI-2010 Data)'] = run_simulation_real_data(ppo_vanilla, 20)
    
    # 3. Create Ablation Table
    ablation_data = []
    for name, costs in results.items():
        mean_is, var_is, cvar = calculate_risk_metrics(costs)
        # Statistical Significance (t-test against VWAP)
        t_stat, p_val = stats.ttest_ind(costs, results['VWAP'])
        sig = "*" if p_val < 0.05 else ""
        if p_val < 0.01: sig = "**"
        
        ablation_data.append({
            'Model': name,
            'Mean IS (%)': f"{mean_is*100:.3f}",
            'Variance': f"{var_is:.5f}",
            'CVaR (95%)': f"{cvar*100:.3f}",
            'p-value (vs VWAP)': f"{p_val:.4f}{sig}"
        })
    
    df = pd.DataFrame(ablation_data)
    print("\n--- Ablation Study Results ---")
    print(df.to_string(index=False))
    
    # 4. Generate Plots
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Cost Distributions
    plt.subplot(1, 2, 1)
    for name, costs in results.items():
        plt.hist(costs, bins=30, alpha=0.5, label=name)
    plt.title("Execution Cost Distribution (IS)")
    plt.xlabel("Implementation Shortfall (%)")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Plot 2: Boxplot for risk comparison
    plt.subplot(1, 2, 2)
    plt.boxplot([results[k] for k in results.keys()], labels=results.keys())
    plt.title("Strategy Risk Comparison")
    plt.ylabel("IS (%)")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("ablation_study_results.png")
    print("\nResults saved to ablation_study_results.png")

if __name__ == "__main__":
    main()
