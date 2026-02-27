import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from simulator.models import Side, Order, OrderType
from simulator.impact import MarketImpactModel, ImpactAwareOrderBook
from simulator.strategies import TWAPStrategy, VWAPStrategy, AlmgrenChrissStrategy
import os

def run_monte_carlo(n_sims: int = 100, horizon: int = 50, total_qty: int = 1000, volatility: float = 0.02):
    """
    Simulates different execution strategies under a market impact model.
    """
    # Setup Impact Model
    model = MarketImpactModel(temp_impact_param=0.1, perm_impact_param=0.01, volatility=volatility)
    
    results = {
        'TWAP': {'IS': [], 'Cost': []},
        'VWAP': {'IS': [], 'Cost': []},
        'Almgren-Chriss (Aggressive)': {'IS': [], 'Cost': []},
        'Almgren-Chriss (Passive)': {'IS': [], 'Cost': []}
    }
    
    # Pre-calculate schedules (some depend on market volume)
    # Market volume profile: Assume random volume per step
    # We'll generate a new one per simulation run to be realistic
    
    for _ in range(n_sims):
        # 1. Initialization
        arrival_mid = 100.0
        current_mid = arrival_mid
        market_vol_profile = np.random.randint(50, 150, horizon)
        
        # Strategies
        twap = TWAPStrategy(total_qty, horizon).get_schedule()
        vwap = VWAPStrategy(total_qty, horizon).get_schedule(market_vol_profile)
        
        # Almgren-Chriss with high and low risk aversion
        ac_agg = AlmgrenChrissStrategy(total_qty, horizon, risk_aversion=0.5, volatility=volatility).get_schedule()
        ac_pass = AlmgrenChrissStrategy(total_qty, horizon, risk_aversion=0.05, volatility=volatility).get_schedule()
        
        strategies = {
            'TWAP': twap,
            'VWAP': vwap,
            'Almgren-Chriss (Aggressive)': ac_agg,
            'Almgren-Chriss (Passive)': ac_pass
        }
        
        # 2. Simulate Execution
        for name, schedule in strategies.items():
            model.reset() # Reset permanent impact for each strategy
            total_exec_cost = 0.0
            total_shares_filled = 0
            
            # Mid-price walk (Random walk + permanent impact)
            strategy_mid = arrival_mid
            
            for t in range(horizon):
                trade_size = schedule[t]
                if trade_size <= 0:
                    continue
                
                # Liquidity available at this step
                liquidity = market_vol_profile[t]
                
                # Execute and get cost
                # Note: Impact Model uses its internal state for permanent impact
                exec_price, temp_impact, perm_impact = model.get_execution_price(
                    trade_size, Side.BUY, strategy_mid, liquidity
                )
                
                # Update total cost (Implementation Shortfall)
                # IS = (Avg Exec Price - Arrival Price) / Arrival Price
                # But here we track raw cost first
                total_exec_cost += exec_price * trade_size
                total_shares_filled += trade_size
                
                # Update market mid-price (Random Walk + Permanent Impact)
                # Price discovery move
                strategy_mid += perm_impact + np.random.normal(0, volatility * arrival_mid / np.sqrt(horizon))
                model.update_state(trade_size, Side.BUY)
            
            # 3. Finalize Results
            avg_exec_price = total_exec_cost / total_shares_filled
            is_cost = (avg_exec_price - arrival_mid) # Absolute cost
            
            results[name]['IS'].append(is_cost)
            results[name]['Cost'].append(avg_exec_price)

    return results

def verify_strategies():
    print("--- Running Strategy Comparison (Phase 4 Verification) ---")
    
    # 1. Low Volatility Regime
    print("Simulating Low Volatility Regime...")
    low_vol_results = run_monte_carlo(n_sims=100, volatility=0.01)
    
    # 2. High Volatility Regime
    print("Simulating High Volatility Regime...")
    high_vol_results = run_monte_carlo(n_sims=100, volatility=0.05)
    
    # --- Print Stats ---
    for regime_name, results in [("Low Vol", low_vol_results), ("High Vol", high_vol_results)]:
        print(f"\n{regime_name} Results (Mean IS +/- Std):")
        for name, data in results.items():
            mean_is = np.mean(data['IS'])
            std_is = np.std(data['IS'])
            print(f"  {name:25}: {mean_is:8.4f} +/- {std_is:8.4f}")

    # --- Plotting Distributions ---
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (regime_name, results) in enumerate([("Low Volatility", low_vol_results), ("High Volatility", high_vol_results)]):
        for name, data in results.items():
            axs[i].hist(data['IS'], bins=20, alpha=0.5, label=name)
        axs[i].set_title(f"Implementation Shortfall Distribution ({regime_name})")
        axs[i].set_xlabel("IS (Cost per Share)")
        axs[i].set_ylabel("Frequency")
        axs[i].legend()
    
    plt.tight_layout()
    plot_path = "phase4_strategy_comparison.png"
    plt.savefig(plot_path)
    print(f"\nVerification plots saved to: {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    verify_strategies()
