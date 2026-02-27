import numpy as np
import matplotlib.pyplot as plt
from simulator.models import Side, Order, OrderType
from simulator.impact import MarketImpactModel
from simulator.strategies import TWAPStrategy, AlmgrenChrissStrategy
from simulator.regime import RegimeDetector, MarketState
import os

def simulate_regime_shifts(total_steps=100, shift_step=40, end_shift_step=70):
    """
    Simulates a price process with regime shifts in volatility.
    """
    prices = [100.0]
    volumes = []
    spreads = []
    
    low_vol = 0.005
    high_vol = 0.02
    
    print(f"--- Simulating Regime Shifts (Total Steps: {total_steps}) ---")
    print(f"Regime Shift at Step {shift_step} (High Vol) and Step {end_shift_step} (Back to Low Vol)")

    for t in range(total_steps):
        # Determine current volatility
        if shift_step <= t < end_shift_step:
            vol = high_vol
            vol_state = 1
        else:
            vol = low_vol
            vol_state = 0
            
        # Update price (Random Walk)
        price_change = np.random.normal(0, vol * prices[-1])
        prices.append(prices[-1] + price_change)
        
        # Update liquidity proxies
        spread = 0.02 if vol_state == 0 else 0.1
        volume = 100 if vol_state == 0 else 50 # Lower volume in high vol
        
        spreads.append(spread)
        volumes.append(volume)
        
    return np.array(prices), np.array(spreads), np.array(volumes), shift_step, end_shift_step

def verify_regime_adaptation():
    # 1. Simulate data
    prices, spreads, volumes, shift, end_shift = simulate_regime_shifts()
    
    # 2. Run Regime Detector (No threshold argument anymore)
    detector = RegimeDetector(window_size=10)
    detected_regimes = []
    
    for i in range(len(prices)-1):
        regime = detector.update(prices[i], spreads[i], volumes[i])
        detected_regimes.append(regime.value)

    # 3. Compare Execution Performance (Simulation)
    total_qty = 1000
    horizon = 100
    
    # Pre-calculate schedules
    twap_sched = TWAPStrategy(total_qty, horizon).get_schedule()
    ac_agg_sched = AlmgrenChrissStrategy(total_qty, horizon, risk_aversion=0.5).get_schedule()
    
    # Simulation logic
    def run_strategy_sim(schedule_func):
        model = MarketImpactModel(temp_impact_param=0.1, perm_impact_param=0.01, volatility=0.01)
        arrival_price = prices[0]
        total_cost = 0.0
        
        # We simulate the cost along the generated price path
        for t in range(horizon):
            trade_qty = schedule_func(t)
            
            # Local market conditions
            mid_price = prices[t]
            liq = volumes[t]
            
            exec_price, temp, perm = model.get_execution_price(trade_qty, Side.BUY, mid_price, liq)
            total_cost += exec_price * trade_qty
            
        avg_price = total_cost / total_qty
        is_cost = avg_price - arrival_price
        return is_cost

    # Static TWAP execution
    twap_cost = run_strategy_sim(lambda t: twap_sched[t])
    
    # Adaptive execution
    def adaptive_sched_func(t):
        if t < len(detected_regimes) and detected_regimes[t] == 1:
            return ac_agg_sched[t]
        return twap_sched[t]
    
    adaptive_cost = run_strategy_sim(adaptive_sched_func)

    # 4. Results
    print(f"\nExecution Comparison:")
    print(f"  Static TWAP IS Cost: {twap_cost:.4f}")
    print(f"  Adaptive Strategy IS Cost: {adaptive_cost:.4f}")
    print(f"  Improvement: {twap_cost - adaptive_cost:.4f}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Price Process and Detected Regimes
    ax1.plot(prices, 'k-', label='Mid Price')
    ax1.axvspan(shift, end_shift, color='red', alpha=0.1, label='Actual High Vol Regime')
    ax1.set_title("Market Price Process and Regime Shifts")
    ax1.set_ylabel("Price")
    ax1.legend()
    
    # Detection Accuracy
    ax2.step(range(len(detected_regimes)), detected_regimes, 'b-', where='post', label='Detected Regime (1=High Vol)')
    ax2.axvspan(shift, end_shift, color='red', alpha=0.1, label='Actual High Vol Regime')
    ax2.set_title("Regime Detection Accuracy")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("State")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Low Vol', 'High Vol'])
    ax2.legend()
    
    plt.tight_layout()
    plot_path = "phase6_regime_verification.png"
    plt.savefig(plot_path)
    print(f"\nVerification plots saved to: {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    verify_regime_adaptation()
