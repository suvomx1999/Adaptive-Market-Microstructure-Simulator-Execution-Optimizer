import numpy as np
import matplotlib.pyplot as plt
from simulator.models import Order, Side, OrderType
from simulator.lob import OrderBook
from simulator.impact import MarketImpactModel, ImpactAwareOrderBook
import os

def verify_market_impact(n_samples=50):
    # Setup
    print("--- Simulating Market Impact vs Trade Size ---")
    trade_sizes = np.linspace(1, 100, n_samples, dtype=int)
    
    temp_impacts = []
    perm_impacts = []
    total_costs = []
    
    # 1. Base Impact Model
    model = MarketImpactModel(temp_impact_param=0.1, perm_impact_param=0.01, volatility=0.02)
    
    # Assume fixed liquidity and mid-price for isolation test
    mid_price = 100.0
    liquidity = 500.0
    
    for size in trade_sizes:
        exec_price, temp, perm = model.get_execution_price(size, Side.BUY, mid_price, liquidity)
        
        # Calculate per-share cost relative to mid
        cost = exec_price - mid_price
        
        temp_impacts.append(temp)
        perm_impacts.append(perm)
        total_costs.append(cost)
        
    # --- Verification Checks ---
    
    # 1. Monotonicity
    is_monotonic = np.all(np.diff(total_costs) >= 0)
    print(f"1. Impact increases monotonically with trade size: {'PASS' if is_monotonic else 'FAIL'}")
    
    # 2. Convexity (Temporary impact usually follows square root law ~ size^0.5)
    # Check if second derivative is negative (concave down) or positive (convex)
    # Almgren-Chriss temp impact is typically linear in participation rate or sqrt in volume
    # Our implementation uses sqrt(size). So cost curve should be concave.
    is_concave = np.all(np.diff(np.diff(temp_impacts)) <= 0.001) # Allow small float error
    print(f"2. Temporary impact shape (Sqrt Law): {'PASS' if is_concave else 'FAIL'}")

    # 3. Permanent Impact (Linear)
    # Note: trade_sizes is created with linspace(1, 100, 50, dtype=int).
    # Since we cast to int, the step size might not be perfectly constant.
    # 1, 3, 5, 7, 9... (step 2)
    # But 100/49 is approx 2.04, so sometimes step is 2, sometimes 3.
    # This non-constant step size causes the derivative check to fail for linear function.
    
    # Let's verify linearity by checking R-squared of a linear fit
    slope, intercept = np.polyfit(trade_sizes, perm_impacts, 1)
    predicted = slope * trade_sizes + intercept
    r_squared = 1 - (np.sum((perm_impacts - predicted) ** 2) / np.sum((perm_impacts - np.mean(perm_impacts)) ** 2))
    
    is_linear = r_squared > 0.999
    print(f"3. Permanent impact shape (Linear): {'PASS' if is_linear else 'FAIL'} (R^2: {r_squared:.6f})")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(trade_sizes, total_costs, 'k-', linewidth=2, label='Total Impact Cost')
    plt.plot(trade_sizes, temp_impacts, 'b--', label='Temporary Impact (Slippage)')
    plt.plot(trade_sizes, perm_impacts, 'r:', label='Permanent Impact (Price Move)')
    
    plt.title("Market Impact Cost Analysis")
    plt.xlabel("Trade Size (Shares)")
    plt.ylabel("Price Impact ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "phase3_impact_verification.png"
    plt.savefig(plot_path)
    print(f"\nVerification plot saved to: {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    verify_market_impact()
