import numpy as np
import matplotlib.pyplot as plt
from simulator.generators import MultiAssetGenerator

def verify_lead_lag():
    print("Verifying Phase 9: Multi-Asset Lead-Lag Correlation...")
    
    gen = MultiAssetGenerator(base_lambda=5.0, correlation=0.9, lag_steps=10)
    
    prices_a = []
    prices_b = []
    times = []
    curr_time = 0
    
    # Simulate 500 steps
    for i in range(500):
        events = gen.generate_events()
        for asset_id, dt, order, is_cancel in events:
            if asset_id == "A":
                curr_time += dt
                prices_a.append(gen.mid_a)
                prices_b.append(gen.mid_b)
                times.append(curr_time)
                
    # 1. Plot Price Evolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1.plot(times, prices_a, label='Asset A (Lead)', color='blue', alpha=0.8)
    ax1.plot(times, prices_b, label='Asset B (Lag)', color='red', alpha=0.8)
    ax1.set_title("Multi-Asset Price Evolution (Lead-Lag Relationship)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Mid Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cross-Correlation Analysis
    # Calculate cross-correlation for different lags
    def cross_corr(x, y, max_lag):
        x = (x - np.mean(x)) / (np.std(x) * len(x))
        y = (y - np.mean(y)) / (np.std(y))
        return np.correlate(x, y, mode='full')

    lags = np.arange(-50, 51)
    # Use full cross-correlation and slice the middle part
    corr_full = np.correlate((prices_a - np.mean(prices_a)), (prices_b - np.mean(prices_b)), mode='full')
    mid_idx = len(corr_full) // 2
    corr_window = corr_full[mid_idx - 50 : mid_idx + 51]
    
    # Normalize
    corr_window = corr_window / (np.std(prices_a) * np.std(prices_b) * len(prices_a))
    
    ax2.plot(lags, corr_window, marker='o', color='purple')
    ax2.axvline(x=10, color='r', linestyle='--', label='Theoretical Lag (10 steps)')
    ax2.set_title("Cross-Correlation between Asset A and Asset B")
    ax2.set_xlabel("Lag (steps)")
    ax2.set_ylabel("Correlation Coefficient")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("phase9_correlation_verification.png")
    print("Verification plot saved as phase9_correlation_verification.png")

if __name__ == "__main__":
    verify_lead_lag()
