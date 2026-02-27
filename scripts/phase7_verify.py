import numpy as np
import matplotlib.pyplot as plt
from simulator.generators import OrderFlowGenerator, MarketRegime

def verify_hawkes():
    print("Verifying Phase 7: Hawkes Process & Volatility Clustering...")
    
    # 1. Comparison: Poisson vs Hawkes
    base_lambda = 2.0
    n_events = 500
    
    # Poisson
    gen_poisson = OrderFlowGenerator(base_lambda=base_lambda, use_hawkes=False)
    times_poisson = []
    curr_t = 0
    for _ in range(n_events):
        dt, _, _ = gen_poisson.generate_event(100.0)
        curr_t += dt
        times_poisson.append(curr_t)
    
    # Hawkes
    gen_hawkes = OrderFlowGenerator(base_lambda=base_lambda, use_hawkes=True)
    times_hawkes = []
    curr_t = 0
    for _ in range(n_events):
        dt, _, _ = gen_hawkes.generate_event(100.0)
        curr_t += dt
        times_hawkes.append(curr_t)
        
    # 2. Plotting Inter-arrival Distributions
    diffs_poisson = np.diff(times_poisson)
    diffs_hawkes = np.diff(times_hawkes)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Inter-arrival Histogram
    axes[0, 0].hist(diffs_poisson, bins=30, alpha=0.5, label='Poisson', color='blue', density=True)
    axes[0, 0].hist(diffs_hawkes, bins=30, alpha=0.5, label='Hawkes', color='red', density=True)
    axes[0, 0].set_title("Inter-arrival Time Distribution")
    axes[0, 0].set_xlabel("dt")
    axes[0, 0].legend()
    
    # Clustering Visualization (Time vs Arrival Index)
    axes[0, 1].scatter(times_poisson, range(n_events), s=1, label='Poisson', color='blue')
    axes[0, 1].scatter(times_hawkes, range(n_events), s=1, label='Hawkes', color='red')
    axes[0, 1].set_title("Arrival Clustering Pattern")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Event Index")
    axes[0, 1].legend()
    
    # Intensity over time for Hawkes
    t_grid = np.linspace(0, max(times_hawkes), 500)
    intensities = [gen_hawkes.hawkes.get_intensity(t) for t in t_grid]
    axes[1, 0].plot(t_grid, intensities, color='red', lw=1)
    axes[1, 0].set_title("Hawkes Intensity Function λ(t)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Intensity")
    
    # Autocorrelation of Inter-arrivals
    def autocorr(x):
        result = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
        return result[result.size//2:] / result[result.size//2]

    axes[1, 1].plot(autocorr(diffs_poisson)[:20], label='Poisson', marker='o', color='blue')
    axes[1, 1].plot(autocorr(diffs_hawkes)[:20], label='Hawkes', marker='o', color='red')
    axes[1, 1].set_title("Autocorrelation of Inter-arrival Times")
    axes[1, 1].set_xlabel("Lag")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("phase7_3_hawkes_verification.png")
    print("Verification plot saved as phase7_3_hawkes_verification.png")

if __name__ == "__main__":
    verify_hawkes()
