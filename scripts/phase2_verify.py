import numpy as np
import matplotlib.pyplot as plt
from simulator.generators import OrderFlowGenerator, MarketRegime
from simulator.models import OrderType, Side
import os

def verify_order_flow(n_events=5000):
    gen = OrderFlowGenerator(base_lambda=10.0) # 10 orders per second
    
    inter_arrival_times = []
    event_types = []
    timestamps = []
    
    print(f"--- Simulating {n_events} events for Phase 2 Verification ---")
    
    # 1. Simulate in LOW_VOLATILITY regime
    gen.set_regime(MarketRegime.LOW_VOLATILITY)
    for _ in range(n_events // 2):
        dt, order, is_cancel = gen.generate_event(100.0)
        inter_arrival_times.append(dt)
        timestamps.append(gen.current_time)
        if is_cancel:
            event_types.append("CANCEL")
        else:
            event_types.append(order.order_type.value)

    # 2. Switch to HIGH_VOLATILITY regime
    gen.set_regime(MarketRegime.HIGH_VOLATILITY)
    for _ in range(n_events // 2):
        dt, order, is_cancel = gen.generate_event(100.0)
        inter_arrival_times.append(dt)
        timestamps.append(gen.current_time)
        if is_cancel:
            event_types.append("CANCEL")
        else:
            event_types.append(order.order_type.value)

    # --- Calculations ---
    
    # Empirical vs Theoretical Lambda
    low_vol_dt = inter_arrival_times[:n_events//2]
    high_vol_dt = inter_arrival_times[n_events//2:]
    
    emp_lambda_low = 1.0 / np.mean(low_vol_dt)
    emp_lambda_high = 1.0 / np.mean(high_vol_dt)
    
    print(f"\nEmpirical Arrival Rate (Low Vol): {emp_lambda_low:.2f} (Target: 10.0)")
    print(f"Empirical Arrival Rate (High Vol): {emp_lambda_high:.2f} (Target: 30.0)")

    # Proportions
    counts = {t: event_types.count(t) for t in set(event_types)}
    print("\nOrder Type Proportions:")
    for t, count in counts.items():
        print(f"  {t}: {count/n_events:.2%}")

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Inter-arrival Distribution (Low Vol)
    axs[0, 0].hist(low_vol_dt, bins=50, density=True, alpha=0.7, color='blue', label='Empirical')
    x = np.linspace(0, max(low_vol_dt), 100)
    axs[0, 0].plot(x, 10.0 * np.exp(-10.0 * x), 'r-', lw=2, label='Theoretical (Exp)')
    axs[0, 0].set_title("Inter-arrival Time Distribution (Low Vol)")
    axs[0, 0].set_xlabel("dt (seconds)")
    axs[0, 0].legend()

    # 2. Inter-arrival Distribution (High Vol)
    axs[0, 1].hist(high_vol_dt, bins=50, density=True, alpha=0.7, color='green', label='Empirical')
    x = np.linspace(0, max(high_vol_dt), 100)
    axs[0, 1].plot(x, 30.0 * np.exp(-30.0 * x), 'r-', lw=2, label='Theoretical (Exp)')
    axs[0, 1].set_title("Inter-arrival Time Distribution (High Vol)")
    axs[0, 1].set_xlabel("dt (seconds)")
    axs[0, 1].legend()

    # 3. Order Type Proportions (Pie)
    axs[1, 0].pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    axs[1, 0].set_title("Order Type Proportions")

    # 4. Time Series of Arrivals
    axs[1, 1].plot(timestamps, np.arange(len(timestamps)), color='purple')
    axs[1, 1].axvline(x=timestamps[n_events//2], color='black', linestyle='--', label='Regime Shift')
    axs[1, 1].set_title("Cumulative Arrivals Over Time")
    axs[1, 1].set_xlabel("Time (seconds)")
    axs[1, 1].set_ylabel("Cumulative Count")
    axs[1, 1].legend()

    plt.tight_layout()
    plot_path = "phase2_verification.png"
    plt.savefig(plot_path)
    print(f"\nVerification plots saved to: {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    verify_order_flow()
