import time
import numpy as np
import matplotlib.pyplot as plt
from simulator.fix_engine import FIXEngine
from simulator.models import Order, Side, OrderType

def verify_production_fix():
    print("Verifying Phase 12: Production Engineering & FIX Protocol...")
    
    engine = FIXEngine(sender_comp_id="HFT_AGENT_01", target_comp_id="NSE_LOB")
    
    # 1. Create a Sample Order
    sample_order = Order(
        order_id=5001,
        asset_id="RELIANCE",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=2450.50,
        quantity=500,
        timestamp=time.time()
    )
    
    # 2. Generate FIX Message Type D (NewOrderSingle)
    fix_nos = engine.create_new_order(sample_order)
    print("\nGenerated FIX NewOrderSingle (Type D):")
    print(fix_nos.replace("|", " | "))
    
    # 3. Generate FIX Message Type 8 (ExecutionReport)
    sample_order.filled_quantity = 200 # Partial fill
    fix_er = engine.create_execution_report(sample_order, last_qty=200, last_px=2450.50)
    print("\nGenerated FIX ExecutionReport (Type 8):")
    print(fix_er.replace("|", " | "))
    
    # 4. Latency Benchmark
    print("\nBenchmarking FIX Engine Latency...")
    latencies = []
    for _ in range(1000):
        start_t = time.perf_counter()
        _ = engine.create_new_order(sample_order)
        latencies.append((time.perf_counter() - start_t) * 1e6) # microseconds
        
    avg_latency = np.mean(latencies)
    print(f"Average FIX Encoding Latency: {avg_latency:.2f} μs")
    
    # 5. Plot Latency Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=30, color='orange', alpha=0.7)
    plt.axvline(x=avg_latency, color='red', linestyle='--', label=f'Mean: {avg_latency:.2f}μs')
    plt.title("Phase 12: Production FIX Message Encoding Latency")
    plt.xlabel("Latency (microseconds)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "phase12_fix_latency_verification.png"
    plt.savefig(plot_path)
    print(f"\nVerification plot saved as {plot_path}")

    # Final sanity check on tags
    assert "8=FIX.4.4" in fix_nos, "Invalid FIX version tag"
    assert "35=D" in fix_nos, "Invalid MsgType for NewOrderSingle"
    assert "35=8" in fix_er, "Invalid MsgType for ExecutionReport"
    print("\nFIX Tag-Value Validation: PASS")

if __name__ == "__main__":
    verify_production_fix()
