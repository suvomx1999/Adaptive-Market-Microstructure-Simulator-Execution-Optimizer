import numpy as np
import matplotlib.pyplot as plt
from simulator.lob import OrderBook
from simulator.models import Order, Side, OrderType
from simulator.strategies import PredatoryHFTAgent

def verify_predatory_hft():
    print("Verifying Phase 8: Multi-Agent Competition (Predatory HFT Detection)...")
    
    lob = OrderBook()
    predator = PredatoryHFTAgent(detection_window=5, detection_threshold=1.5)
    
    # 1. Initialize LOB with some liquidity
    for i in range(10):
        lob.add_order(Order(i, Side.BUY, OrderType.LIMIT, 100.0 - i*0.1, 100, timestamp=0))
        lob.add_order(Order(i+10, Side.SELL, OrderType.LIMIT, 100.1 + i*0.1, 100, timestamp=0))
        
    detection_flags = []
    imbalances = []
    
    # 2. Simulate a large buyer entering the market (creating imbalance)
    print("Simulating large buyer entering...")
    for t in range(30):
        # Large buyer adds buy orders every step
        lob.add_order(Order(100+t, Side.BUY, OrderType.LIMIT, 100.0, 300, timestamp=t))
        
        # Predator monitors the book
        bids, asks = lob.get_snapshot(5)
        pred_order = predator.update(bids, asks)
        
        # Track metrics
        total_bid = sum(q for p, q in bids[:3])
        total_ask = sum(q for p, q in asks[:3])
        imbalances.append(total_bid / total_ask)
        detection_flags.append(1 if predator.is_detecting else 0)
        
        if pred_order:
            side, qty, price = pred_order
            # Predator places its order
            lob.add_order(Order(200+t, side, OrderType.LIMIT, price, qty, timestamp=t))
            
    # 3. Plot results
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Order Book Imbalance (B/A)', color='tab:blue')
    ax1.plot(imbalances, color='tab:blue', label='LOB Imbalance')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(y=1.5, color='r', linestyle='--', alpha=0.3, label='Detection Threshold')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Predator Detection Signal', color='tab:red')
    ax2.fill_between(range(len(detection_flags)), 0, detection_flags, color='tab:red', alpha=0.2, label='Detection Active')
    ax2.set_ylim(-0.1, 1.1)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title("Phase 8: Predatory HFT Detection of Large Execution Orders")
    fig.tight_layout()
    plt.savefig("phase8_predatory_verification.png")
    print("Verification plot saved as phase8_predatory_verification.png")

if __name__ == "__main__":
    verify_predatory_hft()
