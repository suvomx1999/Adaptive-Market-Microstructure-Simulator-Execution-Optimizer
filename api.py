from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from simulator.models import Side, Order, OrderType
from simulator.lob import OrderBook
from simulator.generators import OrderFlowGenerator, MarketRegime
from simulator.impact import MarketImpactModel
from simulator.strategies import TWAPStrategy
import numpy as np
from typing import List, Dict, Any

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulator state
class Simulator:
    def __init__(self):
        self.lob = OrderBook()
        self.generator = OrderFlowGenerator(base_lambda=5.0)
        self.mid_price = 100.0
        self.history = []
        self.trades = []
        
        # Warm up
        for _ in range(50):
            self.step()

    def step(self):
        dt, order, is_cancel = self.generator.generate_event(self.mid_price)
        if not is_cancel and order:
            new_trades = self.lob.add_order(order)
            self.trades.extend([{"price": t.price, "qty": t.quantity, "time": t.timestamp} for t in new_trades])
        
        # Update mid price
        best_bid = self.lob.get_best_bid()
        best_ask = self.lob.get_best_ask()
        if best_bid and best_ask:
            self.mid_price = (best_bid + best_ask) / 2.0
        
        self.history.append({
            "time": self.generator.current_time,
            "mid": self.mid_price,
            "bid": best_bid,
            "ask": best_ask
        })
        if len(self.history) > 100:
            self.history.pop(0)

sim = Simulator()

@app.get("/state")
def get_state():
    bids, asks = sim.lob.get_snapshot(10)
    return {
        "mid": sim.mid_price,
        "spread": sim.lob.get_spread(),
        "bids": [{"price": p, "qty": q} for p, q in bids],
        "asks": [{"price": p, "qty": q} for p, q in asks],
        "history": sim.history,
        "trades": sim.trades[-20:] # Last 20 trades
    }

@app.post("/step")
def run_step():
    sim.step()
    return get_state()

@app.post("/reset")
def reset():
    global sim
    sim = Simulator()
    return get_state()

@app.post("/regime/{name}")
def set_regime(name: str):
    if name.upper() == "HIGH":
        sim.generator.set_regime(MarketRegime.HIGH_VOLATILITY)
    else:
        sim.generator.set_regime(MarketRegime.LOW_VOLATILITY)
    return {"status": "success", "regime": sim.generator.current_regime}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
