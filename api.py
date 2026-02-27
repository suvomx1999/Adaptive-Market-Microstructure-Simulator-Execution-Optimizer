from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from simulator.models import Side, Order, OrderType
from simulator.lob import OrderBook
from simulator.generators import OrderFlowGenerator, MarketRegime
from simulator.impact import MarketImpactModel
from simulator.strategies import TWAPStrategy, PredatoryHFTAgent
from simulator.sentiment import SentimentGenerator
from simulator.rl_agent import PPOAgent
import numpy as np
import torch
import random
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
        self.generator = OrderFlowGenerator(base_lambda=5.0, use_hawkes=True)
        self.sentiment_gen = SentimentGenerator()
        self.predator = PredatoryHFTAgent(detection_window=5, detection_threshold=1.5)
        
        # RL Agent for Phase 11 (Distributional)
        self.agent = PPOAgent(state_dim=5, action_dim=6, n_quantiles=32)
        
        self.mid_price = 100.0
        self.history = []
        self.trades = []
        self.current_sentiment = 0.0
        self.is_predator_active = False
        self.signals = []
        
        # Warm up
        for _ in range(50):
            self.step()

    def generate_signal(self):
        """Generate a simulated news/alpha signal based on market state"""
        if random.random() < 0.1: # 10% chance per step
            topics = ["FED Interest Rate", "Tech Earnings", "CPI Data", "Liquidity Shock", "Sector Rotation"]
            impact = "Bullish" if self.current_sentiment > 0.2 else ("Bearish" if self.current_sentiment < -0.2 else "Neutral")
            signal = {
                "topic": random.choice(topics),
                "impact": impact,
                "confidence": round(random.uniform(0.6, 0.98), 2),
                "time": self.generator.current_time
            }
            self.signals.insert(0, signal)
            if len(self.signals) > 5:
                self.signals.pop()

    def step(self):
        # Update sentiment
        self.current_sentiment = self.sentiment_gen.update()
        
        # Generate event with Hawkes + Sentiment influence
        dt, order, is_cancel = self.generator.generate_event(self.mid_price)
        
        if not is_cancel and order:
            new_trades = self.lob.add_order(order)
            self.trades.extend([{"price": t.price, "qty": t.quantity, "time": t.timestamp} for t in new_trades])
        
        # Update predator
        bids, asks = self.lob.get_snapshot(3)
        self.predator.update(bids, asks)
        self.is_predator_active = self.predator.is_detecting
        
        # Update mid price
        best_bid = self.lob.get_best_bid()
        best_ask = self.lob.get_best_ask()
        if best_bid and best_ask:
            self.mid_price = (best_bid + best_ask) / 2.0
        
        # Generate Alpha Signal
        self.generate_signal()
        
        # Simulated Regime Probabilities (HMM output)
        vol = np.std([h["mid"] for h in self.history[-20:]]) if len(self.history) > 20 else 0.01
        regime_prob = min(vol * 5, 0.99) # Proxy for high volatility probability
        
        self.history.append({
            "time": self.generator.current_time,
            "mid": self.mid_price,
            "bid": best_bid,
            "ask": best_ask,
            "intensity": self.generator.hawkes.get_intensity(self.generator.current_time),
            "sentiment": self.current_sentiment,
            "regime_prob": regime_prob
        })
        if len(self.history) > 100:
            self.history.pop(0)

    def get_distribution(self):
        """Get the learned value distribution (quantiles) for the current state"""
        mid = self.mid_price
        best_bid = self.lob.get_best_bid() or (mid - 0.01)
        best_ask = self.lob.get_best_ask() or (mid + 0.01)
        spread = (best_ask - best_bid) / mid
        
        bids, asks = self.lob.get_snapshot(1)
        bid_vol = bids[0][1] if bids else 1
        ask_vol = asks[0][1] if asks else 1
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
        state = np.array([
            min(spread * 100, 1.0),
            imbalance,
            0.5, # Mock inventory 50%
            0.5, # Mock time 50%
            self.current_sentiment
        ], dtype=np.float32)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            _, quantiles = self.agent.policy(state_t)
        
        return quantiles.squeeze().tolist()

sim = Simulator()

@app.get("/state")
def get_state():
    bids, asks = sim.lob.get_snapshot(10)
    
    # Calculate additional metrics
    vol = np.std([h["mid"] for h in sim.history[-20:]]) if len(sim.history) > 20 else 0.01
    imbalance = 0
    if bids and asks:
        bid_vol = sum([q for p, q in bids[:3]])
        ask_vol = sum([q for p, q in asks[:3]])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

    return {
        "mid": sim.mid_price,
        "spread": sim.lob.get_spread(),
        "bids": [{"price": p, "qty": q} for p, q in bids],
        "asks": [{"price": p, "qty": q} for p, q in asks],
        "history": sim.history,
        "trades": sim.trades[-20:],
        "sentiment": sim.current_sentiment,
        "predator_active": sim.is_predator_active,
        "intensity": sim.generator.hawkes.get_intensity(sim.generator.current_time),
        "quantiles": sim.get_distribution(),
        "volatility": vol,
        "imbalance": imbalance,
        "signals": sim.signals
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
