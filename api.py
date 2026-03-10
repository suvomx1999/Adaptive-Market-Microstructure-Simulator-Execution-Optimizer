from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from simulator.models import Side, Order, OrderType
from simulator.lob import OrderBook
from simulator.generators import OrderFlowGenerator, MarketRegime
from simulator.impact import MarketImpactModel
from simulator.strategies import TWAPStrategy, PredatoryHFTAgent, VWAPStrategy
from simulator.sentiment import SentimentGenerator
from simulator.rl_agent import PPOAgent
from simulator.data_loader import FI2010DataLoader
from pydantic import BaseModel
import numpy as np
import torch
import random
from typing import List, Dict, Any, Optional

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StrategyConfig(BaseModel):
    name: str
    params: Dict[str, Any]

# Global simulator state
class Simulator:
    def __init__(self):
        self.lob = OrderBook()
        self.generator = OrderFlowGenerator(base_lambda=5.0, use_hawkes=True)
        self.data_loader = FI2010DataLoader()
        self.use_real_data = False
        self.sentiment_gen = SentimentGenerator()
        self.predator = PredatoryHFTAgent(detection_window=5, detection_threshold=1.5)
        
        # Available Strategies
        self.strategies = {
            "QR-PPO": {
                "agent": PPOAgent(state_dim=5, action_dim=6, n_quantiles=32),
                "description": "State-of-the-art Distributional RL for tail-risk optimization.",
                "active": True
            },
            "TWAP": {
                "agent": TWAPStrategy(total_quantity=1000, horizon=100),
                "description": "Standard Time-Weighted Average Price baseline.",
                "active": False
            },
            "VWAP": {
                "agent": VWAPStrategy(total_quantity=1000, horizon=100),
                "description": "Volume-Weighted Average Price based on historical profile.",
                "active": False
            }
        }
        self.active_strategy = "QR-PPO"
        
        self.mid_price = 100.0
        self.history = []
        self.trades = []
        self.current_sentiment = 0.0
        self.is_predator_active = False
        self.signals = []
        
        # Warm up
        for _ in range(50):
            self.step()

    def set_active_strategy(self, name: str):
        if name in self.strategies:
            self.active_strategy = name
            for k in self.strategies:
                self.strategies[k]["active"] = (k == name)
            return True
        return False

    def update_strategy_params(self, name: str, params: Dict[str, Any]):
        if name == "QR-PPO":
            # For RL, params might be risk aversion or lr (mocked for now)
            pass
        elif name == "TWAP" or name == "VWAP":
            if "total_quantity" in params and "time_horizon" in params:
                if name == "TWAP":
                    self.strategies[name]["agent"] = TWAPStrategy(
                        total_quantity=params["total_quantity"], 
                        horizon=params["time_horizon"]
                    )
                else:
                    self.strategies[name]["agent"] = VWAPStrategy(
                        total_quantity=params["total_quantity"], 
                        horizon=params["time_horizon"]
                    )
        return True

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
        
        if self.use_real_data:
            # Use FI-2010 Data
            bids, asks = self.data_loader.get_lob_snapshot()
            # In real data, we just replace the LOB state or simulate trades
            # For simplicity in this demo, we'll sync the LOB to the snapshot
            self.lob = OrderBook() # Reset for snapshot
            for p, q in bids:
                self.lob.add_order(Order(random.randint(1,1000), Side.BUY, OrderType.LIMIT, p, q))
            for p, q in asks:
                self.lob.add_order(Order(random.randint(1,1000), Side.SELL, OrderType.LIMIT, p, q))
            
            # Advance index
            self.data_loader.get_next_state()
            current_time = self.data_loader.current_idx
            intensity = 0.0 # Not applicable for real data in this simple view
        else:
            # Generate event with Hawkes + Sentiment influence
            dt, order, is_cancel = self.generator.generate_event(self.mid_price)
            
            if not is_cancel and order:
                new_trades = self.lob.add_order(order)
                self.trades.extend([{"price": t.price, "qty": t.quantity, "time": t.timestamp} for t in new_trades])
            
            current_time = self.generator.current_time
            intensity = self.generator.hawkes.get_intensity(current_time)
        
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
            "time": current_time,
            "mid": self.mid_price,
            "bid": best_bid,
            "ask": best_ask,
            "intensity": intensity,
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
        # Use QR-PPO agent even if it's not the active execution strategy for visualization
        agent = self.strategies["QR-PPO"]["agent"]
        with torch.no_grad():
            _, quantiles = agent.policy(state_t)
        
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
        "intensity": sim.generator.hawkes.get_intensity(sim.generator.current_time) if not sim.use_real_data else 0,
        "quantiles": sim.get_distribution(),
        "volatility": vol,
        "imbalance": imbalance,
        "signals": sim.signals,
        "active_strategy": sim.active_strategy,
        "use_real_data": sim.use_real_data,
        "strategies": [
            {"name": k, "description": v["description"], "active": v["active"]} 
            for k, v in sim.strategies.items()
        ]
    }

@app.post("/toggle_data")
def toggle_data():
    sim.use_real_data = not sim.use_real_data
    if sim.use_real_data:
        sim.data_loader.reset()
    return {"use_real_data": sim.use_real_data}

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

@app.post("/strategy/active")
def set_active_strategy(config: StrategyConfig):
    if not sim.set_active_strategy(config.name):
        raise HTTPException(status_code=404, detail="Strategy not found")
    return get_state()

@app.post("/strategy/update")
def update_strategy_params(config: StrategyConfig):
    sim.update_strategy_params(config.name, config.params)
    return get_state()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
