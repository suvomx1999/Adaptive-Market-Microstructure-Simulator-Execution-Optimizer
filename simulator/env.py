import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any
from .models import Order, Side, OrderType
from .lob import OrderBook
from .generators import OrderFlowGenerator, MarketRegime
from .impact import MarketImpactModel

class TradingEnv(gym.Env):
    """
    Gym environment for optimal execution.
    """
    def __init__(self, 
                 total_quantity: int = 1000, 
                 horizon: int = 50,
                 target_side: Side = Side.BUY):
        super(TradingEnv, self).__init__()
        
        self.total_quantity = total_quantity
        self.horizon = horizon
        self.target_side = target_side
        
        # State: [Spread, Imbalance, Remaining Inventory %, Time Remaining %]
        # Normalized between 0 and 1 (or -1 and 1)
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, 0]), 
            high=np.array([1, 1, 1, 1]), 
            dtype=np.float32
        )
        
        # Actions: Fraction of remaining inventory to execute [0%, 5%, 10%, 20%, 50%, 100%]
        self.action_space = spaces.Discrete(6)
        self.action_map = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        
        # Components
        self.lob = OrderBook()
        self.generator = OrderFlowGenerator(base_lambda=10.0)
        self.impact_model = MarketImpactModel()
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.lob = OrderBook()
        self.impact_model.reset()
        self.remaining_qty = self.total_quantity
        self.current_step = 0
        self.arrival_price = 100.0 # Initial mid-price
        self.last_mid = self.arrival_price
        
        # Warm up the book with some initial orders
        for _ in range(50):
            _, order, is_cancel = self.generator.generate_event(self.arrival_price)
            if not is_cancel and order:
                self.lob.add_order(order)
                
        return self._get_obs(), {}

    def _get_obs(self):
        mid = self.lob.get_best_bid() or self.arrival_price # Simplified mid
        best_bid = self.lob.get_best_bid() or (mid - 0.01)
        best_ask = self.lob.get_best_ask() or (mid + 0.01)
        
        spread = (best_ask - best_bid) / mid
        
        # Imbalance = (Bid Vol - Ask Vol) / (Bid Vol + Ask Vol)
        bids, asks = self.lob.get_snapshot(1)
        bid_vol = bids[0][1] if bids else 1
        ask_vol = asks[0][1] if asks else 1
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
        obs = np.array([
            min(spread * 100, 1.0), # Normalized spread
            imbalance,
            self.remaining_qty / self.total_quantity,
            1.0 - (self.current_step / self.horizon)
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        # 1. Background market activity (Simulate a few events)
        mid = (self.lob.get_best_bid() + self.lob.get_best_ask()) / 2.0 if self.lob.get_best_bid() and self.lob.get_best_ask() else self.arrival_price
        for _ in range(5):
            _, order, is_cancel = self.generator.generate_event(mid)
            if is_cancel:
                # Cancel random order if possible (simplified)
                pass
            elif order:
                self.lob.add_order(order)

        # 2. Execute agent action
        frac = self.action_map[action]
        trade_qty = int(self.remaining_qty * frac)
        
        # If it's the last step, we MUST execute everything remaining
        if self.current_step == self.horizon - 1:
            trade_qty = self.remaining_qty
            
        reward = 0
        if trade_qty > 0:
            # Use impact model for execution price
            # Estimate liquidity at top levels
            bids, asks = self.lob.get_snapshot(3)
            liquidity = sum(v for p, v in (asks if self.target_side == Side.BUY else bids))
            
            current_mid = (self.lob.get_best_bid() + self.lob.get_best_ask()) / 2.0 if self.lob.get_best_bid() and self.lob.get_best_ask() else mid
            
            exec_price, temp_impact, perm_impact = self.impact_model.get_execution_price(
                trade_qty, self.target_side, current_mid, liquidity
            )
            
            # Implementation Shortfall reward (negative cost)
            # Cost per share relative to arrival price
            cost_per_share = (exec_price - self.arrival_price) if self.target_side == Side.BUY else (self.arrival_price - exec_price)
            reward = -cost_per_share * (trade_qty / self.total_quantity)
            
            # Update state
            self.remaining_qty -= trade_qty
            self.impact_model.update_state(trade_qty, self.target_side)
            
        # Time penalty / Leftover penalty
        self.current_step += 1
        done = (self.current_step >= self.horizon) or (self.remaining_qty <= 0)
        
        if done and self.remaining_qty > 0:
            # Heavy penalty for leftover inventory
            reward -= 10.0 * (self.remaining_qty / self.total_quantity)
            
        return self._get_obs(), reward, done, False, {}
