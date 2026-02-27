import numpy as np
from typing import List, Tuple, Dict, Optional
from .models import Side

class ExecutionStrategy:
    def __init__(self, total_quantity: int, horizon: int):
        self.total_quantity = total_quantity
        self.horizon = horizon # Total time steps

    def get_schedule(self, **kwargs) -> np.ndarray:
        """Returns an array of quantities to execute at each step"""
        raise NotImplementedError

class TWAPStrategy(ExecutionStrategy):
    def get_schedule(self, **kwargs) -> np.ndarray:
        # Constant rate
        base_qty = self.total_quantity // self.horizon
        remainder = self.total_quantity % self.horizon
        
        schedule = np.full(self.horizon, base_qty)
        # Distribute remainder
        for i in range(remainder):
            schedule[i] += 1
            
        return schedule

class VWAPStrategy(ExecutionStrategy):
    def get_schedule(self, market_volume_profile: np.ndarray, **kwargs) -> np.ndarray:
        """
        Executes proportional to market volume profile.
        market_volume_profile: expected volume at each step
        """
        total_market_vol = np.sum(market_volume_profile)
        proportions = market_volume_profile / total_market_vol
        
        schedule = (proportions * self.total_quantity).astype(int)
        
        # Adjust for rounding errors
        diff = self.total_quantity - np.sum(schedule)
        if diff > 0:
            for i in range(int(diff)):
                schedule[i % self.horizon] += 1
        elif diff < 0:
            for i in range(int(abs(diff))):
                idx = np.argmax(schedule)
                schedule[idx] -= 1
                
        return schedule

class POVStrategy(ExecutionStrategy):
    def __init__(self, total_quantity: int, pov_rate: float = 0.1):
        # POV doesn't have a fixed horizon, it ends when quantity is filled
        super().__init__(total_quantity, horizon=0)
        self.pov_rate = pov_rate

    def get_next_trade(self, current_market_vol: int, remaining_qty: int) -> int:
        trade_size = int(current_market_vol * self.pov_rate)
        return min(trade_size, remaining_qty)

class AlmgrenChrissStrategy(ExecutionStrategy):
    def __init__(self, total_quantity: int, horizon: int, 
                 risk_aversion: float = 0.1, 
                 eta: float = 0.1, # Temp impact param
                 gamma: float = 0.01, # Perm impact param
                 volatility: float = 0.02):
        super().__init__(total_quantity, horizon)
        self.risk_aversion = risk_aversion
        self.eta = eta
        self.gamma = gamma
        self.volatility = volatility

    def get_schedule(self) -> np.ndarray:
        X = self.total_quantity
        T = self.horizon
        kappa_sq = (self.risk_aversion * (self.volatility**2)) / self.eta
        kappa = np.sqrt(kappa_sq)
        
        trajectories = []
        for j in range(T + 1):
            if kappa == 0:
                xj = X * (1 - j/T)
            else:
                num = np.sinh(kappa * (T - j))
                den = np.sinh(kappa * T)
                xj = X * (num / den)
            trajectories.append(xj)
            
        trades = []
        for i in range(T):
            trade = int(round(trajectories[i] - trajectories[i+1]))
            trades.append(trade)
            
        schedule = np.array(trades)
        diff = X - np.sum(schedule)
        if diff != 0:
            schedule[-1] += diff
            
        return schedule

class PredatoryHFTAgent:
    """
    Predatory HFT agent that monitors LOB imbalance and detects large orders.
    If a persistent imbalance is detected (e.g., from a large execution agent), 
    the predatory agent "front-runs" by placing orders at the same price.
    """
    def __init__(self, detection_window: int = 5, detection_threshold: float = 2.0):
        self.window = detection_window
        self.threshold = detection_threshold
        self.imbalance_history = []
        self.is_detecting = False

    def update(self, bids: List[Tuple[float, int]], asks: List[Tuple[float, int]]) -> Optional[Tuple[Side, int, float]]:
        """
        Updates agent state and returns an order if a large trader is detected.
        """
        if not bids or not asks:
            return None
            
        total_bid = sum(q for p, q in bids[:3]) # Top 3 levels
        total_ask = sum(q for p, q in asks[:3])
        
        imbalance = total_bid / max(total_ask, 1e-6)
        self.imbalance_history.append(imbalance)
        
        if len(self.imbalance_history) > self.window:
            self.imbalance_history.pop(0)
            
        avg_imbalance = np.mean(self.imbalance_history)
        
        # Detection: Large imbalance (> threshold) suggests a large buyer
        if avg_imbalance > self.threshold:
            self.is_detecting = True
            best_bid = bids[0][0]
            return Side.BUY, 50, best_bid
            
        elif avg_imbalance < (1.0 / self.threshold):
            self.is_detecting = True
            best_ask = asks[0][0]
            return Side.SELL, 50, best_ask
            
        self.is_detecting = False
        return None
