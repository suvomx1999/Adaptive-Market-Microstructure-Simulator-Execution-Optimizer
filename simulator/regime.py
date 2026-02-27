import numpy as np
from typing import List, Optional
from enum import Enum

class MarketState(Enum):
    LOW_VOL_HIGH_LIQ = 0
    HIGH_VOL_LOW_LIQ = 1

class RegimeDetector:
    def __init__(self, window_size: int = 20, threshold: float = 1.5):
        self.window_size = window_size
        self.threshold = threshold # Sensitivity for change detection
        self.vol_history = []
        self.liq_history = []
        self.current_regime = MarketState.LOW_VOL_HIGH_LIQ
        
        # Benchmarks (Baseline for Low Vol)
        self.baseline_vol = None
        self.baseline_liq = None

    def update(self, mid_price: float, spread: float, volume: int) -> MarketState:
        """
        Updates detector with latest market data.
        Returns the detected regime.
        """
        self.vol_history.append(mid_price)
        self.liq_history.append(volume / max(spread, 1e-6)) # Proxy for liquidity
        
        if len(self.vol_history) > self.window_size:
            self.vol_history.pop(0)
            self.liq_history.pop(0)
            
        if len(self.vol_history) < self.window_size:
            return self.current_regime

        # Calculate local metrics
        returns = np.diff(self.vol_history) / self.vol_history[:-1]
        local_vol = np.std(returns)
        local_liq = np.mean(self.liq_history)
        
        # Initialize baselines if not set
        if self.baseline_vol is None:
            self.baseline_vol = local_vol
            self.baseline_liq = local_liq
            return self.current_regime

        # Detection logic: Bayesian-style shift detection
        # If local vol is significantly higher than baseline, or liquidity is lower
        vol_ratio = local_vol / max(self.baseline_vol, 1e-9)
        liq_ratio = local_liq / max(self.baseline_liq, 1e-9)
        
        if vol_ratio > self.threshold or liq_ratio < (1.0 / self.threshold):
            self.current_regime = MarketState.HIGH_VOL_LOW_LIQ
        else:
            self.current_regime = MarketState.LOW_VOL_HIGH_LIQ
            
        return self.current_regime

class AdaptiveStrategy:
    """
    Wraps execution strategies and switches between them based on detected regime.
    """
    def __init__(self, agg_strategy, pass_strategy):
        self.agg_strategy = agg_strategy # Used for High Vol
        self.pass_strategy = pass_strategy # Used for Low Vol
        self.detector = RegimeDetector()

    def get_action(self, mid_price: float, spread: float, volume: int, remaining_qty: int, time_left: int):
        regime = self.detector.update(mid_price, spread, volume)
        
        if regime == MarketState.HIGH_VOL_LOW_LIQ:
            # Aggressive strategy (e.g. Almgren-Chriss with high risk aversion)
            return self.agg_strategy.get_next_trade(remaining_qty, time_left)
        else:
            # Passive strategy (e.g. TWAP or low risk aversion AC)
            return self.pass_strategy.get_next_trade(remaining_qty, time_left)
