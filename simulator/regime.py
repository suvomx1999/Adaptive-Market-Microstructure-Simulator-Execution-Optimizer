import numpy as np
from typing import List, Optional
from enum import Enum

class MarketState(Enum):
    LOW_VOL_HIGH_LIQ = 0
    HIGH_VOL_LOW_LIQ = 1

class HMMRegimeDetector:
    """
    Gaussian Hidden Markov Model for Regime Detection (Manual Implementation).
    States: 0 = Low Volatility, 1 = High Volatility
    """
    def __init__(self):
        # Initial state probabilities
        self.pi = np.array([0.9, 0.1])
        
        # Transition probabilities (high self-transition)
        self.A = np.array([[0.98, 0.02],
                          [0.05, 0.95]])
        
        # Emission parameters (Means and Std Devs for log-returns)
        self.means = np.array([0.0, 0.0])
        self.stds = np.array([0.002, 0.01]) # Low vs High vol
        
        self.history = []
        self.probs = self.pi.copy()

    def update(self, log_return: float) -> int:
        """
        Bayesian update of state probabilities (Viterbi-style filtered probability).
        """
        # 1. Prediction step: P(x_t | y_{t-1}) = sum_i P(x_t | x_{t-1}=i) * P(x_{t-1}=i | y_{t-1})
        pred_probs = self.probs @ self.A
        
        # 2. Update step (Likelihood)
        likelihoods = np.array([
            (1.0 / (self.stds[i] * np.sqrt(2 * np.pi))) * 
            np.exp(-0.5 * ((log_return - self.means[i]) / self.stds[i])**2)
            for i in [0, 1]
        ])
        
        # 3. Posterior
        new_probs = pred_probs * likelihoods
        sum_probs = np.sum(new_probs)
        
        if sum_probs > 0:
            self.probs = new_probs / sum_probs
        else:
            # Numerical stability: reset if probabilities collapse
            self.probs = self.pi.copy()
            
        return np.argmax(self.probs)

class RegimeDetector:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_history = []
        self.hmm = HMMRegimeDetector()
        self.current_regime = MarketState.LOW_VOL_HIGH_LIQ

    def update(self, mid_price: float, spread: float, volume: int) -> MarketState:
        self.price_history.append(mid_price)
        if len(self.price_history) < 2:
            return self.current_regime
            
        # Calculate log-return
        log_ret = np.log(self.price_history[-1] / self.price_history[-2])
        
        # HMM Update
        state_idx = self.hmm.update(log_ret)
        
        # Change point detection (Simple Bayesian Online approach)
        # If the high vol probability exceeds a threshold
        if state_idx == 1:
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
