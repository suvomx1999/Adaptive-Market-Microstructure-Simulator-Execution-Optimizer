import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from .models import Order, Side, OrderType

class MarketRegime:
    LOW_VOLATILITY = "LOW_VOLATILITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"

class HawkesProcessGenerator:
    """
    Hawkes Process for self-exciting order flow.
    Intensity lambda(t) = mu + alpha * sum(exp(-beta * (t - t_i)))
    """
    def __init__(self, mu: float = 1.0, alpha: float = 0.5, beta: float = 1.0):
        self.mu = mu        # Base intensity
        self.alpha = alpha  # Excitation coefficient (branching ratio)
        self.beta = beta    # Decay coefficient
        self.history = []   # Arrival times
        self.current_time = 0.0

    def get_intensity(self, t: float) -> float:
        if not self.history:
            return self.mu
        # Calculate kernel contribution from previous events
        kernel = np.sum(np.exp(-self.beta * (t - np.array(self.history))))
        return self.mu + self.alpha * self.beta * kernel

    def generate_next_arrival(self) -> float:
        """
        Ogata's thinning algorithm for Hawkes process generation.
        """
        while True:
            # 1. Estimate upper bound for intensity (intensity is monotonically decreasing between arrivals)
            lambda_upper = self.get_intensity(self.current_time)
            
            # 2. Generate arrival from Poisson process with lambda_upper
            dt = np.random.exponential(1.0 / lambda_upper)
            self.current_time += dt
            
            # 3. Acceptance/Rejection
            lambda_actual = self.get_intensity(self.current_time)
            if np.random.random() < (lambda_actual / lambda_upper):
                self.history.append(self.current_time)
                # Keep history manageable (last 100 events)
                if len(self.history) > 100:
                    self.history.pop(0)
                return dt

class OrderFlowGenerator:
    def __init__(self, 
                 base_lambda: float = 1.0,
                 market_order_prob: float = 0.1,
                 limit_order_prob: float = 0.7,
                 cancel_order_prob: float = 0.2,
                 avg_order_size: int = 10,
                 price_std: float = 0.5,
                 use_hawkes: bool = True):
        
        self.base_lambda = base_lambda
        self.market_order_prob = market_order_prob
        self.limit_order_prob = limit_order_prob
        self.cancel_order_prob = cancel_order_prob
        self.avg_order_size = avg_order_size
        self.price_std = price_std
        self.use_hawkes = use_hawkes
        
        self.hawkes = HawkesProcessGenerator(mu=base_lambda, alpha=0.6, beta=1.2)
        self.current_regime = MarketRegime.LOW_VOLATILITY
        self.order_id_counter = 1
        self.current_time = 0.0

    def set_regime(self, regime: str):
        self.current_regime = regime
        if regime == MarketRegime.HIGH_VOLATILITY:
            self.hawkes.mu = self.base_lambda * 3.0
            self.hawkes.alpha = 0.8 # Higher excitation in high vol
        else:
            self.hawkes.mu = self.base_lambda
            self.hawkes.alpha = 0.6

    def generate_event(self, mid_price: float) -> Tuple[float, Optional[Order], bool]:
        if self.use_hawkes:
            dt = self.hawkes.generate_next_arrival()
        else:
            dt = np.random.exponential(1.0 / self.get_lambda())
            
        self.current_time += dt
        
        rand_type = np.random.random()
        side = np.random.choice([Side.BUY, Side.SELL])
        quantity = np.random.poisson(self.avg_order_size) + 1
        
        if rand_type < self.limit_order_prob:
            price = mid_price + (np.random.normal(0, self.get_price_std()) if side == Side.SELL else -np.random.normal(0, self.get_price_std()))
            order = Order(self.order_id_counter, side, OrderType.LIMIT, round(price, 2), quantity, timestamp=self.current_time)
            self.order_id_counter += 1
            return dt, order, False
        elif rand_type < self.limit_order_prob + self.market_order_prob:
            order = Order(self.order_id_counter, side, OrderType.MARKET, 0, quantity, timestamp=self.current_time)
            self.order_id_counter += 1
            return dt, order, False
        else:
            return dt, None, True

    def get_lambda(self) -> float:
        return self.base_lambda * (3.0 if self.current_regime == MarketRegime.HIGH_VOLATILITY else 1.0)

    def get_price_std(self) -> float:
        return self.price_std * (2.5 if self.current_regime == MarketRegime.HIGH_VOLATILITY else 1.0)

class MultiAssetGenerator:
    """
    Generates orders for two correlated assets with a lead-lag relationship.
    Asset A leads Asset B.
    """
    def __init__(self, base_lambda: float = 2.0, correlation: float = 0.8, lag_steps: int = 5):
        self.gen_a = OrderFlowGenerator(base_lambda=base_lambda, use_hawkes=True)
        self.gen_b = OrderFlowGenerator(base_lambda=base_lambda, use_hawkes=True)
        self.correlation = correlation
        self.lag_steps = lag_steps
        self.price_history_a = []
        self.mid_a = 100.0
        self.mid_b = 100.0

    def generate_events(self) -> List[Tuple[str, float, Optional[Order], bool]]:
        """
        Generates events for both assets. 
        Asset B's mid-price is influenced by Asset A's history.
        """
        # 1. Generate event for Asset A
        dt_a, order_a, cancel_a = self.gen_a.generate_event(self.mid_a)
        if order_a and order_a.order_type == OrderType.LIMIT:
            self.mid_a = order_a.price
            self.price_history_a.append(self.mid_a)
            if len(self.price_history_a) > self.lag_steps + 1:
                self.price_history_a.pop(0)

        # 2. Update Asset B's mid-price based on Asset A's lagged price
        if len(self.price_history_a) > self.lag_steps:
            lagged_a = self.price_history_a[0]
            # Simple mean-reversion towards correlated lagged price
            self.mid_b += self.correlation * (lagged_a - self.mid_b) * 0.1
            
        dt_b, order_b, cancel_b = self.gen_b.generate_event(self.mid_b)
        
        # Assign asset IDs
        if order_a: order_a.asset_id = "A"
        if order_b: order_b.asset_id = "B"
        
        events = [("A", dt_a, order_a, cancel_a), ("B", dt_b, order_b, cancel_b)]
        return events
