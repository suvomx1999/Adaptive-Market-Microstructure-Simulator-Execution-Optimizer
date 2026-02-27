import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from .models import Order, Side, OrderType

class MarketRegime:
    LOW_VOLATILITY = "LOW_VOLATILITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"

class OrderFlowGenerator:
    def __init__(self, 
                 base_lambda: float = 1.0, # Average arrivals per second
                 market_order_prob: float = 0.1,
                 limit_order_prob: float = 0.7,
                 cancel_order_prob: float = 0.2,
                 avg_order_size: int = 10,
                 price_std: float = 0.5):
        
        self.base_lambda = base_lambda
        self.market_order_prob = market_order_prob
        self.limit_order_prob = limit_order_prob
        self.cancel_order_prob = cancel_order_prob
        self.avg_order_size = avg_order_size
        self.price_std = price_std
        
        self.current_regime = MarketRegime.LOW_VOLATILITY
        self.order_id_counter = 1
        self.current_time = 0.0

    def set_regime(self, regime: str):
        self.current_regime = regime
        if regime == MarketRegime.HIGH_VOLATILITY:
            # Increase arrival rate and price volatility
            self.current_lambda = self.base_lambda * 3.0
            self.current_price_std = self.price_std * 2.5
        else:
            self.current_lambda = self.base_lambda
            self.current_price_std = self.price_std

    def generate_event(self, mid_price: float) -> Tuple[float, Optional[Order], bool]:
        """
        Generates the next event.
        Returns: (time_increment, Order object or None, is_cancellation)
        """
        # 1. Determine time until next arrival (Exponential distribution for Poisson process)
        # lambda is rate, 1/lambda is scale for exponential
        dt = np.random.exponential(1.0 / self.get_lambda())
        self.current_time += dt
        
        # 2. Determine event type
        rand_type = np.random.random()
        side = np.random.choice([Side.BUY, Side.SELL])
        quantity = np.random.poisson(self.avg_order_size) + 1
        
        if rand_type < self.limit_order_prob:
            # Limit Order
            # Price is relative to mid-price
            if side == Side.BUY:
                price = mid_price - abs(np.random.normal(0, self.get_price_std()))
            else:
                price = mid_price + abs(np.random.normal(0, self.get_price_std()))
            
            order = Order(self.order_id_counter, side, OrderType.LIMIT, round(price, 2), quantity, timestamp=self.current_time)
            self.order_id_counter += 1
            return dt, order, False
            
        elif rand_type < self.limit_order_prob + self.market_order_prob:
            # Market Order
            order = Order(self.order_id_counter, side, OrderType.MARKET, 0, quantity, timestamp=self.current_time)
            self.order_id_counter += 1
            return dt, order, False
            
        else:
            # Cancellation event (represented by None order and True flag)
            # Logic for which order to cancel will be handled by the simulator using OrderBook state
            return dt, None, True

    def get_lambda(self) -> float:
        return self.base_lambda * (3.0 if self.current_regime == MarketRegime.HIGH_VOLATILITY else 1.0)

    def get_price_std(self) -> float:
        return self.price_std * (2.5 if self.current_regime == MarketRegime.HIGH_VOLATILITY else 1.0)
