import numpy as np
from typing import Tuple, List
from .models import Order, Trade, Side, OrderType
from .lob import OrderBook

class MarketImpactModel:
    def __init__(self, 
                 temp_impact_param: float = 0.1,  # Corresponds to eta in Almgren-Chriss
                 perm_impact_param: float = 0.01, # Corresponds to gamma
                 volatility: float = 0.02):
        self.temp_impact_param = temp_impact_param
        self.perm_impact_param = perm_impact_param
        self.volatility = volatility
        self.permanent_impact_state = 0.0 # Cumulative permanent impact

    def get_execution_price(self, trade_size: int, side: Side, current_mid: float, market_liquidity: float) -> Tuple[float, float, float]:
        """
        Calculates the execution price including impact.
        Returns: (avg_execution_price, temporary_impact, permanent_impact)
        
        Using simplified Almgren-Chriss logic:
        - Temporary Impact (Slippage): h(v) = eta * |v| / liquidity
        - Permanent Impact (Price Move): g(v) = gamma * v
        """
        # Direction multiplier: +1 for BUY, -1 for SELL
        direction = 1 if side == Side.BUY else -1
        
        # 1. Temporary Impact (Cost of liquidity)
        # We model this as a widening of the effective spread for the trade
        # Adjusted by volatility
        temp_impact = self.temp_impact_param * self.volatility * (abs(trade_size) / max(market_liquidity, 1.0)) ** 0.5
        
        # 2. Permanent Impact (Information leakage / New equilibrium)
        perm_impact = self.perm_impact_param * self.volatility * direction * abs(trade_size)
        
        # 3. Calculate execution price
        # Exec Price = Mid + Temporary Impact + Pre-trade Permanent Impact
        avg_exec_price = current_mid + (direction * temp_impact) + self.permanent_impact_state
        
        return avg_exec_price, temp_impact, perm_impact

    def update_state(self, trade_size: int, side: Side):
        """Updates the internal state (cumulative permanent impact) after a trade"""
        direction = 1 if side == Side.BUY else -1
        perm_impact = self.perm_impact_param * self.volatility * direction * abs(trade_size)
        self.permanent_impact_state += perm_impact

    def reset(self):
        self.permanent_impact_state = 0.0

class ImpactAwareOrderBook(OrderBook):
    """
    An OrderBook wrapper that simulates market impact for large Market Orders.
    This is used to simulate the 'filled' price from the perspective of an execution agent,
    incorporating the theoretical impact model rather than just walking the book.
    """
    def __init__(self, impact_model: MarketImpactModel):
        super().__init__()
        self.impact_model = impact_model

    def execute_market_order_with_impact(self, order: Order) -> Tuple[List[Trade], float]:
        """
        Executes a market order and returns trades + calculated cost impact.
        Returns: (trades, total_impact_cost)
        """
        # 1. Get current market state before execution
        mid_price = self.get_mid_price()
        if mid_price is None:
            mid_price = 100.0 # Fallback
            
        # Estimate available liquidity (depth at top levels)
        bids, asks = self.get_snapshot(5)
        if order.side == Side.BUY:
            liquidity = sum(v for p, v in asks)
        else:
            liquidity = sum(v for p, v in bids)
            
        # 2. Calculate Theoretical Impact
        exec_price, temp_impact, perm_impact = self.impact_model.get_execution_price(
            order.quantity, order.side, mid_price, liquidity
        )
        
        # 3. Execute against the actual LOB (Walking the book)
        trades = self.add_order(order)
        
        # 4. Apply Permanent Impact to the Book (Shift all orders? Or just track it?)
        # In this simulation, we update the impact model state.
        # The LOB itself naturally adjusts as orders remove liquidity.
        # However, to simulate 'permanent' price moves, we might need to inject new orders
        # or shift existing ones. For now, we'll track the cost component.
        self.impact_model.update_state(order.quantity, order.side)
        
        # Calculate 'Realized' Slippage vs Theoretical
        realized_avg_price = sum(t.price * t.quantity for t in trades) / sum(t.quantity for t in trades) if trades else mid_price
        
        # Total cost impact (difference from mid)
        total_impact_cost = abs(realized_avg_price - mid_price)
        
        return trades, total_impact_cost

    def get_mid_price(self) -> float:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return (bid + ask) / 2.0
        return None
