from enum import Enum
from dataclasses import dataclass, field
import time

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"

@dataclass
class Order:
    order_id: int
    side: Side
    order_type: OrderType
    price: float
    quantity: int
    timestamp: float = field(default_factory=time.time)
    filled_quantity: int = 0

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    def __repr__(self):
        return f"Order(id={self.order_id}, {self.side.value}, {self.order_type.value}, p={self.price}, q={self.quantity}, filled={self.filled_quantity})"

@dataclass
class Trade:
    trade_id: int
    maker_order_id: int
    taker_order_id: int
    side: Side  # Taker's side
    price: float
    quantity: int
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        return f"Trade(id={self.trade_id}, maker={self.maker_order_id}, taker={self.taker_order_id}, {self.side.value}, p={self.price}, q={self.quantity})"
