import heapq
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from .models import Order, Trade, Side, OrderType
import time

class OrderBook:
    def __init__(self):
        # Bids: Max-heap (using negative prices)
        self.bid_prices: List[float] = []
        self.bids: Dict[float, deque[Order]] = defaultdict(deque)
        
        # Asks: Min-heap
        self.ask_prices: List[float] = []
        self.asks: Dict[float, deque[Order]] = defaultdict(deque)
        
        # All active orders for O(1) lookup and cancellation
        self.orders: Dict[int, Order] = {}
        
        self.trade_id_counter = 0
        self.trades: List[Trade] = []

    def get_best_bid(self) -> Optional[float]:
        while self.bid_prices:
            best_bid_neg = self.bid_prices[0]
            price = -best_bid_neg
            if self.bids[price]:
                return price
            heapq.heappop(self.bid_prices)
        return None

    def get_best_ask(self) -> Optional[float]:
        while self.ask_prices:
            best_ask = self.ask_prices[0]
            if self.asks[best_ask]:
                return best_ask
            heapq.heappop(self.ask_prices)
        return None

    def get_spread(self) -> Optional[float]:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None

    def add_order(self, order: Order) -> List[Trade]:
        if order.order_type == OrderType.MARKET:
            return self._match_market(order)
        elif order.order_type == OrderType.LIMIT:
            return self._match_limit(order)
        return []

    def cancel_order(self, order_id: int) -> bool:
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        side_dict = self.bids if order.side == Side.BUY else self.asks
        
        # Remove from deque
        try:
            side_dict[order.price].remove(order)
            if not side_dict[order.price]:
                del side_dict[order.price]
            del self.orders[order_id]
            return True
        except ValueError:
            # Order might have been partially/fully filled already
            return False

    def _match_limit(self, taker_order: Order) -> List[Trade]:
        trades = []
        if taker_order.side == Side.BUY:
            # Match with asks
            while taker_order.remaining_quantity > 0:
                best_ask = self.get_best_ask()
                if best_ask is None or best_ask > taker_order.price:
                    break
                
                trades.extend(self._match_at_price(taker_order, self.asks[best_ask], best_ask))
                if not self.asks[best_ask]:
                    del self.asks[best_ask]
                    heapq.heappop(self.ask_prices)
                    
            if taker_order.remaining_quantity > 0:
                self._post_limit_order(taker_order)
                
        else: # Side.SELL
            # Match with bids
            while taker_order.remaining_quantity > 0:
                best_bid = self.get_best_bid()
                if best_bid is None or best_bid < taker_order.price:
                    break
                
                trades.extend(self._match_at_price(taker_order, self.bids[best_bid], best_bid))
                if not self.bids[best_bid]:
                    del self.bids[best_bid]
                    heapq.heappop(self.bid_prices)
                    
            if taker_order.remaining_quantity > 0:
                self._post_limit_order(taker_order)
                
        return trades

    def _match_market(self, taker_order: Order) -> List[Trade]:
        trades = []
        if taker_order.side == Side.BUY:
            while taker_order.remaining_quantity > 0:
                best_ask = self.get_best_ask()
                if best_ask is None:
                    break # Out of liquidity
                
                trades.extend(self._match_at_price(taker_order, self.asks[best_ask], best_ask))
                if not self.asks[best_ask]:
                    del self.asks[best_ask]
                    heapq.heappop(self.ask_prices)
        else: # Side.SELL
            while taker_order.remaining_quantity > 0:
                best_bid = self.get_best_bid()
                if best_bid is None:
                    break # Out of liquidity
                
                trades.extend(self._match_at_price(taker_order, self.bids[best_bid], best_bid))
                if not self.bids[best_bid]:
                    del self.bids[best_bid]
                    heapq.heappop(self.bid_prices)
                    
        return trades

    def _match_at_price(self, taker_order: Order, price_queue: deque[Order], price: float) -> List[Trade]:
        trades = []
        while taker_order.remaining_quantity > 0 and price_queue:
            maker_order = price_queue[0]
            match_quantity = min(taker_order.remaining_quantity, maker_order.remaining_quantity)
            
            # Update quantities
            taker_order.filled_quantity += match_quantity
            maker_order.filled_quantity += match_quantity
            
            trade = Trade(
                trade_id=self.trade_id_counter,
                maker_order_id=maker_order.order_id,
                taker_order_id=taker_order.order_id,
                side=taker_order.side,
                price=price,
                quantity=match_quantity,
                timestamp=time.time()
            )
            self.trade_id_counter += 1
            trades.append(trade)
            self.trades.append(trade)
            
            if maker_order.remaining_quantity == 0:
                price_queue.popleft()
                if maker_order.order_id in self.orders:
                    del self.orders[maker_order.order_id]
                    
        return trades

    def _post_limit_order(self, order: Order):
        self.orders[order.order_id] = order
        if order.side == Side.BUY:
            if not self.bids[order.price]:
                heapq.heappush(self.bid_prices, -order.price)
            self.bids[order.price].append(order)
        else:
            if not self.asks[order.price]:
                heapq.heappush(self.ask_prices, order.price)
            self.asks[order.price].append(order)

    def get_snapshot(self, n_levels: int = 5) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
        """Returns (bids, asks) snapshot as list of (price, total_quantity)"""
        bid_snapshot = []
        temp_bid_prices = sorted(self.bids.keys(), reverse=True)
        for p in temp_bid_prices[:n_levels]:
            vol = sum(o.remaining_quantity for o in self.bids[p])
            bid_snapshot.append((p, vol))
            
        ask_snapshot = []
        temp_ask_prices = sorted(self.asks.keys())
        for p in temp_ask_prices[:n_levels]:
            vol = sum(o.remaining_quantity for o in self.asks[p])
            ask_snapshot.append((p, vol))
            
        return bid_snapshot, ask_snapshot
