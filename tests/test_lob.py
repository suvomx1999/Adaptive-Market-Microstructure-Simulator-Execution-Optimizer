import unittest
from simulator.models import Order, Side, OrderType
from simulator.lob import OrderBook

class TestLOB(unittest.TestCase):
    def setUp(self):
        self.lob = OrderBook()

    def test_limit_order_matching(self):
        # 1. Add some asks
        self.lob.add_order(Order(1, Side.SELL, OrderType.LIMIT, 100.5, 10))
        self.lob.add_order(Order(2, Side.SELL, OrderType.LIMIT, 101.0, 10))
        
        # 2. Add a buy limit order that matches
        trades = self.lob.add_order(Order(3, Side.BUY, OrderType.LIMIT, 100.5, 5))
        
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].price, 100.5)
        self.assertEqual(trades[0].quantity, 5)
        self.assertEqual(self.lob.get_best_ask(), 100.5)
        
        # 3. Add a buy limit order that clears the level
        trades = self.lob.add_order(Order(4, Side.BUY, OrderType.LIMIT, 100.5, 5))
        self.assertEqual(len(trades), 1)
        self.assertEqual(self.lob.get_best_ask(), 101.0)

    def test_fifo_priority(self):
        # Two orders at the same price
        self.lob.add_order(Order(1, Side.SELL, OrderType.LIMIT, 100.0, 10))
        self.lob.add_order(Order(2, Side.SELL, OrderType.LIMIT, 100.0, 10))
        
        # Taker order that matches only part of the first order
        trades = self.lob.add_order(Order(3, Side.BUY, OrderType.LIMIT, 100.0, 5))
        self.assertEqual(trades[0].maker_order_id, 1)
        self.assertEqual(trades[0].quantity, 5)
        
        # Taker order that matches the rest of the first order and part of the second
        trades = self.lob.add_order(Order(4, Side.BUY, OrderType.LIMIT, 100.0, 10))
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].maker_order_id, 1)
        self.assertEqual(trades[0].quantity, 5)
        self.assertEqual(trades[1].maker_order_id, 2)
        self.assertEqual(trades[1].quantity, 5)

    def test_partial_fills(self):
        self.lob.add_order(Order(1, Side.SELL, OrderType.LIMIT, 100.0, 10))
        
        # Taker order larger than maker order
        trades = self.lob.add_order(Order(2, Side.BUY, OrderType.LIMIT, 101.0, 15))
        
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].quantity, 10)
        
        # Check that the remaining part of the taker order is posted
        self.assertEqual(self.lob.get_best_bid(), 101.0)
        self.assertEqual(self.lob.get_best_ask(), None)

    def test_cancellation(self):
        self.lob.add_order(Order(1, Side.BUY, OrderType.LIMIT, 99.0, 10))
        self.assertTrue(self.lob.cancel_order(1))
        self.assertEqual(self.lob.get_best_bid(), None)
        self.assertFalse(self.lob.cancel_order(1)) # Already cancelled

    def test_market_orders(self):
        self.lob.add_order(Order(1, Side.SELL, OrderType.LIMIT, 100.0, 10))
        self.lob.add_order(Order(2, Side.SELL, OrderType.LIMIT, 101.0, 10))
        
        trades = self.lob.add_order(Order(3, Side.BUY, OrderType.MARKET, 0, 15))
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].price, 100.0)
        self.assertEqual(trades[0].quantity, 10)
        self.assertEqual(trades[1].price, 101.0)
        self.assertEqual(trades[1].quantity, 5)

    def test_spread_calculation(self):
        self.lob.add_order(Order(1, Side.BUY, OrderType.LIMIT, 99.0, 10))
        self.lob.add_order(Order(2, Side.SELL, OrderType.LIMIT, 101.0, 10))
        self.assertEqual(self.lob.get_spread(), 2.0)

if __name__ == '__main__':
    unittest.main()
