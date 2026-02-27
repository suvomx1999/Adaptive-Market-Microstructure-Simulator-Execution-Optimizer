import pandas as pd
import numpy as np
from datetime import datetime
from typing import Iterator, Dict
from .models import Order, Side, OrderType

class RealDataLoader:
    """
    Simulates loading real LOB data (e.g., NASDAQ ITCH or NSE tick data).
    Expects CSV with: timestamp, side, price, quantity, type
    """
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.data = None
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path: str):
        # In a real scenario, we'd read CSV/Parquet
        # For this project, we'll provide a method to ingest a dataframe
        self.data = pd.read_csv(file_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

    def generate_sample_data(self, n_rows: int = 1000):
        """Generates mock real-world data for validation"""
        times = pd.date_range(start="2024-01-01 09:15:00", periods=n_rows, freq='100ms')
        prices = 100 + np.cumsum(np.random.normal(0, 0.05, n_rows))
        sides = np.random.choice(['BUY', 'SELL'], n_rows)
        qtys = np.random.randint(1, 100, n_rows)
        types = np.random.choice(['LIMIT', 'MARKET'], n_rows, p=[0.8, 0.2])
        
        self.data = pd.DataFrame({
            'timestamp': times,
            'side': sides,
            'price': prices,
            'quantity': qtys,
            'type': types
        })
        return self.data

    def iter_orders(self) -> Iterator[Order]:
        if self.data is None:
            return
        
        for i, row in self.data.iterrows():
            side = Side.BUY if row['side'] == 'BUY' else Side.SELL
            order_type = OrderType.LIMIT if row['type'] == 'LIMIT' else OrderType.MARKET
            
            yield Order(
                order_id=i,
                side=side,
                price=float(row['price']),
                quantity=int(row['quantity']),
                order_type=order_type,
                timestamp=row['timestamp'].timestamp()
            )
