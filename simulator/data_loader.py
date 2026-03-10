import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import os

class FI2010DataLoader:
    """
    Data loader for the FI-2010 Limit Order Book dataset.
    The dataset contains 144 features:
    - 1-40: 10 levels of bid/ask prices and volumes
    - 41-144: Technical indicators
    """
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.data = None
        self.current_idx = 0
        
        if file_path and os.path.exists(file_path):
            self.load_data(file_path)
        else:
            print("FI-2010 file not found. Initializing with mock data for simulation.")
            self._generate_mock_data()

    def load_data(self, file_path: str):
        # FI-2010 is usually provided as .txt files with space-separated values
        # The first 40 columns are the LOB levels
        self.data = pd.read_csv(file_path, sep=' ', header=None)
        # Normalize/Scale if necessary
        self.data = self.data.values
        self.current_idx = 0

    def _generate_mock_data(self, num_samples: int = 1000):
        """Generates mock FI-2010 style data for testing"""
        # 144 features, but let's focus on the first 40 (10 levels)
        # Each level: [ask_price, ask_vol, bid_price, bid_vol]
        data = np.zeros((num_samples, 144))
        mid_price = 100.0
        for i in range(num_samples):
            mid_price += np.random.normal(0, 0.05)
            for level in range(10):
                # Ask side
                data[i, level*4] = mid_price + (level + 1) * 0.01
                data[i, level*4 + 1] = np.random.poisson(50)
                # Bid side
                data[i, level*4 + 2] = mid_price - (level + 1) * 0.01
                data[i, level*4 + 3] = np.random.poisson(50)
        self.data = data

    def get_next_state(self) -> Optional[np.ndarray]:
        if self.current_idx >= len(self.data):
            return None
        state = self.data[self.current_idx]
        self.current_idx += 1
        return state

    def get_lob_snapshot(self) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
        """
        Extracts the top 10 levels of the LOB from the current state.
        Returns: (bids, asks) where each is a list of (price, quantity)
        """
        state = self.data[self.current_idx]
        asks = []
        bids = []
        for i in range(10):
            # FI-2010 format: [P_ask1, V_ask1, P_bid1, V_bid1, ...]
            asks.append((state[i*4], int(state[i*4 + 1])))
            bids.append((state[i*4 + 2], int(state[i*4 + 3])))
        return bids, asks

    def reset(self):
        self.current_idx = 0
