import numpy as np

class SentimentGenerator:
    """
    Generates a stochastic sentiment score S_t in [-1, 1].
    S_t follows an Ornstein-Uhlenbeck process to mimic mean-reverting sentiment.
    """
    def __init__(self, theta=0.1, mu=0.0, sigma=0.05):
        self.theta = theta  # Speed of mean reversion
        self.mu = mu        # Long-term mean sentiment
        self.sigma = sigma  # Volatility of sentiment
        self.current_sentiment = 0.0

    def update(self) -> float:
        """
        Updates and returns the next sentiment score.
        """
        # dS_t = theta * (mu - S_t) * dt + sigma * dW_t
        dt = 1.0
        dw = np.random.normal(0, np.sqrt(dt))
        ds = self.theta * (self.mu - self.current_sentiment) * dt + self.sigma * dw
        self.current_sentiment += ds
        
        # Clip to [-1, 1]
        self.current_sentiment = np.clip(self.current_sentiment, -1.0, 1.0)
        return self.current_sentiment

    def set_shock(self, value: float):
        """Manually set a sentiment shock (e.g., news event)"""
        self.current_sentiment = np.clip(value, -1.0, 1.0)
