# Adaptive Market Microstructure Simulator + Execution Optimizer

## 🏗️ Architecture Overview
This project implements a production-grade market microstructure simulator designed for testing and optimizing high-frequency execution strategies.

### Core Components
1. **LOB Engine ([lob.py](file:///Users/suvo/Downloads/Market%20Microstructure%20Simulator/simulator/lob.py))**: Event-driven continuous double auction book with FIFO priority.
2. **Order Flow Generator ([generators.py](file:///Users/suvo/Downloads/Market%20Microstructure%20Simulator/simulator/generators.py))**: Realistic simulation using Poisson processes and regime-dependent intensities.
3. **Market Impact Model ([impact.py](file:///Users/suvo/Downloads/Market%20Microstructure%20Simulator/simulator/impact.py))**: Almgren-Chriss style modeling with temporary (sqrt) and permanent (linear) impact.
4. **Execution Algorithms ([strategies.py](file:///Users/suvo/Downloads/Market%20Microstructure%20Simulator/simulator/strategies.py))**: Benchmarks including TWAP, VWAP, POV, and Almgren-Chriss Optimal Schedules.
5. **RL Agent ([rl_agent.py](file:///Users/suvo/Downloads/Market%20Microstructure%20Simulator/simulator/rl_agent.py))**: Adaptive DQN agent trained in a custom Gymnasium environment.
6. **Regime Detection ([regime.py](file:///Users/suvo/Downloads/Market%20Microstructure%20Simulator/simulator/regime.py))**: Real-time identification of volatility and liquidity shifts.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- `numpy`, `matplotlib`, `torch`, `gymnasium`

### Running Verifications
Each phase includes a dedicated verification script in the `scripts/` directory.
```bash
# Example: Run Phase 6 (Regime Adaptation)
export PYTHONPATH=$PYTHONPATH:.
python3 scripts/phase6_verify.py
```

## 🐳 Deployment (Docker)
The easiest way to deploy the full-stack application is using Docker.

1. **Build and start the containers**:
   ```bash
   docker-compose up --build
   ```
2. **Access the services**:
   - **Frontend Dashboard**: `http://localhost`
   - **Backend API**: `http://localhost:8000`

## ☁️ Cloud Deployment Options
- **Backend (FastAPI)**: Deploy on **Railway**, **Render**, or **AWS App Runner**.
- **Frontend (React)**: Deploy on **Vercel**, **Netlify**, or **GitHub Pages**.

---

## 📊 Microstructure Assumptions
- **Price-Time Priority**: Limit orders are matched based on price, then time of arrival.
- **Poisson Arrivals**: Market events are independent and follow an exponential inter-arrival distribution.
- **Market Impact**: Execution costs are non-linear and depend on current book depth and volatility.
- **Regime Dynamics**: The market shifts between Low Volatility/High Liquidity and High Volatility/Low Liquidity states.

---

## 📈 Performance Summary
Detailed metrics are available in the [FINAL_REPORT.md](file:///Users/suvo/Downloads/Market%20Microstructure%20Simulator/FINAL_REPORT.md).

### Key Findings:
- **Almgren-Chriss** significantly reduces variance in high-volatility environments compared to static TWAP.
- **RL Agent** learns to front-load execution when spread is narrow and imbalance is favorable.
- **Regime Adaptation** provides a ~15-20% improvement in Implementation Shortfall during sudden market shifts.
