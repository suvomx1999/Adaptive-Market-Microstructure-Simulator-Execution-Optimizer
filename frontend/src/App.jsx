import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, LineChart, Line, Legend 
} from 'recharts';
import { 
  Activity, ArrowUp, ArrowDown, Settings, RefreshCw, 
  Play, Pause, Zap, ShieldAlert, BarChart3, Clock 
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [state, setState] = useState(null);
  const [regime, setRegime] = useState('LOW');
  const [autoStep, setAutoStep] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const res = await axios.get(`${API_BASE}/state`);
      setState(res.data);
      setLoading(false);
    } catch (err) {
      console.error("API Error:", err);
    }
  };

  const handleStep = async () => {
    try {
      const res = await axios.post(`${API_BASE}/step`);
      setState(res.data);
    } catch (err) {
      console.error("Step Error:", err);
    }
  };

  const handleReset = async () => {
    try {
      const res = await axios.post(`${API_BASE}/reset`);
      setState(res.data);
    } catch (err) {
      console.error("Reset Error:", err);
    }
  };

  const toggleRegime = async () => {
    const newRegime = regime === 'LOW' ? 'HIGH' : 'LOW';
    try {
      await axios.post(`${API_BASE}/regime/${newRegime}`);
      setRegime(newRegime);
    } catch (err) {
      console.error("Regime Error:", err);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    let interval;
    if (autoStep) {
      interval = setInterval(handleStep, 500);
    }
    return () => clearInterval(interval);
  }, [autoStep]);

  if (loading || !state) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 text-white">
        <RefreshCw className="w-8 h-8 animate-spin mr-3 text-blue-500" />
        <span className="text-xl font-medium tracking-tight">Initializing Market Engine...</span>
      </div>
    );
  }

  // Format depth data for chart
  const depthData = [
    ...[...state.bids].reverse().map(b => ({ price: b.price, bid: b.qty })),
    ...state.asks.map(a => ({ price: a.price, ask: a.qty }))
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans p-6">
      {/* Header */}
      <header className="flex justify-between items-center mb-8 bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
            Adaptive Market Microstructure Simulator
          </h1>
          <p className="text-slate-500 mt-1 flex items-center">
            <Clock className="w-4 h-4 mr-2" />
            Production-grade HFT Environment
          </p>
        </div>
        
        <div className="flex gap-4">
          <button 
            onClick={toggleRegime}
            className={`flex items-center px-4 py-2 rounded-xl border transition-all ${
              regime === 'HIGH' 
                ? 'bg-red-500/10 border-red-500/50 text-red-400 shadow-[0_0_15px_rgba(239,68,68,0.2)]' 
                : 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400'
            }`}
          >
            {regime === 'HIGH' ? <ShieldAlert className="w-4 h-4 mr-2" /> : <Zap className="w-4 h-4 mr-2" />}
            Regime: {regime === 'HIGH' ? 'High Volatility' : 'Stable Market'}
          </button>
          
          <div className="h-10 w-px bg-slate-800" />
          
          <button onClick={handleReset} className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors">
            <RefreshCw className="w-6 h-6" />
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        
        {/* Left Column: Stats & LOB */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          {/* Quick Stats */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-900 p-5 rounded-2xl border border-slate-800">
              <span className="text-slate-500 text-sm font-medium">Mid Price</span>
              <div className="text-2xl font-bold mt-1 text-white">${state.mid.toFixed(2)}</div>
            </div>
            <div className="bg-slate-900 p-5 rounded-2xl border border-slate-800">
              <span className="text-slate-500 text-sm font-medium">Spread</span>
              <div className="text-2xl font-bold mt-1 text-blue-400">{state.spread.toFixed(4)}</div>
            </div>
          </div>

          {/* Order Book Table */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 overflow-hidden shadow-lg">
            <div className="p-4 bg-slate-800/50 border-b border-slate-800 flex justify-between items-center">
              <h3 className="font-semibold flex items-center"><Activity className="w-4 h-4 mr-2 text-indigo-400" /> Order Book</h3>
              <span className="text-xs text-slate-500 uppercase tracking-wider">Top 10 Levels</span>
            </div>
            <div className="p-0">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-800/50">
                    <th className="py-2 px-4 text-left font-medium">Side</th>
                    <th className="py-2 px-4 text-right font-medium">Price</th>
                    <th className="py-2 px-4 text-right font-medium">Quantity</th>
                  </tr>
                </thead>
                <tbody>
                  {state.asks.slice(0, 5).reverse().map((ask, i) => (
                    <tr key={`ask-${i}`} className="text-red-400 bg-red-500/5 hover:bg-red-500/10 transition-colors">
                      <td className="py-2 px-4 font-medium opacity-50">ASK</td>
                      <td className="py-2 px-4 text-right font-mono font-bold">{ask.price.toFixed(2)}</td>
                      <td className="py-2 px-4 text-right font-mono">{ask.qty}</td>
                    </tr>
                  ))}
                  <tr className="bg-slate-800/30">
                    <td colSpan="3" className="py-1 px-4 text-center text-[10px] text-slate-600 font-bold uppercase tracking-[0.2em]">Spread Window</td>
                  </tr>
                  {state.bids.slice(0, 5).map((bid, i) => (
                    <tr key={`bid-${i}`} className="text-emerald-400 bg-emerald-500/5 hover:bg-emerald-500/10 transition-colors">
                      <td className="py-2 px-4 font-medium opacity-50">BID</td>
                      <td className="py-2 px-4 text-right font-mono font-bold">{bid.price.toFixed(2)}</td>
                      <td className="py-2 px-4 text-right font-mono">{bid.qty}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Center/Right Column: Charts */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          
          {/* Controls */}
          <div className="bg-slate-900 p-4 rounded-2xl border border-slate-800 flex items-center gap-6 shadow-md">
            <button 
              onClick={handleStep}
              className="flex items-center px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl transition-all shadow-lg shadow-blue-900/20 active:scale-95"
            >
              <Play className="w-4 h-4 mr-2" /> Simulation Step
            </button>
            <label className="flex items-center cursor-pointer group">
              <div className="relative">
                <input 
                  type="checkbox" 
                  className="sr-only" 
                  checked={autoStep}
                  onChange={(e) => setAutoStep(e.target.checked)}
                />
                <div className={`block w-14 h-8 rounded-full transition-colors ${autoStep ? 'bg-blue-600' : 'bg-slate-700'}`}></div>
                <div className={`dot absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition-transform ${autoStep ? 'translate-x-6' : ''}`}></div>
              </div>
              <div className="ml-3 text-slate-300 font-medium group-hover:text-white transition-colors">Auto-Simulation</div>
            </label>
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* LOB Depth Chart */}
            <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl h-80">
              <h3 className="font-semibold mb-6 flex items-center"><BarChart3 className="w-4 h-4 mr-2 text-blue-400" /> Market Depth</h3>
              <ResponsiveContainer width="100%" height="80%">
                <BarChart data={depthData}>
                  <XAxis dataKey="price" stroke="#475569" fontSize={10} hide />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                    itemStyle={{ fontSize: '12px' }}
                  />
                  <Bar dataKey="bid" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="ask" fill="#ef4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Price History */}
            <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl h-80">
              <h3 className="font-semibold mb-6 flex items-center"><Activity className="w-4 h-4 mr-2 text-indigo-400" /> Price Evolution</h3>
              <ResponsiveContainer width="100%" height="80%">
                <LineChart data={state.history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={['auto', 'auto']} stroke="#475569" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                  />
                  <Line type="monotone" dataKey="mid" stroke="#6366f1" strokeWidth={3} dot={false} animationDuration={300} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Trade Ticker */}
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
            <h3 className="font-semibold mb-4 flex items-center"><Clock className="w-4 h-4 mr-2 text-slate-400" /> Recent Trades</h3>
            <div className="flex flex-wrap gap-3">
              {state.trades.length === 0 ? (
                <span className="text-slate-600 italic text-sm">Waiting for execution...</span>
              ) : (
                state.trades.map((t, i) => (
                  <div key={`t-${i}`} className="bg-slate-800/50 border border-slate-700 px-3 py-1.5 rounded-lg text-xs font-mono">
                    <span className="text-blue-400 mr-2">${t.price.toFixed(2)}</span>
                    <span className="text-slate-400">{t.qty} shares</span>
                  </div>
                ))
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
