import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, LineChart, Line, Legend, Cell, ReferenceLine, PieChart, Pie
} from 'recharts';
import { 
  Activity, ArrowUp, ArrowDown, Settings, RefreshCw, 
  Play, Pause, Zap, ShieldAlert, BarChart3, Clock, Brain, 
  TrendingUp, AlertTriangle, Cpu, Terminal, Layers, 
  ZapOff, Globe, Database, ShieldCheck, Info, Radio, 
  Zap as ZapIcon, Fingerprint, Gauge, Crosshair
} from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8000`;

// Custom components for a modern look
const Card = ({ children, className = "", title, icon: Icon, badge }) => (
  <div className={`bg-slate-900/40 backdrop-blur-md border border-slate-800/50 rounded-2xl overflow-hidden flex flex-col shadow-2xl transition-all hover:border-slate-700/50 ${className}`}>
    {title && (
      <div className="px-6 py-4 border-b border-slate-800/50 flex justify-between items-center bg-slate-900/20">
        <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 flex items-center">
          {Icon && <Icon className="w-3.5 h-3.5 mr-2.5 text-blue-500" />}
          {title}
        </h3>
        {badge && (
          <span className="px-2 py-0.5 bg-blue-500/10 text-blue-400 text-[9px] font-black rounded border border-blue-500/20 uppercase tracking-tighter">
            {badge}
          </span>
        )}
      </div>
    )}
    <div className="p-6 flex-1">
      {children}
    </div>
  </div>
);

const Stat = ({ label, value, subValue, icon: Icon, color = "blue", trend }) => (
  <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800/50 p-6 rounded-2xl shadow-xl group hover:border-blue-500/30 transition-all">
    <div className="flex justify-between items-start mb-2">
      <span className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">{label}</span>
      <div className={`p-2 rounded-xl bg-${color}-500/10 border border-${color}-500/20 text-${color}-400 group-hover:scale-110 transition-transform`}>
        {Icon && <Icon className="w-4 h-4" />}
      </div>
    </div>
    <div className="flex items-baseline gap-2">
      <span className="text-2xl font-bold text-white tracking-tight">{value}</span>
      {trend !== undefined && (
        <span className={`text-[10px] font-bold ${parseFloat(trend) > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
          {parseFloat(trend) > 0 ? '▲' : '▼'} {Math.abs(trend)}%
        </span>
      )}
    </div>
    {subValue && <div className="text-[10px] text-slate-500 mt-1 font-medium tracking-tight">{subValue}</div>}
  </div>
);

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
      interval = setInterval(handleStep, 400);
    }
    return () => clearInterval(interval);
  }, [autoStep]);

  if (loading || !state) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 text-white">
        <div className="flex flex-col items-center">
          <div className="w-12 h-12 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin mb-4" />
          <span className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500">Initializing Quant Engine...</span>
        </div>
      </div>
    );
  }

  const depthData = [
    ...[...state.bids].reverse().map(b => ({ price: b.price, bid: b.qty, type: 'BID' })),
    ...state.asks.map(a => ({ price: a.price, ask: a.qty, type: 'ASK' }))
  ];

  const quantileData = state.quantiles.map((val, i) => ({
    index: i,
    value: val
  }));

  const midPriceTrend = state.history.length > 1 
    ? ((state.mid - state.history[0].mid) / state.history[0].mid * 100).toFixed(2)
    : 0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30 overflow-x-hidden">
      {/* Decorative background gradients */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden opacity-30">
        <div className="absolute -top-[10%] -left-[10%] w-[50%] h-[50%] bg-blue-600/10 blur-[120px] rounded-full" />
        <div className="absolute top-[40%] -right-[10%] w-[40%] h-[40%] bg-indigo-600/10 blur-[100px] rounded-full" />
        <div className="absolute bottom-[-10%] left-[20%] w-[30%] h-[30%] bg-rose-600/5 blur-[100px] rounded-full" />
      </div>

      <div className="relative z-10 max-w-[1800px] mx-auto p-6 lg:p-10">
        
        {/* Top Navbar */}
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-8 mb-12 border-b border-slate-900 pb-8">
          <div className="flex items-center gap-6">
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-900/40">
              <Crosshair className="text-white w-7 h-7" />
            </div>
            <div>
              <div className="flex items-center gap-3 mb-1">
                <span className="text-blue-500 text-[10px] font-black uppercase tracking-[0.3em]">Institutional Grade</span>
                <div className="h-1 w-1 bg-slate-700 rounded-full" />
                <span className="text-slate-500 text-[10px] font-black uppercase tracking-[0.3em]">v4.2.0-STABLE</span>
              </div>
              <h1 className="text-4xl font-black tracking-tighter text-white">
                NEURO<span className="text-blue-500">QUANT</span> <span className="text-slate-700 font-thin ml-2">SIMULATOR</span>
              </h1>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <div className="flex bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800 shadow-inner">
              <button 
                onClick={() => autoStep ? setAutoStep(false) : setAutoStep(true)}
                className={`px-6 py-2.5 rounded-xl text-xs font-black transition-all flex items-center tracking-widest uppercase ${autoStep ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-white'}`}
              >
                {autoStep ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                {autoStep ? 'LIVE' : 'HALT'}
              </button>
              <button 
                onClick={handleStep}
                className="px-6 py-2.5 rounded-xl text-xs font-black text-slate-500 hover:text-white hover:bg-slate-800 transition-all flex items-center tracking-widest uppercase"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                STEP
              </button>
            </div>

            <button 
              onClick={toggleRegime}
              className={`group relative flex items-center px-6 py-3 rounded-2xl border font-black text-[10px] uppercase tracking-[0.2em] transition-all overflow-hidden ${
                regime === 'HIGH' 
                  ? 'bg-rose-500/10 border-rose-500/50 text-rose-400' 
                  : 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400'
              }`}
            >
              <div className={`absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity ${regime === 'HIGH' ? 'bg-rose-500' : 'bg-emerald-500'}`} />
              {regime === 'HIGH' ? <ShieldAlert className="w-4 h-4 mr-2" /> : <ZapIcon className="w-4 h-4 mr-2" />}
              {regime === 'HIGH' ? 'High Vol Regime' : 'Stable Regime'}
            </button>
            
            <button onClick={handleReset} className="p-3 bg-slate-900 border border-slate-800 rounded-2xl text-slate-500 hover:text-white transition-colors">
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Top Grid: Core Vitals */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
          <Stat 
            label="Current Mid Price" 
            value={`$${state.mid.toFixed(2)}`} 
            subValue={`Bid: ${state.bids[0]?.price.toFixed(2)} | Ask: ${state.asks[0]?.price.toFixed(2)}`} 
            icon={TrendingUp} 
            trend={midPriceTrend}
          />
          <Stat 
            label="Hawkes Intensity λ(t)" 
            value={state.intensity.toFixed(3)} 
            subValue={`${(state.volatility * 100).toFixed(2)}% Realized Vol`} 
            icon={Activity} 
            color="indigo"
          />
          <Stat 
            label="Alpha Sentiment" 
            value={`${(state.sentiment * 100).toFixed(1)}%`} 
            subValue={state.sentiment > 0 ? "Momentum: Positive" : "Momentum: Negative"} 
            icon={Fingerprint} 
            color={state.sentiment > 0 ? "emerald" : "rose"}
          />
          <Stat 
            label="Order Imbalance" 
            value={`${(state.imbalance * 100).toFixed(1)}%`} 
            subValue={state.imbalance > 0 ? "Buy Side Pressure" : "Sell Side Pressure"} 
            icon={Gauge} 
            color="blue"
          />
        </div>

        {/* Middle Section: Advanced Charts */}
        <div className="grid grid-cols-12 gap-8 mb-10">
          
          {/* Main Chart: Dynamics */}
          <div className="col-span-12 lg:col-span-8">
            <Card title="Market Microstructure Dynamics" icon={Activity} badge="Self-Exciting Kernel">
              <div className="h-[600px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={state.history} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="colorMid" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="colorRegime" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#818cf8" stopOpacity={0.15}/>
                        <stop offset="95%" stopColor="#818cf8" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="colorSpread" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.1}/>
                        <stop offset="95%" stopColor="#94a3b8" stopOpacity={0.05}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.3} />
                    <XAxis dataKey="time" hide />
                    <YAxis yId="left" domain={['auto', 'auto']} stroke="#475569" fontSize={10} axisLine={false} tickLine={false} tickFormatter={v => `$${v.toFixed(2)}`} />
                    <YAxis yId="right" orientation="right" domain={[0, 'auto']} stroke="#818cf8" fontSize={10} axisLine={false} tickLine={false} tickFormatter={v => `${(v).toFixed(1)}`} />
                    
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', fontSize: '11px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' }} 
                      itemStyle={{ padding: '2px 0' }}
                      cursor={{ stroke: '#334155', strokeWidth: 1 }}
                    />

                    {/* Background Regime Probability */}
                    <Area yId="right" type="stepAfter" dataKey="regime_prob" stroke="#818cf8" strokeWidth={1} fill="url(#colorRegime)" fillOpacity={0.6} name="Regime Shift Prob" />
                    
                    {/* Bid-Ask Spread Area */}
                    <Area yId="left" type="monotone" dataKey="ask" stroke="none" fill="url(#colorSpread)" fillOpacity={0.5} name="Spread Band" />
                    <Area yId="left" type="monotone" dataKey="bid" stroke="none" fill="#0f172a" fillOpacity={1} />

                    {/* Main Price Line */}
                    <Area yId="left" type="monotone" dataKey="mid" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorMid)" animationDuration={500} name="Mid Price" />
                    
                    {/* Secondary Metrics */}
                    <Line yId="right" type="monotone" dataKey="intensity" stroke="#818cf8" strokeWidth={2} dot={false} strokeDasharray="4 4" name="Hawkes Intensity" />
                    <Line yId="left" type="monotone" dataKey="bid" stroke="#10b981" strokeWidth={1.5} dot={false} opacity={0.6} name="Best Bid" />
                    <Line yId="left" type="monotone" dataKey="ask" stroke="#f43f5e" strokeWidth={1.5} dot={false} opacity={0.6} name="Best Ask" />

                    {/* Reference Lines */}
                    <ReferenceLine yId="left" y={Math.max(...state.history.map(h => h.mid))} stroke="#334155" strokeDasharray="3 3" label={{ position: 'right', value: 'High', fill: '#475569', fontSize: 9 }} />
                    <ReferenceLine yId="left" y={Math.min(...state.history.map(h => h.mid))} stroke="#334155" strokeDasharray="3 3" label={{ position: 'right', value: 'Low', fill: '#475569', fontSize: 9 }} />

                    {state.predator_active && (
                      <ReferenceLine yId="left" x={state.history[state.history.length-1].time} stroke="#f43f5e" strokeDasharray="3 3" label={{ position: 'top', value: 'PREDATORY FOOTPRINT', fill: '#f43f5e', fontSize: 9, fontWeight: '900', letterSpacing: '0.1em' }} />
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          {/* Right: Intelligence Feed & Risks */}
          <div className="col-span-12 lg:col-span-4 space-y-8">
            <Card title="Live Intelligence Feed" icon={Radio} badge="Alpha Signals">
              <div className="space-y-4">
                {state.signals.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-10 opacity-20">
                    <Radio className="w-8 h-8 mb-2 animate-pulse" />
                    <span className="text-[10px] font-black uppercase tracking-widest">Scanning Market...</span>
                  </div>
                ) : (
                  state.signals.map((sig, i) => (
                    <div key={i} className="bg-slate-950/50 border border-slate-800/50 p-4 rounded-2xl flex items-start gap-4 group hover:border-blue-500/30 transition-all">
                      <div className={`p-2 rounded-lg ${sig.impact === 'Bullish' ? 'bg-emerald-500/10 text-emerald-400' : sig.impact === 'Bearish' ? 'bg-rose-500/10 text-rose-400' : 'bg-slate-500/10 text-slate-400'}`}>
                        {sig.impact === 'Bullish' ? <ArrowUp className="w-4 h-4" /> : sig.impact === 'Bearish' ? <ArrowDown className="w-4 h-4" /> : <Info className="w-4 h-4" />}
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-[10px] font-black text-white uppercase tracking-tighter">{sig.topic}</span>
                          <span className="text-[9px] font-bold text-blue-500">{(sig.confidence * 100).toFixed(0)}% CONF</span>
                        </div>
                        <p className="text-[11px] text-slate-500 leading-snug">
                          Detected {sig.impact.toLowerCase()} pressure shift. Adjusting execution schedule.
                        </p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </Card>

            <Card title="Distributional Risk" icon={Brain} badge="QR-PPO">
              <div className="h-40 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={quantileData}>
                    <Bar dataKey="value">
                      {quantileData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#10b981' : '#f43f5e'} fillOpacity={0.4} stroke={entry.value > 0 ? '#10b981' : '#f43f5e'} strokeWidth={1} />
                      ))}
                    </Bar>
                    <Tooltip cursor={{fill: 'transparent'}} contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', fontSize: '10px'}} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/50">
                  <div className="text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">95% CVaR</div>
                  <div className="text-xs font-bold text-rose-400">${Math.abs(Math.min(...state.quantiles)).toFixed(4)}</div>
                </div>
                <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/50">
                  <div className="text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">Exp. Return</div>
                  <div className="text-xs font-bold text-emerald-400">${(state.quantiles.reduce((a, b) => a + b, 0) / state.quantiles.length).toFixed(4)}</div>
                </div>
              </div>
            </Card>
          </div>
        </div>

        {/* Bottom Grid: Depth & Flow */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          <Card title="LOB Depth Analysis" icon={Layers} badge="ITCH Feed">
            <div className="space-y-1 font-mono">
              {state.asks.slice(0, 5).reverse().map((ask, i) => (
                <div key={`ask-${i}`} className="flex justify-between items-center group relative overflow-hidden h-8 px-4 rounded hover:bg-rose-500/10 transition-all border border-transparent hover:border-rose-500/20">
                  <div className="absolute right-0 top-0 bottom-0 bg-rose-500/5 transition-all group-hover:bg-rose-500/10" style={{ width: `${Math.min(ask.qty / 5, 100)}%` }} />
                  <span className="text-[10px] font-black text-rose-500/60 z-10">ASK</span>
                  <span className="text-xs font-bold text-rose-400 z-10">{ask.price.toFixed(2)}</span>
                  <span className="text-[10px] text-slate-500 z-10">{ask.qty}</span>
                </div>
              ))}
              <div className="h-10 flex items-center justify-center border-y border-slate-800/50 my-2 bg-slate-900/20">
                <span className="text-[9px] font-black text-slate-500 uppercase tracking-[0.5em]">SPREAD: {state.spread.toFixed(4)}</span>
              </div>
              {state.bids.slice(0, 5).map((bid, i) => (
                <div key={`bid-${i}`} className="flex justify-between items-center group relative overflow-hidden h-8 px-4 rounded hover:bg-emerald-500/10 transition-all border border-transparent hover:border-emerald-500/20">
                  <div className="absolute left-0 top-0 bottom-0 bg-emerald-500/5 transition-all group-hover:bg-emerald-500/10" style={{ width: `${Math.min(bid.qty / 5, 100)}%` }} />
                  <span className="text-[10px] font-black text-emerald-500/60 z-10">BID</span>
                  <span className="text-xs font-bold text-emerald-400 z-10">{bid.price.toFixed(2)}</span>
                  <span className="text-[10px] text-slate-500 z-10">{bid.qty}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Liquidity Profile" icon={BarChart3} badge="Volume Dist">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={depthData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.3} />
                  <XAxis dataKey="price" hide />
                  <YAxis stroke="#475569" fontSize={9} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', fontSize: '10px' }} />
                  <Bar dataKey="bid" fill="#10b981" radius={[2, 2, 0, 0]} fillOpacity={0.3} stroke="#10b981" strokeWidth={1} />
                  <Bar dataKey="ask" fill="#f43f5e" radius={[2, 2, 0, 0]} fillOpacity={0.3} stroke="#f43f5e" strokeWidth={1} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card title="FIX Execution Flow" icon={Terminal} badge="Secure Link">
            <div className="flex flex-col h-full space-y-3">
              {state.trades.length === 0 ? (
                <div className="flex-1 flex items-center justify-center opacity-20 border border-dashed border-slate-800 rounded-2xl">
                  <Database className="w-8 h-8" />
                </div>
              ) : (
                state.trades.slice(-6).reverse().map((t, i) => (
                  <div key={i} className="flex items-center gap-4 bg-slate-950/40 p-3 rounded-2xl border border-slate-800/50 hover:border-blue-500/30 transition-all">
                    <div className="w-8 h-8 bg-blue-500/10 rounded-lg flex items-center justify-center border border-blue-500/20">
                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse" />
                    </div>
                    <div className="flex-1">
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] font-black text-white tracking-tighter">FILL_CONFIRM</span>
                        <span className="text-[9px] font-mono text-slate-600">SEQ_{i+1024}</span>
                      </div>
                      <div className="flex justify-between items-end mt-1">
                        <span className="text-xs font-bold text-blue-400">${t.price.toFixed(2)}</span>
                        <span className="text-[9px] font-black text-slate-500 uppercase tracking-tighter">{t.qty} SHARES</span>
                      </div>
                    </div>
                  </div>
                ))
              )}
              <div className="pt-4 mt-auto border-t border-slate-900 flex items-center justify-between opacity-50">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
                  <span className="text-[8px] font-black uppercase tracking-widest">FIX 4.4 Encrypted</span>
                </div>
                <span className="text-[8px] font-mono">LAT: 6.3μs</span>
              </div>
            </div>
          </Card>

        </div>

        {/* System Footer */}
        <footer className="mt-16 pt-10 border-t border-slate-900 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-[9px] font-black text-slate-500 uppercase tracking-[0.3em]">Node: US-EAST-01-PROD</span>
            </div>
            <div className="flex items-center gap-3">
              <Globe className="w-4 h-4 text-slate-700" />
              <span className="text-[9px] font-black text-slate-500 uppercase tracking-[0.3em]">Global Sync: ACTIVE</span>
            </div>
          </div>
          
          <div className="flex items-center gap-6 text-slate-700">
            <span className="text-[9px] font-medium italic">© 2026 Neuro-Quant Research Framework</span>
            <div className="h-4 w-px bg-slate-900" />
            <span className="text-[9px] font-black uppercase tracking-[0.3em] text-blue-900/50">Restricted Access</span>
          </div>
        </footer>

      </div>
    </div>
  );
}

export default App;
