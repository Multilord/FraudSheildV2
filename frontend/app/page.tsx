"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  Shield, AlertTriangle, XCircle, Activity, RefreshCw,
  Wallet, TrendingUp, Clock, ChevronRight, Radio, Cpu,
} from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import {
  api, DashboardStats, StoredTransaction, HealthStatus, LiveAlert,
  parseReasons, useAlertStream,
} from "@/lib/api";
import { useCurrency, codeFromLocation } from "./CurrencyContext";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmt(n: number | undefined, dec = 0) {
  if (n == null) return "—";
  return n.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function fmtTime(iso: string) {
  try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
  catch { return iso; }
}

function DecisionBadge({ decision }: { decision: "APPROVE" | "FLAG" | "BLOCK" }) {
  const styles = {
    APPROVE: "bg-green-900 text-green-300",
    FLAG:    "bg-yellow-900 text-yellow-300",
    BLOCK:   "bg-red-900 text-red-300",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${styles[decision]}`}>
      {decision}
    </span>
  );
}

function RiskBar({ score }: { score: number }) {
  const color = score >= 70 ? "bg-red-500" : score >= 40 ? "bg-yellow-500" : "bg-green-500";
  const text  = score >= 70 ? "text-red-400" : score >= 40 ? "text-yellow-400" : "text-green-400";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-700 rounded-full h-1.5 w-16">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className={`text-xs font-mono font-bold ${text}`}>{score}</span>
    </div>
  );
}

function StatCard({
  label, value, sub, icon, color = "text-white",
}: {
  label: string; value: string; sub?: string;
  icon: React.ReactNode; color?: string;
}) {
  return (
    <div className="bg-[#111827] border border-gray-800 rounded-xl p-4">
      <div className="flex items-start justify-between mb-2">
        <span className="text-gray-400 text-xs uppercase tracking-wider">{label}</span>
        <span className="text-gray-600">{icon}</span>
      </div>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-gray-500 text-xs mt-1">{sub}</p>}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { currency, display, convert } = useCurrency();

  const [health, setHealth]             = useState<HealthStatus | null>(null);
  const [stats, setStats]               = useState<DashboardStats | null>(null);
  const [transactions, setTransactions] = useState<StoredTransaction[]>([]);
  const [liveAlerts, setLiveAlerts]     = useState<LiveAlert[]>([]);
  const [activeTab, setActiveTab]       = useState<"live" | "recent">("live");
  const [loading, setLoading]           = useState(true);
  const [lastRefresh, setLastRefresh]   = useState<Date>(new Date());

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [h, s, t] = await Promise.all([
        api.getHealth(),
        api.getDashboardStats(),
        api.getDashboardTransactions(20),
      ]);
      setHealth(h);
      setStats(s);
      setTransactions(t.transactions);
      setLastRefresh(new Date());
    } catch (e) {
      console.error("Dashboard refresh failed", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 15_000);
    return () => clearInterval(interval);
  }, [refresh]);

  useAlertStream((alert) => {
    setLiveAlerts(prev => [alert, ...prev].slice(0, 20));
    api.getDashboardStats().then(setStats).catch(() => null);
  });

  // Compute aggregate amounts converted from individual transactions
  const convertedBlocked = transactions
    .filter(tx => tx.decision === "BLOCK")
    .reduce((sum, tx) => sum + convert(tx.amount, codeFromLocation(tx.location)), 0);

  const convertedApproved = transactions
    .filter(tx => tx.decision === "APPROVE")
    .reduce((sum, tx) => sum + convert(tx.amount, codeFromLocation(tx.location)), 0);

  const chartData = stats ? [
    { name: "Approved", value: stats.approved_count, color: "#22c55e" },
    { name: "Flagged",  value: stats.flagged_count,  color: "#eab308" },
    { name: "Blocked",  value: stats.blocked_count,  color: "#ef4444" },
  ] : [];

  const modelLoaded = health?.model_loaded ?? false;

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-white">
      <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">

        {/* Header */}
        <div className="flex items-start justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <Shield size={24} className="text-blue-400" />
              Fraud Intelligence Platform
            </h1>
            <p className="text-gray-400 text-sm mt-1">
              Real-time risk monitoring across ASEAN — V HACK 2026
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border font-medium ${
              modelLoaded
                ? "bg-green-900/40 border-green-600 text-green-300"
                : "bg-red-900/40 border-red-600 text-red-300"
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${modelLoaded ? "bg-green-400" : "bg-red-400"}`} />
              {modelLoaded ? "Model Active" : "Model Offline"}
            </span>
            <Link
              href="/wallet"
              className="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1.5 rounded-full font-medium transition"
            >
              <Wallet size={12} />
              Open eWallet
            </Link>
            <button
              onClick={refresh}
              className="p-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 transition"
              title="Refresh"
            >
              <RefreshCw size={14} className={`text-gray-400 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* Model not loaded warning */}
        {!modelLoaded && (
          <div className="bg-yellow-900/30 border border-yellow-700 rounded-xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-yellow-300 font-semibold">
              <AlertTriangle size={18} />
              Fraud Detection Model Not Loaded
            </div>
            <p className="text-yellow-200/80 text-sm">
              Train the model to activate real fraud scoring. Dashboard will populate as wallet transactions are submitted.
            </p>
            <div className="bg-black/40 rounded-lg p-3 font-mono text-xs text-green-300 space-y-1">
              <p># Step 1: Download IEEE-CIS dataset from Kaggle</p>
              <p className="text-gray-400"># https://www.kaggle.com/c/ieee-fraud-detection/data</p>
              <p className="mt-1"># Step 2: Train the model</p>
              <p>cd backend</p>
              <p>python training/train_engine.py --data-dir ./data/ieee-cis</p>
              <p className="mt-1"># Step 3: Restart the backend</p>
              <p>python main.py</p>
            </div>
          </div>
        )}

        {/* Stats grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Total Transactions"
            value={stats ? fmt(stats.total) : "—"}
            sub={stats ? `Refreshed ${lastRefresh.toLocaleTimeString()}` : "Waiting..."}
            icon={<Activity size={16} />}
          />
          <StatCard
            label="Blocked"
            value={stats ? fmt(stats.blocked_count) : "—"}
            sub={stats && stats.total > 0 ? `${(stats.blocked_count / stats.total * 100).toFixed(1)}% of total` : undefined}
            icon={<XCircle size={16} />}
            color="text-red-400"
          />
          <StatCard
            label="Flagged for Review"
            value={stats ? fmt(stats.flagged_count) : "—"}
            sub="Pending analyst review"
            icon={<AlertTriangle size={16} />}
            color="text-yellow-400"
          />
          <StatCard
            label="Fraud Rate"
            value={stats && stats.total > 0 ? `${(stats.fraud_rate * 100).toFixed(1)}%` : "—"}
            sub={stats ? `Avg risk: ${fmt(stats.avg_risk_score, 1)}` : undefined}
            icon={<TrendingUp size={16} />}
            color={stats && stats.fraud_rate > 0.1 ? "text-red-400" : "text-white"}
          />
        </div>

        {/* Chart + amount protected */}
        {stats && stats.total > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 bg-[#111827] border border-gray-800 rounded-xl p-5">
              <p className="text-sm font-semibold text-gray-300 mb-4">Decision Distribution</p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData} barSize={48}>
                  <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#9ca3af", fontSize: 12 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: "#1f2937", border: "1px solid #374151", borderRadius: 8 }}
                    labelStyle={{ color: "#e5e7eb" }}
                    itemStyle={{ color: "#9ca3af" }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={index} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-4">
              <div className="bg-[#111827] border border-gray-800 rounded-xl p-4">
                <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Blocked Value</p>
                <p className="text-xl font-bold text-green-400">
                  {currency.symbol}{convertedBlocked > 0
                    ? Math.round(convertedBlocked).toLocaleString()
                    : fmt(stats.total_blocked_amount)}
                </p>
                <p className="text-gray-500 text-xs mt-1">In {currency.code} · recent {transactions.filter(t => t.decision === "BLOCK").length} txns</p>
              </div>
              <div className="bg-[#111827] border border-gray-800 rounded-xl p-4">
                <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">Approved Volume</p>
                <p className="text-xl font-bold text-blue-400">
                  {currency.symbol}{convertedApproved > 0
                    ? Math.round(convertedApproved).toLocaleString()
                    : fmt(stats.total_amount - stats.total_blocked_amount)}
                </p>
                <p className="text-gray-500 text-xs mt-1">In {currency.code} · recent {transactions.filter(t => t.decision === "APPROVE").length} txns</p>
              </div>
              {health?.artifact_version && (
                <div className="bg-[#111827] border border-gray-800 rounded-xl p-4">
                  <p className="text-gray-400 text-xs uppercase tracking-wider mb-1 flex items-center gap-1"><Cpu size={10} /> Model Version</p>
                  <p className="text-xs font-mono text-gray-300 truncate">{health.artifact_version.slice(0, 19)}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="bg-[#111827] border border-gray-800 rounded-xl overflow-hidden">
          <div className="flex border-b border-gray-800">
            {(["live", "recent"] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition ${
                  activeTab === tab
                    ? "bg-gray-800 text-white border-b-2 border-blue-500"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                {tab === "live" ? <Radio size={14} className="text-green-400" /> : <Clock size={14} />}
                {tab === "live" ? "Live Feed" : "Recent Transactions"}
                {tab === "live" && liveAlerts.length > 0 && (
                  <span className="bg-blue-600 text-white text-xs px-1.5 py-0.5 rounded-full">
                    {liveAlerts.length}
                  </span>
                )}
              </button>
            ))}
          </div>

          <div className="p-4">
            {activeTab === "live" && (
              liveAlerts.length === 0 ? (
                <div className="text-center py-12 text-gray-500 space-y-2">
                  <Radio size={32} className="mx-auto text-gray-600" />
                  <p className="font-medium">Waiting for transactions from the eWallet...</p>
                  <p className="text-sm">
                    <Link href="/wallet" className="text-blue-400 hover:underline">Open the eWallet</Link> and submit transactions to see live scoring here.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {liveAlerts.map((a, i) => (
                    <div key={i} className="flex items-center gap-3 p-3 bg-gray-900/60 rounded-xl border border-gray-800 hover:bg-gray-800/60 transition">
                      <DecisionBadge decision={a.decision} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-mono text-gray-300 truncate">{a.user_id}</span>
                          <span className="text-gray-600">·</span>
                          <span className="text-sm font-semibold text-white">
                            {currency.symbol}{a.amount.toLocaleString()}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 truncate">{a.reasons[0]}</p>
                      </div>
                      <RiskBar score={a.risk_score} />
                      <span className="text-xs text-gray-500 whitespace-nowrap">{fmtTime(a.timestamp)}</span>
                      <Link href={`/case/${a.transaction_id}`} className="text-gray-600 hover:text-blue-400 transition">
                        <ChevronRight size={16} />
                      </Link>
                    </div>
                  ))}
                </div>
              )
            )}

            {activeTab === "recent" && (
              transactions.length === 0 ? (
                <div className="text-center py-12 text-gray-500 space-y-2">
                  <Activity size={32} className="mx-auto text-gray-600" />
                  <p className="font-medium">No transactions yet</p>
                  <p className="text-sm">
                    <Link href="/wallet" className="text-blue-400 hover:underline">Use the eWallet</Link> to generate real fraud scores.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-gray-500 text-xs border-b border-gray-800">
                        <th className="text-left py-2 pr-4">Time</th>
                        <th className="text-left py-2 pr-4">User</th>
                        <th className="text-right py-2 pr-4">Amount ({currency.code})</th>
                        <th className="text-left py-2 pr-4">Type</th>
                        <th className="text-left py-2 pr-4">Location</th>
                        <th className="text-left py-2 pr-4">Risk</th>
                        <th className="text-left py-2 pr-4">Decision</th>
                        <th className="text-left py-2">Signal</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/50">
                      {transactions.map(tx => {
                        const reasons = parseReasons(tx.reasons);
                        const fromCode = codeFromLocation(tx.location);
                        return (
                          <tr key={tx.transaction_id} className="hover:bg-gray-800/30 transition">
                            <td className="py-2.5 pr-4 text-gray-500 text-xs whitespace-nowrap">{fmtTime(tx.timestamp)}</td>
                            <td className="py-2.5 pr-4 font-mono text-gray-300 text-xs truncate max-w-[100px]">{tx.user_id}</td>
                            <td className="py-2.5 pr-4 text-right font-semibold text-white">
                              {display(tx.amount, fromCode)}
                            </td>
                            <td className="py-2.5 pr-4 text-gray-400 capitalize text-xs">{tx.transaction_type}</td>
                            <td className="py-2.5 pr-4 text-gray-500 text-xs whitespace-nowrap">{tx.location || "—"}</td>
                            <td className="py-2.5 pr-4"><RiskBar score={tx.risk_score} /></td>
                            <td className="py-2.5 pr-4"><DecisionBadge decision={tx.decision} /></td>
                            <td className="py-2.5 pr-4 text-xs text-gray-500 max-w-[180px] truncate">{reasons[0] ?? "—"}</td>
                            <td className="py-2.5">
                              <Link href={`/case/${tx.transaction_id}`} className="text-gray-600 hover:text-blue-400 transition">
                                <ChevronRight size={14} />
                              </Link>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )
            )}
          </div>
        </div>

        <p className="text-center text-gray-600 text-xs">
          FraudShield — V HACK 2026 · Amounts converted at indicative rates · {currency.flag} Displaying in {currency.code}
        </p>
      </div>
    </div>
  );
}
