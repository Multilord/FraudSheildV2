"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  Shield, AlertTriangle, XCircle, Activity, RefreshCw,
  Wallet, TrendingUp, Clock, ChevronRight, Radio, Cpu,
  CheckCircle,
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
  const cls = {
    APPROVE: "badge-approve",
    FLAG:    "badge-flag",
    BLOCK:   "badge-block",
  }[decision];
  return (
    <span className={`text-[11px] px-2.5 py-0.5 rounded-full font-semibold tracking-wide ${cls}`}>
      {decision}
    </span>
  );
}

function RiskBar({ score }: { score: number }) {
  const color = score >= 70 ? "bg-[#FF453A]" : score >= 40 ? "bg-[#FF9F0A]" : "bg-[#30D158]";
  const text  = score >= 70 ? "text-[#FF453A]" : score >= 40 ? "text-[#FF9F0A]" : "text-[#30D158]";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-white/[0.08] rounded-full h-1 w-16">
        <div className={`h-1 rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className={`text-xs font-mono font-bold tabular-nums ${text}`}>{score}</span>
    </div>
  );
}

function StatCard({
  label, value, sub, icon, accent = "text-white",
}: {
  label: string; value: string; sub?: string;
  icon: React.ReactNode; accent?: string;
}) {
  return (
    <div className="card p-5 animate-fade-up">
      <div className="flex items-start justify-between mb-3">
        <span className="text-[11px] text-white/40 uppercase tracking-widest font-medium">{label}</span>
        <span className="text-white/20">{icon}</span>
      </div>
      <p className={`text-[28px] font-bold leading-none tracking-tight ${accent}`}>{value}</p>
      {sub && <p className="text-white/35 text-xs mt-2">{sub}</p>}
    </div>
  );
}

// ─── Custom Tooltip ──────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{value: number}>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl px-3 py-2 text-sm shadow-xl" style={{ background: "rgba(28,28,30,0.95)", border: "1px solid rgba(255,255,255,0.10)" }}>
      <p className="text-white/50 text-xs mb-0.5">{label}</p>
      <p className="text-white font-semibold">{payload[0].value.toLocaleString()}</p>
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

  const convertedBlocked = transactions
    .filter(tx => tx.decision === "BLOCK")
    .reduce((sum, tx) => sum + convert(tx.amount, codeFromLocation(tx.location)), 0);

  const convertedApproved = transactions
    .filter(tx => tx.decision === "APPROVE")
    .reduce((sum, tx) => sum + convert(tx.amount, codeFromLocation(tx.location)), 0);

  const chartData = stats ? [
    { name: "Approved", value: stats.approved_count, color: "#30D158" },
    { name: "Flagged",  value: stats.flagged_count,  color: "#FF9F0A" },
    { name: "Blocked",  value: stats.blocked_count,  color: "#FF453A" },
  ] : [];

  const modelLoaded = health?.model_loaded ?? false;

  return (
    <div className="min-h-screen bg-[#09090b]">
      <div className="max-w-6xl mx-auto px-5 py-8 space-y-8">

        {/* ── Header ── */}
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <Shield size={22} className="text-[#0A84FF]" />
              Fraud Intelligence Platform
            </h1>
            <p className="text-white/40 text-sm mt-1.5">
              Real-time risk monitoring across ASEAN · V HACK 2026
            </p>
          </div>
          <div className="flex items-center gap-2.5">
            {/* Model status pill */}
            <span className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded-full font-medium border ${
              modelLoaded
                ? "bg-[#30D158]/10 border-[#30D158]/20 text-[#30D158]"
                : "bg-[#FF453A]/10 border-[#FF453A]/20 text-[#FF453A]"
            }`}>
              {modelLoaded ? (
                <span className="live-dot" style={{ background: "#30D158" }} />
              ) : (
                <span className="w-1.5 h-1.5 rounded-full bg-[#FF453A]" />
              )}
              {modelLoaded ? "Model Active" : "Model Offline"}
            </span>
            <Link
              href="/wallet"
              className="flex items-center gap-1.5 bg-[#0A84FF] hover:bg-[#0A84FF]/85 text-white text-xs px-4 py-1.5 rounded-full font-semibold transition-all shadow-lg shadow-[#0A84FF]/25"
            >
              <Wallet size={12} />
              Open eWallet
            </Link>
            <button
              onClick={refresh}
              className="p-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.09] border border-white/[0.07] transition-all"
              title="Refresh"
            >
              <RefreshCw size={14} className={`text-white/40 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* ── Model offline banner ── */}
        {!modelLoaded && (
          <div className="card p-5 border-[#FF9F0A]/20 bg-[#FF9F0A]/[0.05]">
            <div className="flex items-center gap-2 text-[#FF9F0A] font-semibold mb-2">
              <AlertTriangle size={16} />
              Fraud Detection Model Not Loaded
            </div>
            <p className="text-white/50 text-sm mb-4">
              Train the model to activate real fraud scoring. The dashboard will populate as wallet transactions are submitted.
            </p>
            <div className="rounded-xl p-4 font-mono text-xs space-y-1 text-[#30D158]"
              style={{ background: "rgba(0,0,0,0.5)", border: "1px solid rgba(255,255,255,0.06)" }}
            >
              <p className="text-white/30"># Download IEEE-CIS dataset from Kaggle, then:</p>
              <p>cd backend</p>
              <p>python training/train_engine.py --data-dir ./data/ieee-cis</p>
              <p className="text-white/30"># Restart the backend</p>
              <p>python main.py</p>
            </div>
          </div>
        )}

        {/* ── Stats grid ── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 stagger">
          <StatCard
            label="Total Transactions"
            value={stats ? fmt(stats.total) : "—"}
            sub={stats ? `Refreshed ${lastRefresh.toLocaleTimeString()}` : "Waiting…"}
            icon={<Activity size={15} />}
          />
          <StatCard
            label="Blocked"
            value={stats ? fmt(stats.blocked_count) : "—"}
            sub={stats && stats.total > 0 ? `${(stats.blocked_count / stats.total * 100).toFixed(1)}% of total` : undefined}
            icon={<XCircle size={15} />}
            accent="text-[#FF453A]"
          />
          <StatCard
            label="Flagged"
            value={stats ? fmt(stats.flagged_count) : "—"}
            sub="Pending analyst review"
            icon={<AlertTriangle size={15} />}
            accent="text-[#FF9F0A]"
          />
          <StatCard
            label="Fraud Rate"
            value={stats && stats.total > 0 ? `${(stats.fraud_rate * 100).toFixed(1)}%` : "—"}
            sub={stats ? `Avg risk: ${fmt(stats.avg_risk_score, 1)}` : undefined}
            icon={<TrendingUp size={15} />}
            accent={stats && stats.fraud_rate > 0.1 ? "text-[#FF453A]" : "text-white"}
          />
        </div>

        {/* ── Chart + amounts ── */}
        {stats && stats.total > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            {/* Bar chart */}
            <div className="lg:col-span-2 card p-5">
              <p className="text-xs text-white/40 uppercase tracking-widest font-medium mb-5">Decision Distribution</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={chartData} barSize={44} barGap={8}>
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 12 }}
                    axisLine={false} tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 12 }}
                    axisLine={false} tickLine={false}
                  />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)", radius: 8 }} />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={index} fill={entry.color} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Side cards */}
            <div className="space-y-3">
              <div className="card p-4">
                <p className="text-[11px] text-white/40 uppercase tracking-widest font-medium mb-2">Blocked Value</p>
                <p className="text-xl font-bold text-[#30D158]">
                  {currency.symbol}{convertedBlocked > 0
                    ? Math.round(convertedBlocked).toLocaleString()
                    : fmt(stats.total_blocked_amount)}
                </p>
                <p className="text-white/30 text-xs mt-1.5">
                  {currency.code} · {transactions.filter(t => t.decision === "BLOCK").length} blocked txns
                </p>
              </div>
              <div className="card p-4">
                <p className="text-[11px] text-white/40 uppercase tracking-widest font-medium mb-2">Approved Volume</p>
                <p className="text-xl font-bold text-[#0A84FF]">
                  {currency.symbol}{convertedApproved > 0
                    ? Math.round(convertedApproved).toLocaleString()
                    : fmt(stats.total_amount - stats.total_blocked_amount)}
                </p>
                <p className="text-white/30 text-xs mt-1.5">
                  {currency.code} · {transactions.filter(t => t.decision === "APPROVE").length} approved txns
                </p>
              </div>
              {health?.artifact_version && (
                <div className="card p-4">
                  <p className="text-[11px] text-white/40 uppercase tracking-widest font-medium mb-2 flex items-center gap-1.5">
                    <Cpu size={10} /> Model Version
                  </p>
                  <p className="text-xs font-mono text-white/50 truncate">{health.artifact_version.slice(0, 20)}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Tabs ── */}
        <div className="card overflow-hidden">
          {/* Tab bar */}
          <div className="flex border-b border-white/[0.06] px-2 pt-2 gap-1">
            {(["live", "recent"] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-all ${
                  activeTab === tab
                    ? "bg-white/[0.08] text-white border-b-2 border-[#0A84FF]"
                    : "text-white/40 hover:text-white/70 hover:bg-white/[0.04]"
                }`}
              >
                {tab === "live" ? (
                  <>
                    <span className="live-dot" style={{ width: 6, height: 6 }} />
                    Live Feed
                  </>
                ) : (
                  <><Clock size={13} /> Recent</>
                )}
                {tab === "live" && liveAlerts.length > 0 && (
                  <span className="bg-[#0A84FF] text-white text-[10px] px-1.5 py-0.5 rounded-full font-bold min-w-[18px] text-center">
                    {liveAlerts.length}
                  </span>
                )}
              </button>
            ))}
          </div>

          <div className="p-4">
            {/* Live feed */}
            {activeTab === "live" && (
              liveAlerts.length === 0 ? (
                <div className="text-center py-14 space-y-3">
                  <div className="w-12 h-12 rounded-2xl bg-white/[0.04] flex items-center justify-center mx-auto">
                    <Radio size={22} className="text-white/20" />
                  </div>
                  <p className="text-white/40 font-medium text-sm">Waiting for live transactions…</p>
                  <p className="text-white/25 text-xs">
                    <Link href="/wallet" className="text-[#0A84FF] hover:underline">Open the eWallet</Link> and submit a transaction to see it here.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {liveAlerts.map((a, i) => (
                    <div key={i} className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.05] hover:border-white/[0.10] transition-all">
                      <DecisionBadge decision={a.decision} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-mono text-white/60 truncate">{a.user_id}</span>
                          <span className="text-white/20 text-xs">·</span>
                          <span className="text-sm font-semibold text-white">
                            {currency.symbol}{a.amount.toLocaleString()}
                          </span>
                        </div>
                        <p className="text-xs text-white/30 truncate mt-0.5">{a.reasons[0]}</p>
                      </div>
                      <RiskBar score={a.risk_score} />
                      <span className="text-xs text-white/25 whitespace-nowrap tabular-nums">{fmtTime(a.timestamp)}</span>
                      <Link href={`/case/${a.transaction_id}`} className="text-white/20 hover:text-[#0A84FF] transition-colors">
                        <ChevronRight size={15} />
                      </Link>
                    </div>
                  ))}
                </div>
              )
            )}

            {/* Recent table */}
            {activeTab === "recent" && (
              transactions.length === 0 ? (
                <div className="text-center py-14 space-y-3">
                  <div className="w-12 h-12 rounded-2xl bg-white/[0.04] flex items-center justify-center mx-auto">
                    <Activity size={22} className="text-white/20" />
                  </div>
                  <p className="text-white/40 font-medium text-sm">No transactions yet</p>
                  <p className="text-white/25 text-xs">
                    <Link href="/wallet" className="text-[#0A84FF] hover:underline">Use the eWallet</Link> to generate fraud scores.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-[11px] text-white/30 uppercase tracking-widest border-b border-white/[0.06]">
                        <th className="text-left py-2.5 pr-4 font-medium">Time</th>
                        <th className="text-left py-2.5 pr-4 font-medium">User</th>
                        <th className="text-right py-2.5 pr-4 font-medium">Amount</th>
                        <th className="text-left py-2.5 pr-4 font-medium">Type</th>
                        <th className="text-left py-2.5 pr-4 font-medium">Location</th>
                        <th className="text-left py-2.5 pr-4 font-medium">Risk</th>
                        <th className="text-left py-2.5 pr-4 font-medium">Decision</th>
                        <th className="text-left py-2.5 pr-4 font-medium">Signal</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {transactions.map(tx => {
                        const reasons = parseReasons(tx.reasons);
                        const fromCode = codeFromLocation(tx.location);
                        return (
                          <tr key={tx.transaction_id} className="hover:bg-white/[0.03] transition-colors group">
                            <td className="py-3 pr-4 text-white/30 text-xs whitespace-nowrap tabular-nums">{fmtTime(tx.timestamp)}</td>
                            <td className="py-3 pr-4 font-mono text-white/50 text-xs truncate max-w-[100px]">{tx.user_id}</td>
                            <td className="py-3 pr-4 text-right font-semibold text-white tabular-nums">{display(tx.amount, fromCode)}</td>
                            <td className="py-3 pr-4 text-white/40 capitalize text-xs">{tx.transaction_type}</td>
                            <td className="py-3 pr-4 text-white/30 text-xs whitespace-nowrap">{tx.location || "—"}</td>
                            <td className="py-3 pr-4"><RiskBar score={tx.risk_score} /></td>
                            <td className="py-3 pr-4"><DecisionBadge decision={tx.decision} /></td>
                            <td className="py-3 pr-4 text-xs text-white/30 max-w-[160px] truncate">{reasons[0] ?? "—"}</td>
                            <td className="py-3">
                              <Link href={`/case/${tx.transaction_id}`} className="text-white/20 hover:text-[#0A84FF] transition-colors opacity-0 group-hover:opacity-100">
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

        <p className="text-center text-white/20 text-xs pb-2">
          FraudShield · V HACK 2026 · {currency.flag} Displaying in {currency.code} at indicative rates
        </p>
      </div>
    </div>
  );
}
