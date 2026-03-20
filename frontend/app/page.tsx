"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  Shield, AlertTriangle, XCircle, Activity, RefreshCw,
  Wallet, TrendingUp, ChevronRight, Radio, Cpu,
  CheckCircle, ArrowUpRight, Lock,
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

// ─── Decision Badge ───────────────────────────────────────────────────────────
function DecisionBadge({ decision }: { decision: "APPROVE" | "FLAG" | "BLOCK" }) {
  const cls = { APPROVE: "badge-approve", FLAG: "badge-flag", BLOCK: "badge-block" }[decision];
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold tracking-wide ${cls}`}>
      {decision}
    </span>
  );
}

// ─── Risk Chip ────────────────────────────────────────────────────────────────
function RiskChip({ score }: { score: number }) {
  const color = score >= 70 ? "#FF453A" : score >= 40 ? "#FF9F0A" : "#30D158";
  return (
    <div className="flex items-center gap-2 min-w-[72px]">
      <div className="flex-1 h-1 rounded-full bg-white/[0.08] overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${score}%`, background: color }} />
      </div>
      <span className="text-[11px] font-mono font-bold tabular-nums" style={{ color }}>{score}</span>
    </div>
  );
}

// ─── Stat Card ────────────────────────────────────────────────────────────────
function StatCard({
  label, value, sub, icon, accent = "text-white", trend,
}: {
  label: string; value: string; sub?: string;
  icon: React.ReactNode; accent?: string; trend?: "up" | "down" | null;
}) {
  return (
    <div className="card p-5 animate-fade-up">
      <div className="flex items-start justify-between mb-3">
        <p className="section-label">{label}</p>
        <span className="text-white/18">{icon}</span>
      </div>
      <p className={`text-[30px] font-bold leading-none tracking-tight ${accent}`}>{value}</p>
      {sub && (
        <div className="flex items-center gap-1.5 mt-2">
          {trend === "up" && <ArrowUpRight size={11} className="text-[#FF453A]" />}
          {trend === "down" && <ArrowUpRight size={11} className="text-[#30D158] rotate-90" />}
          <p className="text-white/30 text-xs">{sub}</p>
        </div>
      )}
    </div>
  );
}

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }: {
  active?: boolean; payload?: Array<{ value: number }>; label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl px-3 py-2 text-sm shadow-2xl"
      style={{ background: "rgba(24,24,27,0.96)", border: "1px solid rgba(255,255,255,0.09)" }}>
      <p className="text-white/40 text-xs mb-0.5">{label}</p>
      <p className="text-white font-semibold tabular-nums">{payload[0].value.toLocaleString()}</p>
    </div>
  );
}

// ─── Live Alert Row ───────────────────────────────────────────────────────────
function LiveAlertRow({ alert, currency }: { alert: LiveAlert; currency: { symbol: string } }) {
  const isBlock  = alert.decision === "BLOCK";
  const isFlag   = alert.decision === "FLAG";
  const rowColor = isBlock ? "border-[#FF453A]/20 bg-[#FF453A]/[0.04]"
    : isFlag ? "border-[#FF9F0A]/15 bg-[#FF9F0A]/[0.03]"
    : "border-white/[0.05] bg-white/[0.02]";

  return (
    <div className={`flex items-center gap-3 p-3 rounded-xl border transition-all ${rowColor}`}>
      <DecisionBadge decision={alert.decision} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono text-white/55 truncate">{alert.user_id}</span>
          <span className="text-white/20 text-xs">·</span>
          <span className="text-sm font-semibold text-white tabular-nums">
            {currency.symbol}{alert.amount.toLocaleString()}
          </span>
        </div>
        <p className="text-xs text-white/30 truncate mt-0.5">{alert.reasons[0] ?? "—"}</p>
      </div>
      <RiskChip score={alert.risk_score} />
      <span className="text-[11px] text-white/20 whitespace-nowrap tabular-nums">{fmtTime(alert.timestamp)}</span>
      <Link href={`/case/${alert.transaction_id}`} className="text-white/20 hover:text-[#0A84FF] transition-colors">
        <ChevronRight size={14} />
      </Link>
    </div>
  );
}

// ─── Recent TX Row ────────────────────────────────────────────────────────────
function RecentTxRow({ tx, display, currency }: {
  tx: StoredTransaction;
  display: (amount: number, fromCode: string) => string;
  currency: { code: string };
}) {
  const reasons = parseReasons(tx.reasons);
  const fromCode = codeFromLocation(tx.location);
  return (
    <tr className="hover:bg-white/[0.025] transition-colors group">
      <td className="py-3 pr-4 text-[11px] text-white/25 whitespace-nowrap tabular-nums">{fmtTime(tx.timestamp)}</td>
      <td className="py-3 pr-4 font-mono text-xs text-white/45 truncate max-w-[90px]">{tx.user_id}</td>
      <td className="py-3 pr-4 text-right text-sm font-semibold text-white tabular-nums">{display(tx.amount, fromCode)}</td>
      <td className="py-3 pr-4 text-xs text-white/35 capitalize">{tx.transaction_type}</td>
      <td className="py-3 pr-4 text-xs text-white/25">{tx.location || "—"}</td>
      <td className="py-3 pr-4"><RiskChip score={tx.risk_score} /></td>
      <td className="py-3 pr-4"><DecisionBadge decision={tx.decision} /></td>
      <td className="py-3 pr-4 text-xs text-white/25 max-w-[140px] truncate">{reasons[0] ?? "—"}</td>
      <td className="py-3">
        <Link href={`/case/${tx.transaction_id}`}
          className="text-white/15 hover:text-[#0A84FF] transition-colors opacity-0 group-hover:opacity-100">
          <ChevronRight size={13} />
        </Link>
      </td>
    </tr>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
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
        api.getHealth(), api.getDashboardStats(), api.getDashboardTransactions(20),
      ]);
      setHealth(h); setStats(s); setTransactions(t.transactions);
      setLastRefresh(new Date());
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    refresh();
    const i = setInterval(refresh, 15_000);
    return () => clearInterval(i);
  }, [refresh]);

  useAlertStream(useCallback((alert: LiveAlert) => {
    setLiveAlerts(prev => [alert, ...prev].slice(0, 25));
    Promise.all([
      api.getDashboardStats(),
      api.getDashboardTransactions(20),
    ]).then(([s, t]) => {
      setStats(s);
      setTransactions(t.transactions);
      setLastRefresh(new Date());
    }).catch(() => null);
  }, []));

  const convertedBlocked = transactions
    .filter(t => t.decision === "BLOCK")
    .reduce((s, t) => s + convert(t.amount, codeFromLocation(t.location)), 0);
  const convertedApproved = transactions
    .filter(t => t.decision === "APPROVE")
    .reduce((s, t) => s + convert(t.amount, codeFromLocation(t.location)), 0);

  const chartData = stats ? [
    { name: "Approved", value: stats.approved_count, color: "#30D158" },
    { name: "Flagged",  value: stats.flagged_count,  color: "#FF9F0A" },
    { name: "Blocked",  value: stats.blocked_count,  color: "#FF453A" },
  ] : [];

  const modelLoaded  = health?.model_loaded ?? false;
  const fraudRatePct = stats && stats.total > 0 ? (stats.fraud_rate * 100).toFixed(1) : null;
  const threatLevel  = fraudRatePct === null ? null
    : parseFloat(fraudRatePct) > 15 ? "High"
    : parseFloat(fraudRatePct) > 5  ? "Elevated"
    : "Normal";

  return (
    <div className="min-h-screen bg-[#09090b]">
      <div className="max-w-6xl mx-auto px-5 py-8 space-y-7">

        {/* ── Header ── */}
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <Shield size={22} style={{ color: "#0A84FF" }} />
              Fraud Intelligence Platform
            </h1>
            <p className="text-white/35 text-sm mt-1.5">
              Real-time risk monitoring across ASEAN · V HACK 2026
            </p>
          </div>
          <div className="flex items-center gap-2.5 flex-wrap">
            {/* Model status */}
            <span className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded-full font-medium ${
              modelLoaded
                ? "bg-[#30D158]/10 border border-[#30D158]/20 text-[#30D158]"
                : "bg-[#FF453A]/10 border border-[#FF453A]/20 text-[#FF453A]"
            }`}>
              {modelLoaded
                ? <span className="live-dot" style={{ width: 6, height: 6, background: "#30D158" }} />
                : <span className="w-1.5 h-1.5 rounded-full bg-[#FF453A]" />}
              {modelLoaded ? "Model Active" : "Model Offline"}
            </span>

            {/* Threat level */}
            {threatLevel && (
              <span className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded-full font-medium border ${
                threatLevel === "High"     ? "bg-[#FF453A]/10 border-[#FF453A]/20 text-[#FF453A]"
                : threatLevel === "Elevated" ? "bg-[#FF9F0A]/10 border-[#FF9F0A]/20 text-[#FF9F0A]"
                :                             "bg-[#30D158]/10 border-[#30D158]/20 text-[#30D158]"
              }`}>
                <Lock size={10} />Threat: {threatLevel}
              </span>
            )}

            <Link href="/wallet"
              className="flex items-center gap-1.5 bg-[#0A84FF] hover:bg-[#0A84FF]/85 text-white text-xs px-4 py-1.5 rounded-full font-semibold transition-all shadow-lg shadow-[#0A84FF]/20">
              <Wallet size={12} />Open eWallet
            </Link>
            <button onClick={refresh}
              className="p-2 rounded-xl card hover:bg-white/[0.07] transition-all"
              title="Refresh">
              <RefreshCw size={14} className={`text-white/35 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* ── Model Offline Banner ── */}
        {!modelLoaded && (
          <div className="card p-5" style={{ borderColor: "rgba(255,159,10,0.25)", background: "rgba(255,159,10,0.05)" }}>
            <div className="flex items-center gap-2 text-[#FF9F0A] font-semibold mb-2.5">
              <AlertTriangle size={16} />Fraud Detection Model Not Loaded
            </div>
            <p className="text-white/45 text-sm mb-4">
              Train the model to activate real fraud scoring. The dashboard populates as wallet transactions are submitted.
            </p>
            <div className="rounded-xl p-4 text-xs font-mono space-y-1 text-[#30D158]"
              style={{ background: "rgba(0,0,0,0.5)", border: "1px solid rgba(255,255,255,0.07)" }}>
              <p className="text-white/25"># Download IEEE-CIS dataset from Kaggle, then:</p>
              <p>cd backend</p>
              <p>python training/train_engine.py --data-dir ./data/ieee-cis</p>
              <p className="text-white/25"># Restart backend</p>
              <p>python main.py</p>
            </div>
          </div>
        )}

        {/* ── Stats Grid ── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 stagger">
          <StatCard
            label="Total Transactions"
            value={stats ? fmt(stats.total) : "—"}
            sub={`Refreshed ${lastRefresh.toLocaleTimeString()}`}
            icon={<Activity size={15} />}
          />
          <StatCard
            label="Blocked"
            value={stats ? fmt(stats.blocked_count) : "—"}
            sub={stats && stats.total > 0
              ? `${(stats.blocked_count / stats.total * 100).toFixed(1)}% of total`
              : undefined}
            icon={<XCircle size={15} />}
            accent="text-[#FF453A]"
            trend={stats && stats.blocked_count > 0 ? "up" : null}
          />
          <StatCard
            label="Flagged for Review"
            value={stats ? fmt(stats.flagged_count) : "—"}
            sub="Pending analyst review"
            icon={<AlertTriangle size={15} />}
            accent="text-[#FF9F0A]"
          />
          <StatCard
            label="Fraud Rate"
            value={stats && stats.total > 0
              ? `${(stats.fraud_rate * 100).toFixed(1)}%`
              : "—"}
            sub={stats ? `Avg risk score: ${fmt(stats.avg_risk_score, 1)}` : undefined}
            icon={<TrendingUp size={15} />}
            accent={stats && stats.fraud_rate > 0.1 ? "text-[#FF453A]" : "text-white"}
          />
        </div>

        {/* ── Chart + Side Cards ── */}
        {stats && stats.total > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Bar chart */}
            <div className="lg:col-span-2 card p-5">
              <div className="flex items-center justify-between mb-5">
                <p className="section-label">Decision Distribution</p>
                <p className="text-[11px] text-white/25 tabular-nums">
                  {stats.total.toLocaleString()} total
                </p>
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={chartData} barSize={48} barGap={8}>
                  <XAxis dataKey="name" tick={{ fill: "rgba(255,255,255,0.30)", fontSize: 12 }}
                    axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "rgba(255,255,255,0.30)", fontSize: 11 }}
                    axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />}
                    cursor={{ fill: "rgba(255,255,255,0.03)", radius: 8 }} />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                    {chartData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} fillOpacity={0.80} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              {/* Mini legend */}
              <div className="flex gap-5 mt-3 pt-3 border-t border-white/[0.05]">
                {chartData.map(d => (
                  <div key={d.name} className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                    <span className="text-xs text-white/35">{d.name}</span>
                    <span className="text-xs font-mono text-white/55 tabular-nums">{d.value.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Side metric cards */}
            <div className="space-y-3">
              <div className="card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lock size={11} className="text-[#30D158]" />
                  <p className="section-label">Value Blocked</p>
                </div>
                <p className="text-xl font-bold text-[#30D158] tabular-nums">
                  {currency.symbol}{convertedBlocked > 0
                    ? Math.round(convertedBlocked).toLocaleString()
                    : fmt(stats.total_blocked_amount)}
                </p>
                <p className="text-white/25 text-xs mt-1">
                  {currency.code} · {transactions.filter(t => t.decision === "BLOCK").length} blocked txns
                </p>
              </div>

              <div className="card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle size={11} className="text-[#0A84FF]" />
                  <p className="section-label">Approved Volume</p>
                </div>
                <p className="text-xl font-bold text-[#0A84FF] tabular-nums">
                  {currency.symbol}{convertedApproved > 0
                    ? Math.round(convertedApproved).toLocaleString()
                    : fmt(stats.total_amount - stats.total_blocked_amount)}
                </p>
                <p className="text-white/25 text-xs mt-1">
                  {currency.code} · {transactions.filter(t => t.decision === "APPROVE").length} approved txns
                </p>
              </div>

              {health?.artifact_version && (
                <div className="card p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Cpu size={11} className="text-white/30" />
                    <p className="section-label">Model Version</p>
                  </div>
                  <p className="text-xs font-mono text-white/45 truncate">
                    {health.artifact_version.slice(0, 22)}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Live Feed + Recent ── */}
        <div className="card overflow-hidden">
          {/* Tab bar */}
          <div className="flex border-b border-white/[0.06] px-1.5 pt-1.5 gap-1">
            {(["live", "recent"] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-all ${
                  activeTab === tab
                    ? "bg-white/[0.07] text-white border-b-2 border-[#0A84FF]"
                    : "text-white/35 hover:text-white/65 hover:bg-white/[0.03]"
                }`}
              >
                {tab === "live" ? (
                  <>
                    <span className="live-dot" style={{ width: 6, height: 6 }} />
                    Live Feed
                  </>
                ) : (
                  <><Activity size={13} />Recent Transactions</>
                )}
                {tab === "live" && liveAlerts.length > 0 && (
                  <span className="bg-[#0A84FF] text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full min-w-[18px] text-center">
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
                  <div className="w-12 h-12 rounded-2xl card flex items-center justify-center mx-auto">
                    <Radio size={20} className="text-white/18" />
                  </div>
                  <p className="text-white/35 font-medium text-sm">Waiting for live transactions…</p>
                  <p className="text-white/20 text-xs">
                    <Link href="/wallet" className="text-[#0A84FF] hover:underline">Open the eWallet</Link>
                    {" "}and submit a transaction to see real-time scoring here.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {liveAlerts.map((a) => (
                    <LiveAlertRow key={a.transaction_id} alert={a} currency={currency} />
                  ))}
                </div>
              )
            )}

            {/* Recent transactions table */}
            {activeTab === "recent" && (
              transactions.length === 0 ? (
                <div className="text-center py-14 space-y-3">
                  <div className="w-12 h-12 rounded-2xl card flex items-center justify-center mx-auto">
                    <Activity size={20} className="text-white/18" />
                  </div>
                  <p className="text-white/35 font-medium text-sm">No transactions yet</p>
                  <p className="text-white/20 text-xs">
                    <Link href="/wallet" className="text-[#0A84FF] hover:underline">Use the eWallet</Link>
                    {" "}to generate fraud scores.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/[0.05]">
                        {["Time","User",`Amount (${currency.code})`,"Type","Location","Risk","Decision","Signal",""].map((h, i) => (
                          <th key={i} className={`section-label py-3 pr-4 font-medium text-left ${i === 2 ? "text-right" : ""}`}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {transactions.map(tx => (
                        <RecentTxRow key={tx.transaction_id} tx={tx} display={display} currency={currency} />
                      ))}
                    </tbody>
                  </table>
                </div>
              )
            )}
          </div>
        </div>

        <p className="text-center text-white/15 text-xs pb-2">
          FraudShield · V HACK 2026 · {currency.flag} Displaying in {currency.code} at indicative rates
        </p>
      </div>
    </div>
  );
}
