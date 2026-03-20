"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  Shield, AlertTriangle, XCircle, Activity, RefreshCw,
  Wallet, TrendingUp, ChevronRight, Radio, Cpu,
  CheckCircle, ArrowUpRight, Lock, Target, Clock,
  BarChart2, Zap,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import {
  api, DashboardStats, DashboardCharts, StoredTransaction,
  HealthStatus, LiveAlert, parseReasons, useAlertStream,
} from "@/lib/api";
import { useCurrency, codeFromLocation } from "./CurrencyContext";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmt(n: number | undefined, dec = 0) {
  if (n == null) return "—";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: dec,
    maximumFractionDigits: dec,
  });
}

function fmtTime(iso: string) {
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch { return iso; }
}

/** Parse "2026-03-20T14" → "14:00" for chart x-axis labels */
function formatHour(isoHour: string): string {
  try {
    const h = parseInt(isoHour.split("T")[1] ?? "0", 10);
    return `${h.toString().padStart(2, "0")}:00`;
  } catch { return isoHour; }
}

/** Bar color based on risk score bucket start */
function riskColor(bucketStart: number): string {
  if (bucketStart >= 70) return "#FF453A";
  if (bucketStart >= 40) return "#FF9F0A";
  return "#30D158";
}

// ─── Decision Badge ───────────────────────────────────────────────────────────
function DecisionBadge({ decision }: { decision: "APPROVE" | "FLAG" | "BLOCK" }) {
  const cls = {
    APPROVE: "badge-approve",
    FLAG: "badge-flag",
    BLOCK: "badge-block",
  }[decision];
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
      <span className="text-[11px] font-mono font-bold tabular-nums" style={{ color }}>
        {score}
      </span>
    </div>
  );
}

// ─── Primary Stat Card ────────────────────────────────────────────────────────
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

// ─── Secondary Metric Card ────────────────────────────────────────────────────
function MetricCard({
  label, value, sub, icon, color = "text-white/80",
}: {
  label: string; value: string; sub?: string;
  icon: React.ReactNode; color?: string;
}) {
  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <p className="section-label">{label}</p>
      </div>
      <p className={`text-xl font-bold tabular-nums ${color}`}>{value}</p>
      {sub && <p className="text-white/25 text-xs mt-1">{sub}</p>}
    </div>
  );
}

// ─── Chart Tooltip ────────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; fill?: string; color?: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-xl px-3 py-2 text-sm shadow-2xl"
      style={{ background: "rgba(24,24,27,0.97)", border: "1px solid rgba(255,255,255,0.09)" }}
    >
      <p className="text-white/40 text-xs mb-1.5">{label}</p>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ background: p.fill || p.color || "#fff" }} />
          <span className="text-white/55 text-xs capitalize">{p.name}:</span>
          <span className="text-white font-semibold tabular-nums text-xs">
            {p.value.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Live Alert Row ───────────────────────────────────────────────────────────
function LiveAlertRow({ alert, currency }: { alert: LiveAlert; currency: { symbol: string } }) {
  const isBlock = alert.decision === "BLOCK";
  const isFlag = alert.decision === "FLAG";
  const rowColor = isBlock
    ? "border-[#FF453A]/20 bg-[#FF453A]/[0.04]"
    : isFlag
    ? "border-[#FF9F0A]/15 bg-[#FF9F0A]/[0.03]"
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
      <span className="text-[11px] text-white/20 whitespace-nowrap tabular-nums">
        {fmtTime(alert.timestamp)}
      </span>
      <Link
        href={`/case/${alert.transaction_id}`}
        className="text-white/20 hover:text-[#0A84FF] transition-colors"
      >
        <ChevronRight size={14} />
      </Link>
    </div>
  );
}

// ─── Recent TX Row ────────────────────────────────────────────────────────────
function RecentTxRow({
  tx, display, currency,
}: {
  tx: StoredTransaction;
  display: (amount: number, fromCode: string) => string;
  currency: { code: string };
}) {
  const reasons = parseReasons(tx.reasons);
  const fromCode = codeFromLocation(tx.location);
  return (
    <tr className="hover:bg-white/[0.025] transition-colors group">
      <td className="py-3 pr-4 text-[11px] text-white/25 whitespace-nowrap tabular-nums">
        {fmtTime(tx.timestamp)}
      </td>
      <td className="py-3 pr-4 font-mono text-xs text-white/45 truncate max-w-[90px]">
        {tx.user_id}
      </td>
      <td className="py-3 pr-4 text-right text-sm font-semibold text-white tabular-nums">
        {display(tx.amount, fromCode)}
      </td>
      <td className="py-3 pr-4 text-xs text-white/35 capitalize">{tx.transaction_type}</td>
      <td className="py-3 pr-4 text-xs text-white/25">{tx.location || "—"}</td>
      <td className="py-3 pr-4"><RiskChip score={tx.risk_score} /></td>
      <td className="py-3 pr-4"><DecisionBadge decision={tx.decision} /></td>
      <td className="py-3 pr-4 text-xs text-white/25 max-w-[140px] truncate">
        {reasons[0] ?? "—"}
      </td>
      <td className="py-3">
        <Link
          href={`/case/${tx.transaction_id}`}
          className="text-white/15 hover:text-[#0A84FF] transition-colors opacity-0 group-hover:opacity-100"
        >
          <ChevronRight size={13} />
        </Link>
      </td>
    </tr>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────
export default function DashboardPage() {
  const { currency, display } = useCurrency();

  const [health, setHealth]             = useState<HealthStatus | null>(null);
  const [stats, setStats]               = useState<DashboardStats | null>(null);
  const [transactions, setTransactions] = useState<StoredTransaction[]>([]);
  const [charts, setCharts]             = useState<DashboardCharts | null>(null);
  const [metrics, setMetrics]           = useState<Record<string, unknown> | null>(null);
  const [liveAlerts, setLiveAlerts]     = useState<LiveAlert[]>([]);
  const [activeTab, setActiveTab]       = useState<"live" | "recent">("live");
  const [loading, setLoading]           = useState(true);
  const [lastRefresh, setLastRefresh]   = useState<Date>(new Date());

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [h, s, t, c] = await Promise.all([
        api.getHealth(),
        api.getDashboardStats(),
        api.getDashboardTransactions(25),
        api.getDashboardCharts(),
      ]);
      setHealth(h);
      setStats(s);
      setTransactions(t.transactions);
      setCharts(c);
      setLastRefresh(new Date());
      if (h.model_loaded) {
        api.getMetrics().then(m => setMetrics(m)).catch(() => null);
      }
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
      api.getDashboardTransactions(25),
      api.getDashboardCharts(),
    ]).then(([s, t, c]) => {
      setStats(s);
      setTransactions(t.transactions);
      setCharts(c);
      setLastRefresh(new Date());
    }).catch(() => null);
  }, []));

  // ── Derived ─────────────────────────────────────────────────────────────────
  const modelLoaded = health?.model_loaded ?? false;
  const fraudRatePct = stats && stats.total > 0
    ? parseFloat((stats.fraud_rate * 100).toFixed(1))
    : null;
  const threatLevel = fraudRatePct === null ? null
    : fraudRatePct > 15 ? "High"
    : fraudRatePct > 5  ? "Elevated"
    : "Normal";

  // Decision distribution chart
  const decisionData = stats ? [
    { name: "Approved", value: stats.approved_count, color: "#30D158" },
    { name: "Flagged",  value: stats.flagged_count,  color: "#FF9F0A" },
    { name: "Blocked",  value: stats.blocked_count,  color: "#FF453A" },
  ] : [];

  // Hourly trend: add formatted label for x-axis
  const trendData = (charts?.hourly_trend ?? []).map(h => ({
    ...h,
    label: formatHour(h.hour),
  }));

  // Risk distribution — always 10 buckets from backend
  const riskData = charts?.risk_distribution ?? [];
  const hasRiskData = riskData.some(r => r.count > 0);

  // Recent blocked (from latest 25 fetched)
  const blockedTxs = transactions.filter(t => t.decision === "BLOCK").slice(0, 5);

  // Model metrics (from /api/metrics)
  const xgbAuc  = (metrics as any)?.xgboost?.roc_auc as number | undefined;
  const lgbmAuc = (metrics as any)?.lightgbm?.roc_auc as number | undefined;
  const ifAuc   = (metrics as any)?.isolation_forest?.roc_auc as number | undefined;
  const lofAuc  = (metrics as any)?.lof?.roc_auc as number | undefined;
  const metaAuc = ((metrics as any)?.meta_ensemble?.roc_auc
    ?? (metrics as any)?.ensemble?.roc_auc) as number | undefined;
  const latMs   = (metrics as any)?.latency?.mean_ms as number | undefined;

  const hasData = stats !== null && stats.total > 0;

  return (
    <div className="min-h-screen bg-[#09090b]">
      <div className="max-w-6xl mx-auto px-5 py-8 space-y-6">

        {/* ── Header ── */}
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <Shield size={22} style={{ color: "#0A84FF" }} />
              Fraud Intelligence Platform
            </h1>
            <p className="text-white/35 text-sm mt-1">
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
                threatLevel === "High"
                  ? "bg-[#FF453A]/10 border-[#FF453A]/20 text-[#FF453A]"
                  : threatLevel === "Elevated"
                  ? "bg-[#FF9F0A]/10 border-[#FF9F0A]/20 text-[#FF9F0A]"
                  : "bg-[#30D158]/10 border-[#30D158]/20 text-[#30D158]"
              }`}>
                <Lock size={10} />Threat: {threatLevel}
              </span>
            )}

            <Link
              href="/wallet"
              className="flex items-center gap-1.5 bg-[#0A84FF] hover:bg-[#0A84FF]/85 text-white text-xs px-4 py-1.5 rounded-full font-semibold transition-all shadow-lg shadow-[#0A84FF]/20"
            >
              <Wallet size={12} />Open eWallet
            </Link>
            <button
              onClick={refresh}
              className="p-2 rounded-xl card hover:bg-white/[0.07] transition-all"
              title={`Last refresh: ${lastRefresh.toLocaleTimeString()}`}
            >
              <RefreshCw size={14} className={`text-white/35 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* ── Model Offline Banner ── */}
        {!modelLoaded && (
          <div
            className="card p-5"
            style={{ borderColor: "rgba(255,159,10,0.25)", background: "rgba(255,159,10,0.05)" }}
          >
            <div className="flex items-center gap-2 text-[#FF9F0A] font-semibold mb-2.5">
              <AlertTriangle size={16} />Fraud Detection Model Not Loaded
            </div>
            <p className="text-white/45 text-sm mb-4">
              Train the model to activate real fraud scoring. The dashboard populates as wallet
              transactions are submitted.
            </p>
            <div
              className="rounded-xl p-4 text-xs font-mono space-y-1 text-[#30D158]"
              style={{ background: "rgba(0,0,0,0.5)", border: "1px solid rgba(255,255,255,0.07)" }}
            >
              <p className="text-white/25"># Download IEEE-CIS dataset from Kaggle, then:</p>
              <p>cd backend</p>
              <p>python training/train_engine.py --data-dir ./data/ieee-cis</p>
              <p className="text-white/25"># Restart backend</p>
              <p>python main.py</p>
            </div>
          </div>
        )}

        {/* ── Primary KPIs ── */}
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
            sub={
              stats && stats.total > 0
                ? `${(stats.blocked_count / stats.total * 100).toFixed(1)}% of total`
                : "No transactions yet"
            }
            icon={<XCircle size={15} />}
            accent="text-[#FF453A]"
            trend={stats && stats.blocked_count > 0 ? "up" : null}
          />
          <StatCard
            label="Flagged for Review"
            value={stats ? fmt(stats.flagged_count) : "—"}
            sub={
              stats && stats.total > 0
                ? `${(stats.flagged_count / stats.total * 100).toFixed(1)}% of total`
                : "Pending analyst review"
            }
            icon={<AlertTriangle size={15} />}
            accent="text-[#FF9F0A]"
          />
          <StatCard
            label="Fraud Rate"
            value={
              stats && stats.total > 0
                ? `${(stats.fraud_rate * 100).toFixed(1)}%`
                : "—"
            }
            sub={
              stats
                ? `${stats.approved_count} approved of ${stats.total} total`
                : undefined
            }
            icon={<TrendingUp size={15} />}
            accent={stats && stats.fraud_rate > 0.1 ? "text-[#FF453A]" : "text-white"}
          />
        </div>

        {/* ── Secondary Metrics ── */}
        {hasData && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard
              label="Value Blocked"
              value={`${currency.symbol}${Math.round(stats!.total_blocked_amount).toLocaleString()}`}
              sub={`${stats!.blocked_count} blocked txns`}
              icon={<Lock size={11} className="text-[#FF453A]" />}
              color="text-[#FF453A]"
            />
            <MetricCard
              label="Approved Volume"
              value={`${currency.symbol}${Math.round(stats!.total_approved_amount).toLocaleString()}`}
              sub={`${stats!.approved_count} approved txns`}
              icon={<CheckCircle size={11} className="text-[#30D158]" />}
              color="text-[#30D158]"
            />
            <MetricCard
              label="Avg Risk Score"
              value={stats!.avg_risk_score > 0 ? fmt(stats!.avg_risk_score, 1) : "—"}
              sub={
                stats!.avg_risk_score >= 70
                  ? "High average risk"
                  : stats!.avg_risk_score >= 40
                  ? "Moderate risk level"
                  : "Low risk baseline"
              }
              icon={<Target size={11} className="text-[#BF5AF2]" />}
              color={
                stats!.avg_risk_score >= 70
                  ? "text-[#FF453A]"
                  : stats!.avg_risk_score >= 40
                  ? "text-[#FF9F0A]"
                  : "text-[#30D158]"
              }
            />
            <MetricCard
              label="Approval Rate"
              value={
                stats!.total > 0
                  ? `${(stats!.approved_count / stats!.total * 100).toFixed(1)}%`
                  : "—"
              }
              sub={`${stats!.approved_count} of ${stats!.total} transactions`}
              icon={<Activity size={11} className="text-[#0A84FF]" />}
              color="text-[#0A84FF]"
            />
          </div>
        )}

        {/* ── Charts Row: Decision Distribution + Hourly Trend ── */}
        {hasData && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">

            {/* Decision Distribution — 2/5 */}
            <div className="lg:col-span-2 card p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <BarChart2 size={12} className="text-white/30" />
                  <p className="section-label">Decision Distribution</p>
                </div>
                <p className="text-[11px] text-white/25 tabular-nums">
                  {stats!.total.toLocaleString()} total
                </p>
              </div>
              <ResponsiveContainer width="100%" height={155}>
                <BarChart data={decisionData} barSize={46} barGap={8}>
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "rgba(255,255,255,0.30)", fontSize: 12 }}
                    axisLine={false} tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "rgba(255,255,255,0.28)", fontSize: 11 }}
                    axisLine={false} tickLine={false}
                  />
                  <Tooltip
                    content={<ChartTooltip />}
                    cursor={{ fill: "rgba(255,255,255,0.03)", radius: 8 }}
                  />
                  <Bar dataKey="value" name="Count" radius={[6, 6, 0, 0]}>
                    {decisionData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} fillOpacity={0.82} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="flex gap-4 mt-2 pt-3 border-t border-white/[0.05]">
                {decisionData.map(d => (
                  <div key={d.name} className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                    <span className="text-xs text-white/35">{d.name}</span>
                    <span className="text-xs font-mono text-white/55 tabular-nums">
                      {d.value.toLocaleString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Hourly Trend — 3/5 */}
            <div className="lg:col-span-3 card p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Clock size={12} className="text-white/30" />
                  <p className="section-label">Transaction Trend · Last 24h</p>
                </div>
                <span className="text-[11px] text-white/25">Stacked by decision</span>
              </div>

              {trendData.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-[155px] gap-2">
                  <Clock size={22} className="text-white/12" />
                  <p className="text-white/25 text-sm">No hourly data yet</p>
                  <p className="text-white/15 text-xs">Data appears as transactions accumulate</p>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={155}>
                  <BarChart
                    data={trendData}
                    barSize={Math.max(10, Math.min(36, 220 / trendData.length))}
                    barCategoryGap="25%"
                  >
                    <XAxis
                      dataKey="label"
                      tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 10 }}
                      axisLine={false} tickLine={false}
                    />
                    <YAxis
                      tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 10 }}
                      axisLine={false} tickLine={false} allowDecimals={false}
                    />
                    <Tooltip
                      content={<ChartTooltip />}
                      cursor={{ fill: "rgba(255,255,255,0.03)", radius: 4 }}
                    />
                    <Bar dataKey="approved" name="Approved" stackId="a" fill="#30D158" fillOpacity={0.80} />
                    <Bar dataKey="flagged"  name="Flagged"  stackId="a" fill="#FF9F0A" fillOpacity={0.80} />
                    <Bar dataKey="blocked"  name="Blocked"  stackId="a" fill="#FF453A" fillOpacity={0.80} radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}

              {trendData.length > 0 && (
                <div className="flex gap-4 mt-2 pt-3 border-t border-white/[0.05]">
                  {[
                    { name: "Approved", color: "#30D158" },
                    { name: "Flagged",  color: "#FF9F0A" },
                    { name: "Blocked",  color: "#FF453A" },
                  ].map(d => (
                    <div key={d.name} className="flex items-center gap-1.5">
                      <div className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                      <span className="text-xs text-white/35">{d.name}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Risk Score Distribution ── */}
        {hasData && (
          <div className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Target size={12} className="text-white/30" />
                <p className="section-label">Risk Score Distribution</p>
              </div>
              <span className="text-[11px] text-white/25">Score buckets 0–100</span>
            </div>

            {!hasRiskData ? (
              <div className="flex items-center justify-center h-[110px]">
                <p className="text-white/20 text-sm">Processing risk data…</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={110}>
                <BarChart data={riskData} barSize={30} barCategoryGap="18%">
                  <XAxis
                    dataKey="bucket"
                    tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 9 }}
                    axisLine={false} tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 10 }}
                    axisLine={false} tickLine={false} allowDecimals={false}
                  />
                  <Tooltip
                    content={<ChartTooltip />}
                    cursor={{ fill: "rgba(255,255,255,0.03)", radius: 4 }}
                  />
                  <Bar dataKey="count" name="Transactions" radius={[4, 4, 0, 0]}>
                    {riskData.map((entry, i) => (
                      <Cell key={i} fill={riskColor(entry.bucket_start)} fillOpacity={0.82} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}

            <div className="flex gap-5 mt-3 pt-3 border-t border-white/[0.05]">
              {[
                { label: "Low risk (0–39)",    color: "#30D158" },
                { label: "Flagged range (40–69)", color: "#FF9F0A" },
                { label: "Blocked range (70–100)", color: "#FF453A" },
              ].map(d => (
                <div key={d.label} className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                  <span className="text-xs text-white/30">{d.label}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Live Feed + Recent Transactions ── */}
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
                    <Link href="/wallet" className="text-[#0A84FF] hover:underline">
                      Open the eWallet
                    </Link>
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
                    <Link href="/wallet" className="text-[#0A84FF] hover:underline">
                      Use the eWallet
                    </Link>
                    {" "}to generate fraud scores.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/[0.05]">
                        {["Time", "User", `Amount (${currency.code})`, "Type", "Location", "Risk", "Decision", "Signal", ""].map((h, i) => (
                          <th
                            key={i}
                            className={`section-label py-3 pr-4 font-medium text-left ${i === 2 ? "text-right" : ""}`}
                          >
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

        {/* ── Recently Blocked ── */}
        {blockedTxs.length > 0 && (
          <div className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <XCircle size={13} className="text-[#FF453A]" />
                <p className="section-label">Recently Blocked</p>
              </div>
              <Link
                href="/triage"
                className="text-xs text-[#0A84FF] hover:underline flex items-center gap-1"
              >
                View all in Triage <ChevronRight size={11} />
              </Link>
            </div>
            <div className="space-y-2">
              {blockedTxs.map(tx => {
                const reasons = parseReasons(tx.reasons);
                return (
                  <div
                    key={tx.transaction_id}
                    className="flex items-center gap-3 p-3 rounded-xl border border-[#FF453A]/15 bg-[#FF453A]/[0.03]"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 flex-wrap">
                        <span className="font-mono text-xs text-white/50">{tx.user_id}</span>
                        <span className="text-sm font-semibold text-white tabular-nums">
                          {currency.symbol}{Math.round(tx.amount).toLocaleString()}
                        </span>
                        <span className="text-xs text-white/30 capitalize">{tx.transaction_type}</span>
                        {tx.location && (
                          <span className="text-xs text-white/20">{tx.location}</span>
                        )}
                      </div>
                      {reasons[0] && (
                        <p className="text-xs text-[#FF453A]/60 mt-0.5 truncate">{reasons[0]}</p>
                      )}
                    </div>
                    <RiskChip score={tx.risk_score} />
                    <span className="text-[11px] text-white/20 whitespace-nowrap tabular-nums">
                      {fmtTime(tx.timestamp)}
                    </span>
                    <Link
                      href={`/case/${tx.transaction_id}`}
                      className="text-white/20 hover:text-[#0A84FF] transition-colors"
                    >
                      <ChevronRight size={13} />
                    </Link>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Model Performance ── */}
        {modelLoaded && metrics && (
          <div className="card p-5">
            <div className="flex items-center gap-2 mb-4">
              <Cpu size={13} className="text-white/30" />
              <p className="section-label">Model Performance</p>
              {health?.artifact_version && (
                <span className="text-[11px] text-white/20 font-mono ml-auto truncate max-w-[220px]">
                  v{health.artifact_version.slice(0, 19)}
                </span>
              )}
            </div>
            <div className="grid grid-cols-3 lg:grid-cols-6 gap-3">
              {[
                { label: "XGBoost",          value: xgbAuc  != null ? xgbAuc.toFixed(4)  : "—", color: "text-[#0A84FF]" },
                { label: "LightGBM",         value: lgbmAuc != null ? lgbmAuc.toFixed(4) : "—", color: "text-[#BF5AF2]" },
                { label: "Isolation Forest", value: ifAuc   != null ? ifAuc.toFixed(4)   : "—", color: "text-[#FF6B35]" },
                { label: "LOF",              value: lofAuc  != null ? lofAuc.toFixed(4)  : "—", color: "text-[#FFD60A]" },
                { label: "Meta-Ensemble",    value: metaAuc != null ? metaAuc.toFixed(4) : "—", color: "text-[#30D158]" },
                { label: "Avg Latency",      value: latMs   != null ? `${latMs.toFixed(2)}ms`   : "—", color: "text-[#FF9F0A]" },
              ].map(m => (
                <div
                  key={m.label}
                  className="rounded-xl p-3"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}
                >
                  <p className="text-[11px] text-white/30 mb-1">{m.label}</p>
                  <p className={`text-lg font-bold tabular-nums font-mono ${m.color}`}>{m.value}</p>
                </div>
              ))}
            </div>
            {health?.thresholds && (
              <p className="text-[11px] text-white/18 mt-3">
                ROC-AUC on IEEE-CIS Fraud Detection dataset (590k transactions, 3.28% fraud rate) · FLAG ≥ {health.thresholds.flag} · BLOCK ≥ {health.thresholds.block}
              </p>
            )}
          </div>
        )}

        <p className="text-center text-white/15 text-xs pb-2">
          FraudShield · V HACK 2026 · {currency.flag} Displaying in {currency.code} at indicative rates
        </p>
      </div>
    </div>
  );
}
