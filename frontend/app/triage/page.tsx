"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  Shield, AlertTriangle, XCircle, CheckCircle,
  Search, ChevronRight, RefreshCw, Filter,
} from "lucide-react";
import { api, StoredTransaction, parseReasons } from "@/lib/api";
import { useCurrency, codeFromLocation } from "../CurrencyContext";

type FilterType = "all" | "BLOCK" | "FLAG";

const FLAGS: Record<string, string> = {
  BN: "🇧🇳", KH: "🇰🇭", ID: "🇮🇩", LA: "🇱🇦", MY: "🇲🇾",
  MM: "🇲🇲", PH: "🇵🇭", SG: "🇸🇬", TH: "🇹🇭", TL: "🇹🇱", VN: "🇻🇳",
};
function countryFlag(loc: string) {
  return FLAGS[loc?.split(", ").pop()?.toUpperCase() ?? ""] ?? "🌏";
}

// ─── Risk Bar ─────────────────────────────────────────────────────────────────
function RiskBar({ score }: { score: number }) {
  const color = score >= 70 ? "#FF453A" : score >= 40 ? "#FF9F0A" : "#30D158";
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-16 h-1 rounded-full bg-white/[0.08] overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${score}%`, background: color }} />
      </div>
      <span className="text-[11px] font-mono font-bold tabular-nums" style={{ color }}>{score}</span>
    </div>
  );
}

// ─── Decision Badge ───────────────────────────────────────────────────────────
function DecisionBadge({ decision }: { decision: "APPROVE" | "FLAG" | "BLOCK" }) {
  const cfg = {
    APPROVE: { cls: "badge-approve", Icon: CheckCircle },
    FLAG:    { cls: "badge-flag",    Icon: AlertTriangle },
    BLOCK:   { cls: "badge-block",   Icon: XCircle },
  }[decision];
  return (
    <span className={`flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full font-semibold tracking-wide w-fit ${cfg.cls}`}>
      <cfg.Icon size={10} />{decision}
    </span>
  );
}

// ─── Triage Card ─────────────────────────────────────────────────────────────
function TriageCard({ tx }: { tx: StoredTransaction }) {
  const { display } = useCurrency();
  const reasons = parseReasons(tx.reasons);
  const isBlock = tx.decision === "BLOCK";
  const isFlag  = tx.decision === "FLAG";
  const accent  = isBlock ? "#FF453A" : isFlag ? "#FF9F0A" : "#30D158";

  return (
    <div className="card-interactive rounded-xl p-5 group">
      {/* Header row */}
      <div className="flex items-start justify-between gap-4 flex-wrap mb-3">
        <div className="space-y-2">
          <div className="flex items-center gap-2.5 flex-wrap">
            <DecisionBadge decision={tx.decision} />
            <span className="text-white/45 text-xs font-mono">{tx.user_id}</span>
            <span className="text-white/20 text-xs">·</span>
            <span className="text-white font-semibold text-sm">
              {display(tx.amount, codeFromLocation(tx.location))}
            </span>
            <span className="text-white/30 text-xs capitalize">{tx.transaction_type}</span>
          </div>
          <p className="text-[10px] font-mono text-white/20">{tx.transaction_id}</p>
        </div>
        <div className="flex items-center gap-5 shrink-0">
          <RiskBar score={tx.risk_score} />
          <span className="text-xs text-white/20 whitespace-nowrap tabular-nums hidden sm:block">
            {new Date(tx.timestamp).toLocaleString()}
          </span>
        </div>
      </div>

      {/* Risk signals */}
      {reasons.length > 0 && (
        <div className="space-y-1.5 mb-3.5">
          {reasons.slice(0, 3).map((r, i) => (
            <div key={i} className="flex items-start gap-2.5 text-xs text-white/40">
              <div className="w-4 h-4 rounded-full flex items-center justify-center shrink-0 mt-0.5"
                style={{ background: `${accent}15` }}>
                {isBlock ? <XCircle size={9} style={{ color: accent }} />
                  : isFlag ? <AlertTriangle size={9} style={{ color: accent }} />
                  : <CheckCircle size={9} style={{ color: accent }} />}
              </div>
              <span className="leading-snug">{r}</span>
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-white/[0.05]">
        <div className="flex gap-4 text-[11px] text-white/22 flex-wrap">
          {tx.location && <span>{countryFlag(tx.location)} {tx.location}</span>}
          {tx.device_type && <span>{tx.device_type}</span>}
          {tx.confidence != null && (
            <span>{(tx.confidence * 100).toFixed(0)}% confidence</span>
          )}
        </div>
        <Link
          href={`/case/${tx.transaction_id}`}
          className="flex items-center gap-1 text-xs text-[#0A84FF] hover:text-[#0A84FF]/70 transition-colors font-medium"
        >
          Investigate <ChevronRight size={12} />
        </Link>
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function TriagePage() {
  const { currency } = useCurrency();
  const [transactions, setTransactions] = useState<StoredTransaction[]>([]);
  const [filter, setFilter]             = useState<FilterType>("all");
  const [search, setSearch]             = useState("");
  const [loading, setLoading]           = useState(true);
  const [lastRefresh, setLastRefresh]   = useState<Date>(new Date());

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [flags, blocks] = await Promise.all([
        api.getDashboardTransactions(50, 0, "FLAG"),
        api.getDashboardTransactions(50, 0, "BLOCK"),
      ]);
      const combined = [...blocks.transactions, ...flags.transactions].sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
      setTransactions(combined);
      setLastRefresh(new Date());
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    load();
    const i = setInterval(load, 30_000);
    return () => clearInterval(i);
  }, [load]);

  const filtered = transactions.filter(tx => {
    if (filter !== "all" && tx.decision !== filter) return false;
    if (search) {
      const q = search.toLowerCase();
      return tx.user_id.toLowerCase().includes(q) || tx.transaction_id.toLowerCase().includes(q);
    }
    return true;
  });

  const blockCount = transactions.filter(t => t.decision === "BLOCK").length;
  const flagCount  = transactions.filter(t => t.decision === "FLAG").length;

  return (
    <div className="min-h-screen bg-[#09090b]">
      <div className="max-w-5xl mx-auto px-5 py-8 space-y-6">

        {/* ── Header ── */}
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <AlertTriangle size={22} style={{ color: "#FF9F0A" }} />
              Suspicious Transactions
            </h1>
            <p className="text-white/35 text-sm mt-1.5">
              Flagged and blocked transactions requiring review · {currency.flag} {currency.code}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Mini stats */}
            <div className="hidden sm:flex items-center gap-3">
              {blockCount > 0 && (
                <span className="flex items-center gap-1.5 text-xs text-[#FF453A] badge-block px-2.5 py-1 rounded-full">
                  <XCircle size={11} />{blockCount} blocked
                </span>
              )}
              {flagCount > 0 && (
                <span className="flex items-center gap-1.5 text-xs text-[#FF9F0A] badge-flag px-2.5 py-1 rounded-full">
                  <AlertTriangle size={11} />{flagCount} flagged
                </span>
              )}
            </div>
            <span className="text-xs text-white/20 tabular-nums hidden sm:block">
              Updated {lastRefresh.toLocaleTimeString()}
            </span>
            <button onClick={load}
              className="p-2 rounded-xl card hover:bg-white/[0.07] transition-all">
              <RefreshCw size={14} className={`text-white/35 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* ── Search + Filter ── */}
        <div className="flex gap-3 flex-wrap items-center">
          {/* Search */}
          <div className="relative flex-1 min-w-52">
            <Search size={13} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-white/25" />
            <input
              className="input-apple w-full pl-9 pr-4 py-2.5 text-sm"
              placeholder="Search by user ID or transaction ID…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          {/* Filter pills */}
          <div className="flex items-center gap-1 p-1 rounded-xl card">
            <Filter size={11} className="text-white/20 ml-1.5 mr-0.5" />
            {(["all", "BLOCK", "FLAG"] as FilterType[]).map(f => {
              const active = filter === f;
              const activeStyle = f === "BLOCK"
                ? { background: "rgba(255,69,58,0.15)", color: "#FF453A" }
                : f === "FLAG"
                ? { background: "rgba(255,159,10,0.15)", color: "#FF9F0A" }
                : { background: "rgba(255,255,255,0.10)", color: "rgba(255,255,255,0.90)" };
              return (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className="px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all"
                  style={active ? activeStyle : { color: "rgba(255,255,255,0.35)" }}
                >
                  {f === "all" ? "All" : f}
                </button>
              );
            })}
          </div>
        </div>

        {/* ── Content ── */}
        {loading ? (
          <div className="text-center py-20">
            <RefreshCw size={20} className="animate-spin mx-auto text-white/20 mb-3" />
            <p className="text-white/25 text-sm">Loading suspicious transactions…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-20 space-y-4">
            <div className="w-14 h-14 rounded-3xl card flex items-center justify-center mx-auto">
              <Shield size={24} className="text-white/18" />
            </div>
            <p className="text-white/40 font-medium">
              {transactions.length === 0
                ? "No suspicious transactions yet"
                : "No results match this filter"}
            </p>
            {transactions.length === 0 && (
              <p className="text-white/20 text-sm">
                Submit transactions via the{" "}
                <Link href="/wallet" className="text-[#0A84FF] hover:underline">eWallet</Link>
                {" "}to generate fraud scores.
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <p className="section-label">
              {filtered.length} result{filtered.length !== 1 ? "s" : ""}
              {filter !== "all" && ` · ${filter}`}
            </p>
            {filtered.map(tx => (
              <TriageCard key={tx.transaction_id} tx={tx} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
