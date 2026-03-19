"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  Shield, AlertTriangle, XCircle, Search, ChevronRight, RefreshCw,
} from "lucide-react";
import { api, StoredTransaction, parseReasons } from "@/lib/api";
import { useCurrency, codeFromLocation } from "../CurrencyContext";

type FilterType = "all" | "BLOCK" | "FLAG";

const FLAGS: Record<string, string> = {
  BN: "🇧🇳", KH: "🇰🇭", ID: "🇮🇩", LA: "🇱🇦", MY: "🇲🇾",
  MM: "🇲🇲", PH: "🇵🇭", SG: "🇸🇬", TH: "🇹🇭", TL: "🇹🇱", VN: "🇻🇳",
};

function countryFlag(location: string): string {
  const code = location?.split(", ").pop()?.toUpperCase() ?? "";
  return FLAGS[code] ?? "🌏";
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

function RiskScore({ score }: { score: number }) {
  const color = score >= 70 ? "#FF453A" : score >= 40 ? "#FF9F0A" : "#30D158";
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-20 bg-white/[0.08] rounded-full h-1">
        <div className="h-1 rounded-full transition-all" style={{ width: `${score}%`, background: color }} />
      </div>
      <span className="text-xs font-mono font-bold tabular-nums" style={{ color }}>{score}</span>
    </div>
  );
}

export default function TriagePage() {
  const { currency, display } = useCurrency();
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
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 30_000);
    return () => clearInterval(interval);
  }, [load]);

  const filtered = transactions.filter(tx => {
    if (filter !== "all" && tx.decision !== filter) return false;
    if (search && !tx.user_id.toLowerCase().includes(search.toLowerCase()) &&
        !tx.transaction_id.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="min-h-screen bg-[#09090b]">
      <div className="max-w-5xl mx-auto px-5 py-8 space-y-6">

        {/* ── Header ── */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <AlertTriangle size={22} className="text-[#FF9F0A]" />
              Suspicious Transactions
            </h1>
            <p className="text-white/40 text-sm mt-1.5">
              Flagged and blocked transactions requiring review · {currency.flag} {currency.code}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-white/25 tabular-nums">
              Updated {lastRefresh.toLocaleTimeString()}
            </span>
            <button
              onClick={load}
              className="p-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.09] border border-white/[0.07] transition-all"
            >
              <RefreshCw size={14} className={`text-white/40 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* ── Filters ── */}
        <div className="flex gap-3 flex-wrap items-center">
          <div className="relative flex-1 min-w-52">
            <Search size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-white/30" />
            <input
              className="input-apple w-full pl-9 pr-4 py-2.5 text-sm"
              placeholder="Search by user or transaction ID…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <div className="flex gap-1.5 p-1 rounded-xl bg-white/[0.04] border border-white/[0.07]">
            {(["all", "BLOCK", "FLAG"] as FilterType[]).map(f => {
              const active = filter === f;
              const activeColor = f === "BLOCK" ? "bg-[#FF453A]/15 text-[#FF453A]" : f === "FLAG" ? "bg-[#FF9F0A]/15 text-[#FF9F0A]" : "bg-white/[0.10] text-white";
              return (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    active ? activeColor : "text-white/40 hover:text-white/70"
                  }`}
                >
                  {f === "all" ? "All" : f}
                </button>
              );
            })}
          </div>
        </div>

        {/* ── Results ── */}
        {loading ? (
          <div className="text-center py-20 space-y-3">
            <RefreshCw size={22} className="animate-spin mx-auto text-white/20" />
            <p className="text-white/30 text-sm">Loading…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-20 space-y-4">
            <div className="w-14 h-14 rounded-3xl bg-white/[0.04] flex items-center justify-center mx-auto border border-white/[0.07]">
              <Shield size={26} className="text-white/20" />
            </div>
            <p className="text-white/50 font-medium">
              {transactions.length === 0 ? "No suspicious transactions yet" : "No results for this filter"}
            </p>
            {transactions.length === 0 && (
              <p className="text-white/25 text-sm">
                Submit transactions via the{" "}
                <Link href="/wallet" className="text-[#0A84FF] hover:underline">eWallet</Link>{" "}
                to generate fraud scores.
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-2.5">
            <p className="text-xs text-white/30">{filtered.length} result{filtered.length !== 1 ? "s" : ""}</p>
            {filtered.map(tx => {
              const reasons = parseReasons(tx.reasons);
              return (
                <div
                  key={tx.transaction_id}
                  className="card p-5 hover:bg-white/[0.06] transition-all cursor-default"
                >
                  <div className="flex items-start justify-between gap-4 flex-wrap">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2.5 flex-wrap">
                        <DecisionBadge decision={tx.decision} />
                        <span className="text-white/50 text-xs font-mono">{tx.user_id}</span>
                        <span className="text-white/20 text-xs">·</span>
                        <span className="text-white font-semibold">{display(tx.amount, codeFromLocation(tx.location))}</span>
                        <span className="text-white/30 text-xs capitalize">{tx.transaction_type}</span>
                      </div>
                      <p className="text-[11px] font-mono text-white/25">{tx.transaction_id}</p>
                    </div>
                    <div className="flex items-center gap-5 shrink-0">
                      <RiskScore score={tx.risk_score} />
                      <span className="text-xs text-white/25 whitespace-nowrap tabular-nums">
                        {new Date(tx.timestamp).toLocaleString()}
                      </span>
                    </div>
                  </div>

                  {reasons.length > 0 && (
                    <div className="mt-3.5 space-y-1.5">
                      {reasons.slice(0, 3).map((r, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs text-white/40">
                          <span className="text-white/20 mt-0.5 shrink-0">•</span>
                          <span>{r}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="mt-4 pt-3.5 border-t border-white/[0.06] flex items-center justify-between">
                    <div className="flex gap-4 text-[11px] text-white/25">
                      {tx.location && <span>{countryFlag(tx.location)} {tx.location}</span>}
                      {tx.device_type && <span>{tx.device_type}</span>}
                      <span>{tx.confidence ? (tx.confidence * 100).toFixed(0) + "% confidence" : "—"}</span>
                    </div>
                    <Link
                      href={`/case/${tx.transaction_id}`}
                      className="flex items-center gap-1 text-xs text-[#0A84FF] hover:text-[#0A84FF]/80 transition-colors font-medium"
                    >
                      View Case <ChevronRight size={12} />
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
