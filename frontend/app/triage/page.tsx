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
  const styles = {
    APPROVE: "bg-green-900 text-green-300",
    FLAG:    "bg-yellow-900 text-yellow-300",
    BLOCK:   "bg-red-900 text-red-300",
  };
  const icons = { APPROVE: "✓", FLAG: "⚠", BLOCK: "✕" };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium flex items-center gap-1 w-fit ${styles[decision]}`}>
      {icons[decision]} {decision}
    </span>
  );
}

function RiskBar({ score }: { score: number }) {
  const color = score >= 70 ? "bg-red-500" : score >= 40 ? "bg-yellow-500" : "bg-green-500";
  const text  = score >= 70 ? "text-red-400" : score >= 40 ? "text-yellow-400" : "text-green-400";
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 bg-gray-700 rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className={`text-xs font-bold font-mono ${text}`}>{score}</span>
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
    <div className="min-h-screen bg-[#0a0e1a] text-white">
      <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-xl font-bold flex items-center gap-2">
              <AlertTriangle size={20} className="text-yellow-400" />
              Suspicious Transactions
            </h1>
            <p className="text-gray-400 text-sm mt-1">
              Flagged and blocked transactions requiring review · Amounts in {currency.flag} {currency.code}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500">
              Updated {lastRefresh.toLocaleTimeString()}
            </span>
            <button
              onClick={load}
              className="p-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 transition"
              title="Refresh"
            >
              <RefreshCw size={14} className={`text-gray-400 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex gap-3 flex-wrap">
          <div className="relative flex-1 min-w-48">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input
              className="w-full bg-gray-800 border border-gray-700 rounded-xl pl-8 pr-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              placeholder="Search by user or transaction ID…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            {(["all", "BLOCK", "FLAG"] as FilterType[]).map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-2 rounded-xl text-sm font-medium transition ${
                  filter === f
                    ? f === "BLOCK" ? "bg-red-600 text-white"
                      : f === "FLAG" ? "bg-yellow-600 text-white"
                      : "bg-blue-600 text-white"
                    : "bg-gray-800 text-gray-400 hover:text-white"
                }`}
              >
                {f === "all" ? "All" : f}
              </button>
            ))}
          </div>
        </div>

        {/* Results */}
        {loading ? (
          <div className="text-center py-16 text-gray-500">
            <RefreshCw size={24} className="animate-spin mx-auto mb-3" />
            <p>Loading suspicious transactions…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-16 text-gray-500 space-y-3">
            <Shield size={32} className="mx-auto text-gray-600" />
            <p className="font-medium text-gray-400">
              {transactions.length === 0
                ? "No suspicious transactions yet"
                : "No results for current filter"}
            </p>
            {transactions.length === 0 && (
              <p className="text-sm">
                Submit transactions via the{" "}
                <Link href="/wallet" className="text-blue-400 hover:underline">eWallet</Link>{" "}
                to generate real fraud scores.
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-xs text-gray-500">{filtered.length} result{filtered.length !== 1 ? "s" : ""}</p>
            {filtered.map(tx => {
              const reasons = parseReasons(tx.reasons);
              return (
                <div
                  key={tx.transaction_id}
                  className="bg-[#111827] border border-gray-800 rounded-xl p-4 hover:border-gray-600 transition"
                >
                  <div className="flex items-start justify-between gap-3 flex-wrap">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <DecisionBadge decision={tx.decision} />
                        <span className="text-gray-400 text-xs font-mono">{tx.user_id}</span>
                        <span className="text-gray-600 text-xs">·</span>
                        <span className="text-white font-semibold">{display(tx.amount, codeFromLocation(tx.location))}</span>
                        <span className="text-gray-500 text-xs capitalize">{tx.transaction_type}</span>
                      </div>
                      <p className="text-xs font-mono text-gray-500">{tx.transaction_id}</p>
                    </div>
                    <div className="flex items-center gap-4">
                      <RiskBar score={tx.risk_score} />
                      <span className="text-xs text-gray-500 whitespace-nowrap">
                        {new Date(tx.timestamp).toLocaleString()}
                      </span>
                    </div>
                  </div>

                  {/* Reasons */}
                  <div className="mt-3 space-y-1">
                    {reasons.slice(0, 3).map((r, i) => (
                      <div key={i} className="flex items-start gap-2 text-xs text-gray-400">
                        <span className="text-gray-600 mt-0.5">•</span>
                        <span>{r}</span>
                      </div>
                    ))}
                  </div>

                  <div className="mt-3 pt-3 border-t border-gray-800 flex items-center justify-between">
                    <div className="flex gap-3 text-xs text-gray-500">
                      {tx.location && <span>{countryFlag(tx.location)} {tx.location}</span>}
                      {tx.device_type && <span>📱 {tx.device_type}</span>}
                      <span>⚡ {tx.confidence ? (tx.confidence * 100).toFixed(0) + "% confidence" : "—"}</span>
                    </div>
                    <Link
                      href={`/case/${tx.transaction_id}`}
                      className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 transition"
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
