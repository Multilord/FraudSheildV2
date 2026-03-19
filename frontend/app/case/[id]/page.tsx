"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft, Shield, AlertTriangle, XCircle, CheckCircle,
  Cpu, Clock, MapPin, Smartphone, ChevronDown, ChevronUp, TrendingUp,
} from "lucide-react";
import { api, StoredTransaction, parseReasons } from "@/lib/api";
import { useCurrency, codeFromLocation } from "../../CurrencyContext";

const FLAG_MAP: Record<string, string> = {
  BN: "🇧🇳", KH: "🇰🇭", ID: "🇮🇩", LA: "🇱🇦", MY: "🇲🇾",
  MM: "🇲🇲", PH: "🇵🇭", SG: "🇸🇬", TH: "🇹🇭", TL: "🇹🇱", VN: "🇻🇳",
};

function countryFlag(location: string): string {
  const code = location?.split(", ").pop()?.toUpperCase() ?? "";
  return FLAG_MAP[code] ?? "🌏";
}

function DecisionBadge({ decision }: { decision: "APPROVE" | "FLAG" | "BLOCK" }) {
  const cfg = {
    APPROVE: { icon: <CheckCircle size={16} />, cls: "bg-green-900/50 border-green-600 text-green-300" },
    FLAG:    { icon: <AlertTriangle size={16} />, cls: "bg-yellow-900/50 border-yellow-600 text-yellow-300" },
    BLOCK:   { icon: <XCircle size={16} />, cls: "bg-red-900/50 border-red-600 text-red-300" },
  }[decision];
  return (
    <span className={`flex items-center gap-1.5 text-sm font-semibold px-3 py-1.5 rounded-full border ${cfg.cls}`}>
      {cfg.icon} {decision}
    </span>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-[#111827] border border-gray-800 rounded-xl overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-800">
        <p className="text-sm font-semibold text-gray-300">{title}</p>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string | number | undefined }) {
  return (
    <div>
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">{label}</p>
      <p className="text-sm text-white font-medium">{value ?? "—"}</p>
    </div>
  );
}

export default function CasePage() {
  const { display } = useCurrency();
  const params = useParams();
  const id = params.id as string;

  const [tx, setTx]           = useState<StoredTransaction | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    if (!id) return;
    api.getCase(id)
      .then(setTx)
      .catch(() => setNotFound(true))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return (
    <div className="min-h-screen bg-[#0a0e1a] flex items-center justify-center">
      <div className="text-gray-400 text-sm">Loading case…</div>
    </div>
  );

  if (notFound || !tx) return (
    <div className="min-h-screen bg-[#0a0e1a] flex flex-col items-center justify-center gap-4">
      <Shield size={48} className="text-gray-600" />
      <p className="text-gray-400 text-lg font-medium">Case not found</p>
      <p className="text-gray-500 text-sm font-mono">{id}</p>
      <Link href="/triage" className="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1">
        <ArrowLeft size={14} /> Back to Triage
      </Link>
    </div>
  );

  const reasons = parseReasons(tx.reasons);
  const riskColor = tx.risk_score >= 70 ? "text-red-400" : tx.risk_score >= 40 ? "text-yellow-400" : "text-green-400";
  const barColor  = tx.risk_score >= 70 ? "bg-red-500"  : tx.risk_score >= 40 ? "bg-yellow-500"  : "bg-green-500";

  let modelBreakdown: Record<string, number> | null = null;
  let featuresObj: Record<string, unknown> | null = null;
  try {
    featuresObj = typeof tx.features === "string"
      ? JSON.parse(tx.features) as Record<string, unknown>
      : tx.features as unknown as Record<string, unknown>;
    if (featuresObj?.model_breakdown) modelBreakdown = featuresObj.model_breakdown as Record<string, number>;
  } catch { /* ignore */ }

  let rawPayload: Record<string, unknown> | null = null;
  try {
    const raw = (tx as unknown as { raw_payload?: unknown }).raw_payload;
    if (typeof raw === "string") rawPayload = JSON.parse(raw) as Record<string, unknown>;
    else if (raw && typeof raw === "object") rawPayload = raw as Record<string, unknown>;
  } catch { /* ignore */ }

  type XaiFeature = { feature: string; label: string; contribution: number; direction: string };
  let xaiFeatures: XaiFeature[] | null = null;
  try {
    if (featuresObj?.xai_top_features) xaiFeatures = featuresObj.xai_top_features as XaiFeature[];
  } catch { /* ignore */ }

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-white">
      <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">

        {/* Back nav */}
        <Link href="/triage" className="inline-flex items-center gap-1.5 text-gray-400 hover:text-white text-sm transition">
          <ArrowLeft size={14} /> Back to Triage
        </Link>

        {/* Header */}
        <div className="bg-[#111827] border border-gray-800 rounded-xl p-6">
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div>
              <p className="text-xs text-gray-500 font-mono mb-1">{tx.transaction_id}</p>
              <h1 className="text-2xl font-bold text-white">{display(tx.amount, codeFromLocation(tx.location))}</h1>
              <p className="text-gray-400 text-sm mt-1 capitalize">{tx.transaction_type} · {tx.user_id}</p>
              <p className="text-gray-500 text-xs mt-1">{new Date(tx.timestamp).toLocaleString()}</p>
            </div>
            <DecisionBadge decision={tx.decision} />
          </div>

          {/* Risk score bar */}
          <div className="mt-6">
            <div className="flex justify-between mb-2">
              <span className="text-xs text-gray-400">Risk Score</span>
              <span className={`text-sm font-bold ${riskColor}`}>{tx.risk_score} / 100</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div className={`h-3 rounded-full transition-all ${barColor}`} style={{ width: `${tx.risk_score}%` }} />
            </div>
          </div>

          {/* Quick stats */}
          <div className="grid grid-cols-3 gap-4 mt-5 pt-5 border-t border-gray-800">
            <div className="text-center">
              <p className="text-xs text-gray-500 mb-1">Confidence</p>
              <p className="text-lg font-bold text-white">{tx.confidence ? (tx.confidence * 100).toFixed(0) + "%" : "—"}</p>
            </div>
            <div className="text-center border-x border-gray-800">
              <p className="text-xs text-gray-500 mb-1">Latency</p>
              <p className="text-lg font-bold text-white">{tx.latency_ms ? tx.latency_ms.toFixed(1) + "ms" : "—"}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-500 mb-1">Score Engine</p>
              <p className="text-lg font-bold text-white">
                {modelBreakdown
                  ? `${Object.keys(modelBreakdown).filter(k => k !== "ensemble" && k !== "behavioral").length}-Model`
                  : "XGBoost"}
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Risk Signals */}
          <Card title="Risk Signals">
            <div className="space-y-3">
              {reasons.length === 0 ? (
                <p className="text-gray-500 text-sm">No signals recorded.</p>
              ) : reasons.map((r, i) => (
                <div key={i} className="flex gap-3 items-start p-3 bg-gray-900/50 rounded-lg">
                  <span className={`mt-0.5 ${tx.decision === "BLOCK" ? "text-red-400" : tx.decision === "FLAG" ? "text-yellow-400" : "text-green-400"}`}>
                    {tx.decision === "BLOCK" ? <XCircle size={14} /> : tx.decision === "FLAG" ? <AlertTriangle size={14} /> : <CheckCircle size={14} />}
                  </span>
                  <span className="text-sm text-gray-200">{r}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Model Breakdown */}
          {modelBreakdown ? (
            <Card title="Model Breakdown">
              <div className="space-y-4">
                {Object.entries(modelBreakdown).map(([model, score]) => (
                  <div key={model}>
                    <div className="flex justify-between mb-1">
                      <span className="text-xs text-gray-400 capitalize">{model}</span>
                      <span className={`text-xs font-bold font-mono ${Number(score) >= 70 ? "text-red-400" : Number(score) >= 40 ? "text-yellow-400" : "text-green-400"}`}>
                        {typeof score === "number" ? score.toFixed(1) : score}
                      </span>
                    </div>
                    <div className="bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${Number(score) >= 70 ? "bg-red-500" : Number(score) >= 40 ? "bg-yellow-500" : "bg-green-500"}`}
                        style={{ width: `${Math.min(100, Number(score))}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500 mt-4 flex items-center gap-1">
                <Cpu size={10} /> Scores are fraud probability × 100
              </p>
            </Card>
          ) : (
            <Card title="Transaction Context">
              <div className="grid grid-cols-2 gap-4">
                <Field label="Type" value={tx.transaction_type} />
                <Field label="Amount" value={display(tx.amount, codeFromLocation(tx.location))} />
                <Field label="User" value={tx.user_id} />
                <Field label="Merchant" value={tx.merchant || "—"} />
              </div>
            </Card>
          )}
        </div>

        {/* XAI Feature Attribution */}
        {xaiFeatures && xaiFeatures.length > 0 && (
          <Card title="XAI Feature Attribution (SHAP)">
            <div className="space-y-3">
              {xaiFeatures.map((f, i) => {
                const isRisk = f.direction === "increases_risk";
                const pct = Math.min(100, Math.abs(f.contribution) * 500);
                return (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className={isRisk ? "text-red-400" : "text-green-400"}>
                          {isRisk ? <TrendingUp size={12} /> : <TrendingUp size={12} className="rotate-180" />}
                        </span>
                        <span className="text-sm text-gray-200">{f.label}</span>
                      </div>
                      <span className={`text-xs font-mono font-bold ${isRisk ? "text-red-400" : "text-green-400"}`}>
                        {isRisk ? "+" : ""}{f.contribution.toFixed(4)}
                      </span>
                    </div>
                    <div className="bg-gray-700 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${isRisk ? "bg-red-500" : "bg-green-500"}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="text-xs text-gray-500 mt-4 flex items-center gap-1">
              <Cpu size={10} /> SHAP log-odds contributions from XGBoost · ▲ increases risk · ▽ reduces risk
            </p>
          </Card>
        )}

        {/* Device & Location */}
        <Card title="Device & Location">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="flex items-start gap-2">
              <Smartphone size={14} className="text-gray-500 mt-0.5" />
              <Field label="Device Type" value={tx.device_type} />
            </div>
            <div className="flex items-start gap-2">
              <MapPin size={14} className="text-gray-500 mt-0.5" />
              <Field label="Location" value={tx.location ? `${countryFlag(tx.location)} ${tx.location}` : "—"} />
            </div>
            <Field label="IP Address" value={tx.ip_address} />
            <Field label="Merchant Category" value={tx.merchant_category || "—"} />
          </div>
        </Card>

        {/* Raw payload collapsible */}
        {rawPayload && (
          <div className="bg-[#111827] border border-gray-800 rounded-xl overflow-hidden">
            <button
              onClick={() => setShowRaw(!showRaw)}
              className="w-full flex items-center justify-between px-5 py-3 hover:bg-gray-800/50 transition"
            >
              <span className="text-sm font-semibold text-gray-300">Raw Transaction Payload</span>
              {showRaw ? <ChevronUp size={16} className="text-gray-500" /> : <ChevronDown size={16} className="text-gray-500" />}
            </button>
            {showRaw && (
              <div className="px-5 pb-5">
                <pre className="bg-gray-900 rounded-lg p-4 text-xs text-green-300 overflow-x-auto">
                  {JSON.stringify(rawPayload, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        <Link href="/triage" className="inline-flex items-center gap-1.5 text-gray-400 hover:text-white text-sm transition">
          <ArrowLeft size={14} /> Back to Triage
        </Link>
      </div>
    </div>
  );
}
