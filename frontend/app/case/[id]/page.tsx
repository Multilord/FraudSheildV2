"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft, Shield, AlertTriangle, XCircle, CheckCircle,
  Cpu, MapPin, Smartphone, ChevronDown, ChevronUp, TrendingUp,
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
    APPROVE: { icon: <CheckCircle size={15} />, cls: "badge-approve" },
    FLAG:    { icon: <AlertTriangle size={15} />, cls: "badge-flag" },
    BLOCK:   { icon: <XCircle size={15} />, cls: "badge-block" },
  }[decision];
  return (
    <span className={`flex items-center gap-1.5 text-sm font-semibold px-4 py-1.5 rounded-full ${cfg.cls}`}>
      {cfg.icon} {decision}
    </span>
  );
}

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card overflow-hidden">
      <div className="px-5 py-3.5 border-b border-white/[0.06]">
        <p className="text-xs font-semibold text-white/50 uppercase tracking-widest">{title}</p>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string | number | undefined }) {
  return (
    <div>
      <p className="text-[11px] text-white/30 uppercase tracking-widest mb-1">{label}</p>
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
    <div className="min-h-screen bg-[#09090b] flex items-center justify-center">
      <div className="text-white/30 text-sm">Loading case…</div>
    </div>
  );

  if (notFound || !tx) return (
    <div className="min-h-screen bg-[#09090b] flex flex-col items-center justify-center gap-5">
      <div className="w-16 h-16 rounded-3xl bg-white/[0.04] border border-white/[0.07] flex items-center justify-center">
        <Shield size={30} className="text-white/20" />
      </div>
      <div className="text-center space-y-1">
        <p className="text-white/60 font-medium">Case not found</p>
        <p className="text-white/25 text-xs font-mono">{id}</p>
      </div>
      <Link href="/triage" className="text-[#0A84FF] hover:text-[#0A84FF]/80 text-sm flex items-center gap-1 transition-colors">
        <ArrowLeft size={14} /> Back to Triage
      </Link>
    </div>
  );

  const reasons = parseReasons(tx.reasons);
  const riskColor = tx.risk_score >= 70 ? "#FF453A" : tx.risk_score >= 40 ? "#FF9F0A" : "#30D158";

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
    <div className="min-h-screen bg-[#09090b]">
      <div className="max-w-4xl mx-auto px-5 py-8 space-y-5">

        {/* Back */}
        <Link href="/triage" className="inline-flex items-center gap-1.5 text-white/40 hover:text-white/80 text-sm transition-colors">
          <ArrowLeft size={14} /> Triage
        </Link>

        {/* ── Hero card ── */}
        <div className="card p-6">
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div>
              <p className="text-[11px] font-mono text-white/25 mb-2">{tx.transaction_id}</p>
              <h1 className="text-3xl font-bold tracking-tight text-white">{display(tx.amount, codeFromLocation(tx.location))}</h1>
              <p className="text-white/40 text-sm mt-1.5 capitalize">{tx.transaction_type} · {tx.user_id}</p>
              <p className="text-white/25 text-xs mt-0.5 tabular-nums">{new Date(tx.timestamp).toLocaleString()}</p>
            </div>
            <DecisionBadge decision={tx.decision} />
          </div>

          {/* Risk gauge */}
          <div className="mt-6">
            <div className="flex justify-between items-baseline mb-2.5">
              <span className="text-xs text-white/40 uppercase tracking-widest font-medium">Risk Score</span>
              <span className="text-2xl font-bold tabular-nums" style={{ color: riskColor }}>
                {tx.risk_score}
                <span className="text-sm text-white/30 font-normal"> / 100</span>
              </span>
            </div>
            <div className="w-full bg-white/[0.07] rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all"
                style={{ width: `${tx.risk_score}%`, background: riskColor, boxShadow: `0 0 12px ${riskColor}60` }}
              />
            </div>
          </div>

          {/* Quick stats */}
          <div className="grid grid-cols-3 gap-0 mt-6 pt-5 border-t border-white/[0.06]">
            <div className="text-center">
              <p className="text-[11px] text-white/30 uppercase tracking-widest mb-1.5">Confidence</p>
              <p className="text-xl font-bold text-white tabular-nums">
                {tx.confidence ? (tx.confidence * 100).toFixed(0) + "%" : "—"}
              </p>
            </div>
            <div className="text-center border-x border-white/[0.06]">
              <p className="text-[11px] text-white/30 uppercase tracking-widest mb-1.5">Latency</p>
              <p className="text-xl font-bold text-white tabular-nums">
                {tx.latency_ms ? tx.latency_ms.toFixed(1) + "ms" : "—"}
              </p>
            </div>
            <div className="text-center">
              <p className="text-[11px] text-white/30 uppercase tracking-widest mb-1.5">Engine</p>
              <p className="text-xl font-bold text-white">
                {modelBreakdown
                  ? `${Object.keys(modelBreakdown).filter(k => k !== "ensemble" && k !== "behavioral").length}-Model`
                  : "XGBoost"}
              </p>
            </div>
          </div>
        </div>

        {/* ── 2-col grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Risk Signals */}
          <SectionCard title="Risk Signals">
            <div className="space-y-2.5">
              {reasons.length === 0 ? (
                <p className="text-white/30 text-sm">No signals recorded.</p>
              ) : reasons.map((r, i) => {
                const iconColor = tx.decision === "BLOCK" ? "#FF453A" : tx.decision === "FLAG" ? "#FF9F0A" : "#30D158";
                return (
                  <div key={i} className="flex gap-3 items-start p-3 rounded-xl" style={{ background: "rgba(255,255,255,0.03)" }}>
                    <span className="mt-0.5 shrink-0" style={{ color: iconColor }}>
                      {tx.decision === "BLOCK" ? <XCircle size={13} /> : tx.decision === "FLAG" ? <AlertTriangle size={13} /> : <CheckCircle size={13} />}
                    </span>
                    <span className="text-sm text-white/75 leading-relaxed">{r}</span>
                  </div>
                );
              })}
            </div>
          </SectionCard>

          {/* Model Breakdown or TX Context */}
          {modelBreakdown ? (
            <SectionCard title="Model Breakdown">
              <div className="space-y-4">
                {Object.entries(modelBreakdown).map(([model, score]) => {
                  const s = Number(score);
                  const barColor = s >= 70 ? "#FF453A" : s >= 40 ? "#FF9F0A" : "#30D158";
                  return (
                    <div key={model}>
                      <div className="flex justify-between mb-1.5">
                        <span className="text-xs text-white/50 capitalize">{model.replace(/_/g, " ")}</span>
                        <span className="text-xs font-mono font-bold tabular-nums" style={{ color: barColor }}>
                          {typeof score === "number" ? score.toFixed(1) : score}
                        </span>
                      </div>
                      <div className="bg-white/[0.07] rounded-full h-1.5">
                        <div
                          className="h-1.5 rounded-full"
                          style={{ width: `${Math.min(100, s)}%`, background: barColor }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="text-[11px] text-white/25 mt-5 flex items-center gap-1.5">
                <Cpu size={10} /> Scores represent fraud probability × 100
              </p>
            </SectionCard>
          ) : (
            <SectionCard title="Transaction Context">
              <div className="grid grid-cols-2 gap-4">
                <Field label="Type" value={tx.transaction_type} />
                <Field label="Amount" value={display(tx.amount, codeFromLocation(tx.location))} />
                <Field label="User" value={tx.user_id} />
                <Field label="Merchant" value={tx.merchant || "—"} />
              </div>
            </SectionCard>
          )}
        </div>

        {/* ── XAI Feature Attribution ── */}
        {xaiFeatures && xaiFeatures.length > 0 && (
          <SectionCard title="XAI Feature Attribution (SHAP)">
            <div className="space-y-3.5">
              {xaiFeatures.map((f, i) => {
                const isRisk = f.direction === "increases_risk";
                const pct = Math.min(100, Math.abs(f.contribution) * 500);
                const barColor = isRisk ? "#FF453A" : "#30D158";
                return (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        <span style={{ color: barColor }}>
                          <TrendingUp size={12} className={isRisk ? "" : "rotate-180"} />
                        </span>
                        <span className="text-sm text-white/70">{f.label}</span>
                      </div>
                      <span className="text-xs font-mono font-bold tabular-nums" style={{ color: barColor }}>
                        {isRisk ? "+" : ""}{f.contribution.toFixed(4)}
                      </span>
                    </div>
                    <div className="bg-white/[0.07] rounded-full h-1">
                      <div className="h-1 rounded-full" style={{ width: `${pct}%`, background: barColor }} />
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="text-[11px] text-white/25 mt-5 flex items-center gap-1.5">
              <Cpu size={10} /> SHAP log-odds contributions from XGBoost · ▲ increases risk · ▽ reduces risk
            </p>
          </SectionCard>
        )}

        {/* ── Device & Location ── */}
        <SectionCard title="Device & Location">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <div className="flex items-start gap-2.5">
              <Smartphone size={13} className="text-white/25 mt-0.5 shrink-0" />
              <Field label="Device Type" value={tx.device_type} />
            </div>
            <div className="flex items-start gap-2.5">
              <MapPin size={13} className="text-white/25 mt-0.5 shrink-0" />
              <Field label="Location" value={tx.location ? `${countryFlag(tx.location)} ${tx.location}` : "—"} />
            </div>
            <Field label="IP Address" value={tx.ip_address} />
            <Field label="Merchant Category" value={tx.merchant_category || "—"} />
          </div>
        </SectionCard>

        {/* ── Raw payload ── */}
        {rawPayload && (
          <div className="card overflow-hidden">
            <button
              onClick={() => setShowRaw(!showRaw)}
              className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-white/[0.04] transition-colors"
            >
              <span className="text-xs font-semibold text-white/50 uppercase tracking-widest">Raw Transaction Payload</span>
              {showRaw ? <ChevronUp size={15} className="text-white/30" /> : <ChevronDown size={15} className="text-white/30" />}
            </button>
            {showRaw && (
              <div className="px-5 pb-5">
                <pre className="rounded-xl p-4 text-xs text-[#30D158] overflow-x-auto leading-relaxed"
                  style={{ background: "rgba(0,0,0,0.5)", border: "1px solid rgba(255,255,255,0.06)" }}
                >
                  {JSON.stringify(rawPayload, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        <Link href="/triage" className="inline-flex items-center gap-1.5 text-white/40 hover:text-white/80 text-sm transition-colors pb-4">
          <ArrowLeft size={14} /> Back to Triage
        </Link>
      </div>
    </div>
  );
}
