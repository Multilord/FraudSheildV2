"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft, Shield, AlertTriangle, XCircle, CheckCircle,
  Cpu, MapPin, Smartphone, ChevronDown, ChevronUp,
  TrendingUp, TrendingDown, Info, Clock, Fingerprint,
} from "lucide-react";
import { api, StoredTransaction, parseReasons } from "@/lib/api";
import { useCurrency, codeFromLocation } from "../../CurrencyContext";

// ─── Constants ────────────────────────────────────────────────────────────────
const FLAG_MAP: Record<string, string> = {
  BN: "🇧🇳", KH: "🇰🇭", ID: "🇮🇩", LA: "🇱🇦", MY: "🇲🇾",
  MM: "🇲🇲", PH: "🇵🇭", SG: "🇸🇬", TH: "🇹🇭", TL: "🇹🇱", VN: "🇻🇳",
};
function countryFlag(location: string) {
  return FLAG_MAP[location?.split(", ").pop()?.toUpperCase() ?? ""] ?? "🌏";
}

// Keys match contribution_breakdown from backend (points added to final score).
// Legacy raw-probability keys (xgboost, lightgbm, etc.) may appear in older
// stored transactions — they're rendered with a fallback label.
const MODEL_META: Record<string, { label: string; description: string }> = {
  ml_ensemble:     { label: "ML Ensemble",       description: "Supervised meta-learner (XGBoost + LightGBM + IF + LOF)" },
  anomaly:         { label: "Anomaly Detection", description: "Isolation Forest + LOF mean — unsupervised outlier signals" },
  behavioral:      { label: "Behavioral",        description: "Velocity, novelty, z-score context — bounded to prevent override" },
  escalation:      { label: "Risk Escalation",   description: "Added only when multiple independent signals align" },
  ensemble:        { label: "Final Score",       description: "Composed from ML + anomaly + behavioral + escalation" },
  // Legacy keys (older stored transactions)
  xgboost:         { label: "XGBoost (raw)",     description: "Supervised gradient boosting — raw fraud probability" },
  lightgbm:        { label: "LightGBM (raw)",    description: "Fast gradient boosting — raw fraud probability" },
  isolation_forest:{ label: "Isolation Forest (raw)", description: "Unsupervised anomaly detector — raw probability" },
  lof:             { label: "LOF (raw)",          description: "Local outlier factor — raw probability" },
};

// ─── Sub-components ───────────────────────────────────────────────────────────
function SectionCard({ title, icon, children }: {
  title: string; icon?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <div className="card overflow-hidden">
      <div className="px-5 py-3.5 border-b border-white/[0.06] flex items-center gap-2">
        {icon && <span className="text-white/30">{icon}</span>}
        <p className="section-label">{title}</p>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

function Field({ label, value, mono = false }: {
  label: string; value: string | number | undefined; mono?: boolean;
}) {
  return (
    <div>
      <p className="section-label mb-1">{label}</p>
      <p className={`text-sm text-white/80 font-medium ${mono ? "font-mono text-xs" : ""}`}>
        {value ?? "—"}
      </p>
    </div>
  );
}

function AnimatedBar({ value, delay = 0 }: { value: number; delay?: number }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(value), 150 + delay);
    return () => clearTimeout(t);
  }, [value, delay]);

  const color = value >= 70 ? "#FF453A" : value >= 40 ? "#FF9F0A" : "#30D158";
  return (
    <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
      <div
        className="h-full rounded-full"
        style={{
          width: `${width}%`,
          background: color,
          boxShadow: `0 0 8px ${color}40`,
          transition: `width 0.9s cubic-bezier(0.4, 0, 0.2, 1) ${delay}ms`,
        }}
      />
    </div>
  );
}

// ─── Risk Meter ───────────────────────────────────────────────────────────────
function RiskMeter({ score }: { score: number }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => { const t = setTimeout(() => setMounted(true), 100); return () => clearTimeout(t); }, []);

  const color = score >= 70 ? "#FF453A" : score >= 40 ? "#FF9F0A" : "#30D158";
  const level = score >= 85 ? "Critical" : score >= 70 ? "High" : score >= 40 ? "Medium" : "Low";

  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-2">
        <span className="text-4xl font-bold tabular-nums leading-none" style={{ color }}>{score}</span>
        <span className="text-white/25">/100</span>
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase tracking-widest ml-1"
          style={{ background: `${color}18`, color, border: `1px solid ${color}30` }}>
          {level}
        </span>
      </div>
      <div className="relative h-2 bg-white/[0.06] rounded-full overflow-hidden">
        <div className="risk-gradient absolute inset-0 rounded-full" />
        <div
          className="absolute inset-y-0 right-0"
          style={{
            width: `${100 - (mounted ? score : 0)}%`,
            background: "rgba(9,9,11,0.92)",
            transition: mounted ? "width 1s cubic-bezier(0.4, 0, 0.2, 1)" : "none",
          }}
        />
      </div>
      <div className="flex justify-between section-label px-0.5">
        <span>Safe</span><span>Medium</span><span>Critical</span>
      </div>
    </div>
  );
}

// ─── Insight sentence — prefers server explanation, client fallback ───────────
function getInsight(decision: string, score: number, reasons: string[], explanation?: string): string {
  if (explanation) return explanation;
  if (decision === "APPROVE") return score < 15
    ? "Transaction is safe — consistent with this account's behavioral profile."
    : "Transaction approved — risk indicators within acceptable thresholds.";
  const top = reasons[0] ? ` Primary signal: ${reasons[0].toLowerCase().replace(/\.$/, "")}.` : "";
  if (decision === "FLAG") return `Flagged for review (risk ${score}/100).${top}`;
  return `Transaction blocked (risk ${score}/100).${top}`;
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function CasePage() {
  const { display } = useCurrency();
  const params = useParams();
  const id = params.id as string;

  const [tx, setTx]             = useState<StoredTransaction | null>(null);
  const [loading, setLoading]   = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [showRaw, setShowRaw]   = useState(false);

  useEffect(() => {
    if (!id) return;
    api.getCase(id)
      .then(setTx)
      .catch(() => setNotFound(true))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return (
    <div className="min-h-screen bg-[#09090b] flex items-center justify-center">
      <div className="flex items-center gap-2.5 text-white/30 text-sm">
        <Cpu size={16} className="animate-pulse" />Loading case…
      </div>
    </div>
  );

  if (notFound || !tx) return (
    <div className="min-h-screen bg-[#09090b] flex flex-col items-center justify-center gap-5">
      <div className="w-16 h-16 rounded-3xl card flex items-center justify-center">
        <Shield size={28} className="text-white/20" />
      </div>
      <div className="text-center">
        <p className="text-white/60 font-medium">Case not found</p>
        <p className="text-white/25 text-xs font-mono mt-1">{id}</p>
      </div>
      <Link href="/triage" className="text-[#0A84FF] hover:opacity-75 text-sm flex items-center gap-1.5 transition-opacity">
        <ArrowLeft size={14} />Back to Triage
      </Link>
    </div>
  );

  const reasons = parseReasons(tx.reasons);
  const isApprove = tx.decision === "APPROVE";
  const isFlag    = tx.decision === "FLAG";
  const isBlock   = tx.decision === "BLOCK";

  const accentColor = isApprove ? "#30D158" : isFlag ? "#FF9F0A" : "#FF453A";
  const heroGrad    = isApprove
    ? "from-[#0a2318] to-[#09090b]"
    : isFlag
    ? "from-[#241c08] to-[#09090b]"
    : "from-[#24090a] to-[#09090b]";

  const DecIcon = isApprove ? CheckCircle : isFlag ? AlertTriangle : XCircle;
  const heroTitle = isApprove ? "Transaction Approved" : isFlag ? "Flagged for Review" : "Transaction Blocked";

  // Parse stored features JSON first — needed for explanation and model breakdown
  let modelBreakdown: Record<string, number> | null = null;
  let modelRawProbs: Record<string, number> | null = null;
  let featuresObj: Record<string, unknown> | null = null;
  let featuresExplanation: string | undefined;
  try {
    featuresObj = typeof tx.features === "string"
      ? JSON.parse(tx.features) as Record<string, unknown>
      : tx.features as unknown as Record<string, unknown>;
    if (featuresObj?.model_breakdown) modelBreakdown = featuresObj.model_breakdown as Record<string, number>;
    if (featuresObj?.model_raw_probabilities) modelRawProbs = featuresObj.model_raw_probabilities as Record<string, number>;
    if (featuresObj?.explanation) featuresExplanation = featuresObj.explanation as string;
  } catch { /* ignore */ }

  // Pull explanation from features JSON (stored at scoring time) or top-level field
  const storedExplanation = featuresExplanation
    || ((tx as unknown as Record<string, unknown>).explanation as string | undefined);
  const insight = getInsight(tx.decision, tx.risk_score, reasons, storedExplanation);

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

  // Prefer new contribution keys; fall back to legacy raw-prob keys for old records
  const modelOrder = ["ml_ensemble", "anomaly", "behavioral", "escalation",
                      "xgboost", "lightgbm", "isolation_forest", "lof", "ensemble"];

  return (
    <div className="min-h-screen bg-[#09090b] pb-12">
      <div className="max-w-4xl mx-auto px-5 py-7 space-y-5">

        {/* Back */}
        <Link href="/triage" className="inline-flex items-center gap-1.5 text-white/35 hover:text-white/70 text-sm transition-colors">
          <ArrowLeft size={14} />Triage
        </Link>

        {/* ══ SECTION 1: Decision Hero ══ */}
        <div className={`rounded-2xl bg-gradient-to-b ${heroGrad} p-6 border`}
          style={{ borderColor: `${accentColor}20` }}>

          <div className="flex items-start justify-between flex-wrap gap-4 mb-4">
            <div className="flex items-center gap-3.5">
              <div className="w-12 h-12 rounded-2xl flex items-center justify-center shrink-0"
                style={{ background: `${accentColor}18`, border: `1px solid ${accentColor}28` }}>
                <DecIcon size={24} style={{ color: accentColor }} />
              </div>
              <div>
                <p className="text-2xl font-bold text-white leading-tight">{heroTitle}</p>
                <p className="text-[11px] uppercase tracking-widest font-medium mt-0.5"
                  style={{ color: `${accentColor}80` }}>
                  {tx.decision} · {tx.transaction_type}
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-white tabular-nums">
                {display(tx.amount, codeFromLocation(tx.location))}
              </p>
              <p className="text-white/30 text-xs mt-0.5 tabular-nums">
                {new Date(tx.timestamp).toLocaleString()}
              </p>
            </div>
          </div>

          {/* Insight sentence */}
          <div className="flex items-start gap-2.5 px-4 py-3 rounded-xl mb-5"
            style={{ background: `${accentColor}08`, border: `1px solid ${accentColor}15` }}>
            <Info size={13} className="shrink-0 mt-0.5" style={{ color: `${accentColor}80` }} />
            <p className="text-sm text-white/60 leading-relaxed">{insight}</p>
          </div>

          {/* Risk meter */}
          <RiskMeter score={tx.risk_score} />

          {/* 3 quick stats */}
          <div className="grid grid-cols-3 gap-0 mt-5 pt-5 border-t border-white/[0.06]">
            <div className="text-center">
              <p className="section-label mb-1.5">Confidence</p>
              <p className="text-xl font-bold text-white tabular-nums">
                {tx.confidence ? (tx.confidence * 100).toFixed(0) + "%" : "—"}
              </p>
            </div>
            <div className="text-center border-x border-white/[0.06]">
              <p className="section-label mb-1.5">Latency</p>
              <p className="text-xl font-bold text-white tabular-nums">
                {tx.latency_ms ? tx.latency_ms.toFixed(0) + "ms" : "—"}
              </p>
            </div>
            <div className="text-center">
              <p className="section-label mb-1.5">User</p>
              <p className="text-sm font-mono text-white/60 truncate px-2">{tx.user_id}</p>
            </div>
          </div>
        </div>

        {/* ══ SECTION 2: Why This Decision (2-col) ══ */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

          {/* A: Risk Signals */}
          <SectionCard title="Risk Signals" icon={<AlertTriangle size={12} />}>
            {reasons.length === 0 ? (
              <div className="flex items-center gap-2 text-white/30 text-sm">
                <CheckCircle size={14} style={{ color: "#30D158" }} />
                No risk signals detected.
              </div>
            ) : (
              <div className="space-y-2.5">
                {reasons.map((r, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-xl"
                    style={{ background: `${accentColor}07`, border: `1px solid ${accentColor}12` }}>
                    <div className="w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5"
                      style={{ background: `${accentColor}18` }}>
                      <DecIcon size={11} style={{ color: accentColor }} />
                    </div>
                    <span className="text-sm text-white/70 leading-snug">{r}</span>
                  </div>
                ))}
              </div>
            )}
          </SectionCard>

          {/* B: Model Contributions */}
          {modelBreakdown ? (
            <SectionCard title="Contribution to Final Risk Score" icon={<Cpu size={12} />}>
              <div className="space-y-4">
                {modelOrder.map((key, i) => {
                  const val = modelBreakdown![key];
                  if (val == null) return null;
                  const meta     = MODEL_META[key] ?? { label: key, description: "" };
                  const barColor = val >= 70 ? "#FF453A" : val >= 40 ? "#FF9F0A" : "#30D158";
                  const isEnsemble = key === "ensemble";

                  // Raw probability to show beside contribution bar
                  let rawVal: number | undefined;
                  if (modelRawProbs) {
                    if (key === "ml_ensemble")     rawVal = modelRawProbs.ml_ensemble;
                    else if (key === "anomaly") {
                      const ifv  = modelRawProbs.isolation_forest;
                      const lofv = modelRawProbs.lof;
                      if (ifv != null && lofv != null) rawVal = Math.round((ifv + lofv) / 2);
                      else rawVal = ifv ?? lofv;
                    }
                    else if (key === "behavioral") rawVal = modelRawProbs.behavioral;
                  }

                  return (
                    <div key={key} className={isEnsemble ? "pt-3 border-t border-white/[0.06] mt-2" : ""}>
                      <div className="flex items-center justify-between mb-1.5">
                        <div>
                          <span className={`text-xs font-medium ${isEnsemble ? "text-white/80" : "text-white/55"}`}>
                            {meta.label}
                          </span>
                          {meta.description && (
                            <p className="text-[10px] text-white/20 mt-0.5">{meta.description}</p>
                          )}
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          {rawVal != null && (
                            <span className="text-[10px] text-white/20 font-mono tabular-nums">
                              raw {rawVal.toFixed(0)}
                            </span>
                          )}
                          <span className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: barColor }}>
                            {val >= 70 ? "HIGH" : val >= 40 ? "MED" : "LOW"}
                          </span>
                          <span className="text-sm font-bold font-mono tabular-nums" style={{ color: barColor }}>
                            {val.toFixed(1)}
                          </span>
                        </div>
                      </div>
                      {!isEnsemble && <AnimatedBar value={val} delay={i * 80} />}
                    </div>
                  );
                })}
              </div>
              <p className="section-label mt-4 flex items-center gap-1.5">
                <Cpu size={9} />Points contributed to final risk score · raw = actual model output
              </p>
            </SectionCard>
          ) : (
            <SectionCard title="Transaction Context" icon={<Info size={12} />}>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Type" value={tx.transaction_type} />
                <Field label="Amount" value={display(tx.amount, codeFromLocation(tx.location))} />
                <Field label="User" value={tx.user_id} mono />
                <Field label="Merchant" value={tx.merchant || "—"} />
              </div>
            </SectionCard>
          )}
        </div>

        {/* ══ SECTION 3: XAI Feature Attribution ══ */}
        {xaiFeatures && xaiFeatures.length > 0 && (
          <SectionCard title="XAI Feature Attribution (SHAP)" icon={<TrendingUp size={12} />}>
            <div className="space-y-4">
              {xaiFeatures.map((f, i) => {
                const isRisk = f.direction === "increases_risk";
                const fColor = isRisk ? "#FF453A" : "#30D158";
                const pct = Math.min(100, Math.abs(f.contribution) * 450);
                return (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        <div className="w-5 h-5 rounded-md flex items-center justify-center"
                          style={{ background: `${fColor}15` }}>
                          {isRisk
                            ? <TrendingUp size={11} style={{ color: fColor }} />
                            : <TrendingDown size={11} style={{ color: fColor }} />}
                        </div>
                        <span className="text-sm text-white/65">{f.label}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-[11px] text-white/30">
                          {isRisk ? "increases risk" : "reduces risk"}
                        </span>
                        <span className="text-sm font-mono font-bold tabular-nums" style={{ color: fColor }}>
                          {isRisk ? "+" : "−"}{Math.abs(f.contribution).toFixed(4)}
                        </span>
                      </div>
                    </div>
                    <div className="h-1 bg-white/[0.06] rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${pct}%`, background: fColor }} />
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-4 pt-4 border-t border-white/[0.06]">
              <p className="section-label flex items-center gap-1.5">
                <Cpu size={9} />SHAP log-odds contributions from XGBoost · positive = increases fraud risk
              </p>
            </div>
          </SectionCard>
        )}

        {/* ══ SECTION 4: Transaction Context ══ */}
        <SectionCard title="Transaction Context" icon={<Fingerprint size={12} />}>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <div className="flex items-start gap-2.5">
              <Smartphone size={13} className="text-white/25 mt-0.5 shrink-0" />
              <Field label="Device Type" value={tx.device_type} />
            </div>
            <div className="flex items-start gap-2.5">
              <MapPin size={13} className="text-white/25 mt-0.5 shrink-0" />
              <Field label="Location"
                value={tx.location ? `${countryFlag(tx.location)} ${tx.location}` : "—"} />
            </div>
            <Field label="IP Address" value={tx.ip_address} mono />
            <Field label="Merchant Category" value={tx.merchant_category || "—"} />
          </div>

          {tx.merchant && (
            <div className="mt-4 pt-4 border-t border-white/[0.06]">
              <Field label="Merchant" value={tx.merchant} />
            </div>
          )}
        </SectionCard>

        {/* ══ SECTION 5: Raw Payload (collapsible) ══ */}
        {rawPayload && (
          <div className="card overflow-hidden">
            <button
              onClick={() => setShowRaw(!showRaw)}
              className="w-full flex items-center justify-between px-5 py-4 hover:bg-white/[0.03] transition-colors"
            >
              <p className="section-label">Raw Transaction Payload</p>
              {showRaw
                ? <ChevronUp size={14} className="text-white/25" />
                : <ChevronDown size={14} className="text-white/25" />}
            </button>
            {showRaw && (
              <div className="px-5 pb-5 animate-fade-in">
                <pre className="rounded-xl p-4 text-xs text-[#30D158] overflow-x-auto leading-relaxed"
                  style={{ background: "rgba(0,0,0,0.55)", border: "1px solid rgba(255,255,255,0.06)" }}>
                  {JSON.stringify(rawPayload, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Footer ID */}
        <div className="text-center pb-4">
          <p className="section-label">{tx.transaction_id}</p>
        </div>

        <Link href="/triage" className="inline-flex items-center gap-1.5 text-white/35 hover:text-white/70 text-sm transition-colors">
          <ArrowLeft size={14} />Back to Triage
        </Link>
      </div>
    </div>
  );
}
