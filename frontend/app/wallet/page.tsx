"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import {
  Send, CreditCard, ArrowDownLeft, ArrowUpRight, ShoppingBag,
  CheckCircle, AlertTriangle, XCircle, ChevronDown, ChevronUp,
  Loader2, Shield, Clock, Cpu, MapPin, Globe,
  TrendingUp, TrendingDown, ArrowRight, Info,
} from "lucide-react";
import { api, TransactionResult, HealthStatus, XaiFeature } from "@/lib/api";
import { useCurrency } from "../CurrencyContext";

// ─── ASEAN Countries ──────────────────────────────────────────────────────────
const ASEAN_COUNTRIES = [
  { name: "Brunei",      code: "BN", currency: "BND", symbol: "B$", flag: "🇧🇳", cities: ["Bandar Seri Begawan", "Kuala Belait", "Seria", "Tutong", "Bangar", "Muara"] },
  { name: "Cambodia",    code: "KH", currency: "KHR", symbol: "₭",  flag: "🇰🇭", cities: ["Phnom Penh", "Siem Reap", "Battambang", "Sihanoukville", "Kampot", "Kampong Cham", "Kratié", "Pursat", "Prey Veng", "Takeo", "Kampong Thom", "Stung Treng", "Pailin", "Svay Rieng"] },
  { name: "Indonesia",   code: "ID", currency: "IDR", symbol: "Rp", flag: "🇮🇩", cities: ["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar", "Denpasar", "Yogyakarta", "Semarang", "Palembang", "Tangerang", "Depok", "Bekasi", "Bogor", "Pekanbaru", "Bandar Lampung", "Padang", "Malang", "Samarinda", "Balikpapan", "Manado", "Pontianak", "Ambon", "Kupang", "Jayapura", "Batam"] },
  { name: "Laos",        code: "LA", currency: "LAK", symbol: "₭",  flag: "🇱🇦", cities: ["Vientiane", "Luang Prabang", "Pakse", "Savannakhet", "Thakhek", "Phonsavan", "Sam Neua", "Luang Namtha", "Attapeu", "Xam Tai"] },
  { name: "Malaysia",    code: "MY", currency: "MYR", symbol: "RM", flag: "🇲🇾", cities: ["Kuala Lumpur", "Petaling Jaya", "Georgetown", "Johor Bahru", "Kota Kinabalu", "Kuching", "Ipoh", "Shah Alam", "Klang", "Subang Jaya", "Kajang", "Seremban", "Malacca", "Alor Setar", "Kota Bharu", "Kuala Terengganu", "Sandakan", "Tawau", "Miri", "Sibu", "Bintulu"] },
  { name: "Myanmar",     code: "MM", currency: "MMK", symbol: "K",  flag: "🇲🇲", cities: ["Yangon", "Mandalay", "Naypyidaw", "Mawlamyine", "Bago", "Pathein", "Monywa", "Sittwe", "Taunggyi", "Myeik", "Pyay", "Meiktila", "Lashio", "Dawei", "Hpa-An"] },
  { name: "Philippines", code: "PH", currency: "PHP", symbol: "₱",  flag: "🇵🇭", cities: ["Manila", "Quezon City", "Cebu", "Davao", "Makati", "Zamboanga", "Taguig", "Pasig", "Cagayan de Oro", "Parañaque", "Las Piñas", "Antipolo", "Muntinlupa", "Caloocan", "Valenzuela", "Marikina", "Mandaluyong", "Iloilo", "Bacolod", "General Santos", "Butuan", "Iligan", "Tacloban", "Dumaguete", "Baguio", "Olongapo", "Angeles", "Legazpi", "Naga"] },
  { name: "Singapore",   code: "SG", currency: "SGD", symbol: "S$", flag: "🇸🇬", cities: ["Singapore", "Jurong East", "Tampines", "Woodlands", "Ang Mo Kio", "Bedok", "Bishan", "Bukit Merah", "Bukit Panjang", "Choa Chu Kang", "Clementi", "Geylang", "Hougang", "Kallang", "Novena", "Orchard", "Pasir Ris", "Punggol", "Queenstown", "Sembawang", "Sengkang", "Serangoon", "Toa Payoh", "Yishun"] },
  { name: "Thailand",    code: "TH", currency: "THB", symbol: "฿",  flag: "🇹🇭", cities: ["Bangkok", "Chiang Mai", "Phuket", "Pattaya", "Hat Yai", "Chiang Rai", "Nakhon Ratchasima", "Khon Kaen", "Udon Thani", "Nakhon Si Thammarat", "Surat Thani", "Ubon Ratchathani", "Rayong", "Nonthaburi", "Pathum Thani", "Samut Prakan", "Chonburi", "Lampang", "Phitsanulok", "Nakhon Sawan", "Ayutthaya", "Hua Hin"] },
  { name: "Timor-Leste", code: "TL", currency: "USD", symbol: "$",  flag: "🇹🇱", cities: ["Dili", "Baucau", "Maliana", "Suai", "Liquiçá", "Aileu", "Ermera", "Gleno", "Same", "Viqueque", "Lospalos", "Manatuto"] },
  { name: "Vietnam",     code: "VN", currency: "VND", symbol: "₫",  flag: "🇻🇳", cities: ["Ho Chi Minh City", "Hanoi", "Da Nang", "Nha Trang", "Hue", "Can Tho", "Hai Phong", "Bien Hoa", "Vung Tau", "Quy Nhon", "Da Lat", "Buon Ma Thuot", "Thai Nguyen", "Nam Dinh", "Vinh", "Long Xuyen", "Rach Gia", "My Tho", "Phan Thiet", "Ca Mau", "Bac Lieu", "Tra Vinh", "Ben Tre"] },
] as const;

type AseanCountry = (typeof ASEAN_COUNTRIES)[number];
const DEFAULT_COUNTRY = ASEAN_COUNTRIES[6];
const DEFAULT_CITY    = "Manila";

type TxType = "transfer" | "payment" | "cashout" | "topup" | "merchant";
type MerchantCategory = "grocery" | "utility" | "food" | "transport" | "entertainment" | "other";

const TX_META: Record<TxType, { icon: React.ReactNode; label: string; color: string }> = {
  transfer: { icon: <Send size={15} />,          label: "Transfer",  color: "#0A84FF" },
  payment:  { icon: <CreditCard size={15} />,    label: "Pay Bill",  color: "#BF5AF2" },
  cashout:  { icon: <ArrowDownLeft size={15} />, label: "Cash Out",  color: "#FF9F0A" },
  topup:    { icon: <ArrowUpRight size={15} />,  label: "Top Up",    color: "#30D158" },
  merchant: { icon: <ShoppingBag size={15} />,   label: "Purchase",  color: "#FF453A" },
};

function generateDeviceId(): string {
  if (typeof window === "undefined") return "device-server";
  const stored = sessionStorage.getItem("fraudshield_device_id");
  if (stored) return stored;
  const id = "device-" + Math.random().toString(36).substring(2, 11);
  sessionStorage.setItem("fraudshield_device_id", id);
  return id;
}

// ─── Risk Meter ───────────────────────────────────────────────────────────────
function RiskMeter({ score }: { score: number }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => { const t = setTimeout(() => setMounted(true), 80); return () => clearTimeout(t); }, []);

  const color  = score >= 70 ? "#FF453A" : score >= 40 ? "#FF9F0A" : "#30D158";
  const level  = score >= 85 ? "Critical" : score >= 70 ? "High" : score >= 40 ? "Medium" : score >= 15 ? "Low" : "Safe";

  return (
    <div className="space-y-2.5">
      {/* Score + level */}
      <div className="flex items-baseline gap-2.5">
        <span className="text-[52px] font-bold leading-none tabular-nums" style={{ color }}>
          {score}
        </span>
        <div>
          <span className="text-white/25 text-xl">/100</span>
          <div className="mt-0.5">
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase tracking-widest"
              style={{ background: `${color}18`, color, border: `1px solid ${color}30` }}>
              {level}
            </span>
          </div>
        </div>
      </div>

      {/* Gradient progress bar */}
      <div className="relative h-2 rounded-full overflow-hidden bg-white/[0.07]">
        <div className="risk-gradient absolute inset-0 rounded-full" />
        {/* Mask the unfilled portion */}
        <div
          className="absolute inset-y-0 right-0 rounded-r-full transition-none"
          style={{
            width: `${100 - (mounted ? score : 0)}%`,
            background: "rgba(9,9,11,0.92)",
            transition: mounted ? "width 1s cubic-bezier(0.4, 0, 0.2, 1)" : "none",
          }}
        />
        {/* Glow at the tip */}
        {mounted && (
          <div
            className="absolute top-0 bottom-0 w-3"
            style={{
              left: `calc(${score}% - 6px)`,
              background: `radial-gradient(ellipse, ${color}90, transparent)`,
              transition: "left 1s cubic-bezier(0.4, 0, 0.2, 1)",
            }}
          />
        )}
      </div>
      <div className="flex justify-between section-label px-0.5">
        <span>Safe</span><span>Medium</span><span>Critical</span>
      </div>
    </div>
  );
}

// ─── Insight Sentence — uses backend explanation when available ───────────────
function getInsight(result: TransactionResult): string {
  // Use the server-generated explanation if present (grounded in real feature values)
  if (result.explanation) return result.explanation;
  // Client-side fallback
  const { decision, risk_score } = result;
  if (decision === "APPROVE") return risk_score < 15
    ? "Transaction is safe — consistent with this account's behavioral profile."
    : "Low risk — no significant anomalies detected.";
  if (decision === "FLAG") return `Flagged for review (risk ${risk_score}/100) — requires additional verification.`;
  return `Transaction blocked (risk ${risk_score}/100) — high-risk signals detected across multiple models.`;
}

// ─── Model Contribution Bars ──────────────────────────────────────────────────
// Keys match contribution_breakdown from the backend (how many points each
// component added to the 0-100 final score, not raw fraud probabilities).
const MODEL_META: Record<string, { label: string; description: string }> = {
  ml_ensemble: { label: "ML Ensemble",       description: "Supervised meta-learner (XGBoost + LightGBM + IF + LOF)" },
  anomaly:     { label: "Anomaly Detection", description: "Isolation Forest + LOF mean — unsupervised outlier signals" },
  behavioral:  { label: "Behavioral",        description: "Velocity, novelty, z-score context — bounded to prevent override" },
  escalation:  { label: "Risk Escalation",   description: "Added only when multiple independent signals align" },
};

function ModelBar({ name, value, rawValue, delay = 0 }: {
  name: string; value: number; rawValue?: number; delay?: number;
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 200 + delay);
    return () => clearTimeout(t);
  }, [delay]);

  const meta     = MODEL_META[name] ?? { label: name, description: "" };
  const barColor = value >= 70 ? "#FF453A" : value >= 40 ? "#FF9F0A" : "#30D158";
  const levelTxt = value >= 70 ? "HIGH" : value >= 40 ? "MED" : "LOW";
  const pct      = mounted ? value : 0;

  return (
    <div className="group">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <span className="text-xs text-white/60 font-medium">{meta.label}</span>
          <span className="text-[10px] text-white/25 hidden group-hover:block transition-all">
            {meta.description}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {rawValue != null && (
            <span className="text-[10px] text-white/20 tabular-nums font-mono">
              raw {rawValue.toFixed(0)}
            </span>
          )}
          <span className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: barColor }}>
            {levelTxt}
          </span>
          <span className="text-xs font-mono font-bold text-white/70 tabular-nums w-6 text-right">
            {value.toFixed(0)}
          </span>
        </div>
      </div>
      <div className="relative h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{
            width: `${pct}%`,
            background: barColor,
            boxShadow: `0 0 8px ${barColor}50`,
            transition: `width 0.8s cubic-bezier(0.4, 0, 0.2, 1) ${delay}ms`,
          }}
        />
      </div>
    </div>
  );
}

// ─── Decision Result Card ─────────────────────────────────────────────────────
function DecisionCard({
  result, onReset, country,
}: {
  result: TransactionResult;
  onReset: () => void;
  country: AseanCountry;
}) {
  const [showXai, setShowXai] = useState(false);
  const isApprove = result.decision === "APPROVE";
  const isFlag    = result.decision === "FLAG";
  const isBlock   = result.decision === "BLOCK";

  const accentColor = isApprove ? "#30D158" : isFlag ? "#FF9F0A" : "#FF453A";
  const heroGrad    = isApprove
    ? "from-[#0a2318] to-[#09090b]"
    : isFlag
    ? "from-[#241c08] to-[#09090b]"
    : "from-[#24090a] to-[#09090b]";

  const DecIcon = isApprove ? CheckCircle : isFlag ? AlertTriangle : XCircle;
  const heroTitle = isApprove ? "Transaction Approved" : isFlag ? "Flagged for Review" : "Transaction Blocked";
  const insight = getInsight(result);

  // Contribution breakdown keys — how many points each component added to final score
  const modelKeys = ["ml_ensemble", "anomaly", "behavioral", "escalation"] as const;
  // Raw model outputs (for "raw N" secondary text beside each bar)
  const rawProbs = result.model_raw_probabilities ?? {};
  const getRawValue = (key: string): number | undefined => {
    if (key === "ml_ensemble") return rawProbs.ml_ensemble;
    if (key === "anomaly") {
      const ifv = rawProbs.isolation_forest;
      const lofv = rawProbs.lof;
      if (ifv != null && lofv != null) return Math.round((ifv + lofv) / 2);
      return ifv ?? lofv;
    }
    if (key === "behavioral") return rawProbs.behavioral;
    return undefined;
  };
  const hasXai = result.xai_top_features && result.xai_top_features.length > 0;

  return (
    <div className="rounded-2xl overflow-hidden border border-white/[0.08] animate-fade-up shadow-2xl shadow-black/50">

      {/* ══ SECTION 1: Decision Hero ══ */}
      <div className={`bg-gradient-to-b ${heroGrad} p-5 border-b`}
        style={{ borderColor: `${accentColor}20` }}>

        {/* Decision badge + amount */}
        <div className="flex items-start justify-between gap-4 mb-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: `${accentColor}18`, border: `1px solid ${accentColor}30` }}>
              <DecIcon size={20} style={{ color: accentColor }} />
            </div>
            <div>
              <p className="text-lg font-bold text-white leading-tight">{heroTitle}</p>
              <p className="text-[11px] uppercase tracking-widest font-medium mt-0.5"
                style={{ color: `${accentColor}90` }}>
                {result.decision} · {result.transaction_type}
              </p>
            </div>
          </div>
          <div className="text-right shrink-0">
            <p className="text-2xl font-bold text-white tabular-nums">
              {country.symbol}{result.amount.toLocaleString()}
            </p>
            <p className="text-[11px] text-white/30 mt-0.5">{country.currency}</p>
          </div>
        </div>

        {/* Human-readable insight sentence */}
        <div className="flex items-start gap-2.5 px-3 py-2.5 rounded-xl"
          style={{ background: `${accentColor}09`, border: `1px solid ${accentColor}18` }}>
          <Info size={13} className="shrink-0 mt-0.5" style={{ color: `${accentColor}90` }} />
          <p className="text-sm text-white/65 leading-relaxed">{insight}</p>
        </div>
      </div>

      {/* ══ SECTION 2: Risk Score ══ */}
      <div className="p-5 border-b border-white/[0.06]">
        <div className="flex gap-6 items-start">
          <div className="flex-1">
            <p className="section-label mb-3">Risk Assessment</p>
            <RiskMeter score={result.risk_score} />
          </div>
          <div className="shrink-0 space-y-3 pt-0.5">
            <div className="text-right">
              <p className="section-label mb-1">Confidence</p>
              <p className="text-xl font-bold text-white tabular-nums">
                {result.confidence ? (result.confidence * 100).toFixed(0) + "%" : "—"}
              </p>
            </div>
            <div className="text-right">
              <p className="section-label mb-1">Latency</p>
              <p className="text-xl font-bold text-white tabular-nums">
                {result.latency_ms.toFixed(0)}<span className="text-sm text-white/30 font-normal">ms</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ══ SECTION 3: Why This Decision ══ */}
      <div className="p-5 space-y-6">

        {/* A: Key Risk Signals */}
        <div>
          <p className="section-label mb-3">
            {isApprove ? "Approval Factors" : "Risk Signals Detected"}
          </p>
          {result.reasons.length === 0 ? (
            <p className="text-white/30 text-sm">No signals recorded.</p>
          ) : (
            <div className="space-y-2">
              {result.reasons.map((reason, i) => (
                <div key={i}
                  className="flex items-start gap-3 px-3.5 py-2.5 rounded-xl"
                  style={{
                    background: `${accentColor}07`,
                    border: `1px solid ${accentColor}14`,
                  }}>
                  <div className="mt-0.5 shrink-0 w-4 h-4 rounded-full flex items-center justify-center"
                    style={{ background: `${accentColor}20` }}>
                    <DecIcon size={10} style={{ color: accentColor }} />
                  </div>
                  <span className="text-sm text-white/70 leading-snug">{reason}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* B: Model Contributions */}
        <div>
          <p className="section-label mb-3 flex items-center gap-1.5">
            <Cpu size={9} />Contribution to Final Risk Score
            <span className="text-white/15 font-normal normal-case tracking-normal ml-1">
              — points added by each component (sums to score)
            </span>
          </p>
          <div className="space-y-3.5">
            {modelKeys.map((key, i) => {
              const val = result.model_breakdown[key];
              if (val == null) return null;
              return (
                <ModelBar
                  key={key}
                  name={key}
                  value={val}
                  rawValue={getRawValue(key)}
                  delay={i * 90}
                />
              );
            })}
          </div>
          {/* Final risk score summary */}
          <div className="mt-4 pt-4 border-t border-white/[0.06] flex items-center justify-between">
            <div>
              <span className="text-xs text-white/40 font-medium">Final Risk Score</span>
              <p className="text-[10px] text-white/20 mt-0.5">
                {isBlock ? "Multiple signals aligned → blocked"
                  : isFlag ? "Mixed signals → flagged for review"
                  : "Signals within safe range → approved"}
              </p>
            </div>
            <span className="text-base font-bold tabular-nums" style={{
              color: result.risk_score >= 70 ? "#FF453A"
                : result.risk_score >= 40 ? "#FF9F0A" : "#30D158"
            }}>
              {result.risk_score}
            </span>
          </div>
        </div>

        {/* C: XAI Feature Attribution (expandable) */}
        {hasXai && (
          <div>
            <button
              onClick={() => setShowXai(!showXai)}
              className="flex items-center justify-between w-full mb-3"
            >
              <p className="section-label">Feature Attribution (SHAP)</p>
              <span className="text-white/25">
                {showXai ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </span>
            </button>

            {showXai && (
              <div className="space-y-3 animate-fade-in">
                {result.xai_top_features!.slice(0, 6).map((f: XaiFeature, i: number) => {
                  const isRisk = f.direction === "increases_risk";
                  const fColor = isRisk ? "#FF453A" : "#30D158";
                  const pct = Math.min(100, Math.abs(f.contribution) * 450);
                  return (
                    <div key={i}>
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center gap-2">
                          <span style={{ color: fColor }}>
                            {isRisk ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
                          </span>
                          <span className="text-xs text-white/60">{f.label}</span>
                        </div>
                        <span className="text-[11px] font-mono font-bold tabular-nums" style={{ color: fColor }}>
                          {isRisk ? "+" : "−"}{Math.abs(f.contribution).toFixed(4)}
                        </span>
                      </div>
                      <div className="h-0.5 bg-white/[0.06] rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: fColor }} />
                      </div>
                    </div>
                  );
                })}
                <p className="text-[10px] text-white/20 pt-1">
                  SHAP log-odds contributions from XGBoost — positive = increases fraud risk
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ══ Footer: Actions ══ */}
      <div className="px-5 pb-5 pt-1">
        <div className="divider mb-4" />
        <p className="text-[10px] font-mono text-white/20 text-center mb-3">{result.transaction_id}</p>
        <div className="grid grid-cols-2 gap-2.5">
          <Link
            href={`/case/${result.transaction_id}`}
            className="btn-ghost py-2.5 text-sm text-center rounded-xl font-medium flex items-center justify-center gap-1.5"
          >
            Full Case <ArrowRight size={13} />
          </Link>
          <button onClick={onReset} className="btn-primary py-2.5 text-sm rounded-xl">
            New Transaction
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── History Row ──────────────────────────────────────────────────────────────
function HistoryRow({ tx, symbol }: { tx: TransactionResult; symbol: string }) {
  const [open, setOpen] = useState(false);
  const meta  = TX_META[tx.transaction_type as TxType];
  const cls   = tx.decision === "APPROVE" ? "badge-approve" : tx.decision === "FLAG" ? "badge-flag" : "badge-block";
  const color = tx.decision === "APPROVE" ? "#30D158" : tx.decision === "FLAG" ? "#FF9F0A" : "#FF453A";

  return (
    <div className="card-interactive rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left"
      >
        <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
          style={{ background: `${meta?.color ?? "#fff"}15`, color: meta?.color ?? "#fff" }}>
          {meta?.icon ?? <CreditCard size={14} />}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-white/80 tabular-nums">{symbol}{tx.amount.toLocaleString()}</p>
          <p className="text-[11px] text-white/30">{meta?.label ?? tx.transaction_type}</p>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${cls}`}>{tx.decision}</span>
        <div className="w-6 h-6 rounded-md flex items-center justify-center"
          style={{ background: `${color}15` }}>
          <span className="text-[11px] font-bold tabular-nums" style={{ color }}>{tx.risk_score}</span>
        </div>
        {open ? <ChevronUp size={13} className="text-white/20 shrink-0" /> : <ChevronDown size={13} className="text-white/20 shrink-0" />}
      </button>
      {open && (
        <div className="px-4 pb-3.5 pt-0 space-y-1 border-t border-white/[0.06] animate-fade-in">
          <p className="text-[10px] font-mono text-white/20 pt-2.5">{tx.transaction_id}</p>
          {tx.reasons.map((r, i) => (
            <p key={i} className="text-xs text-white/40 flex items-start gap-1.5">
              <span className="mt-0.5 text-white/20">•</span>{r}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function WalletPage() {
  const [userId, setUserId]             = useState("user_demo_001");
  const [balance, setBalance]           = useState(100_000);
  const [txType, setTxType]             = useState<TxType>("payment");
  const [amount, setAmount]             = useState("");
  const [recipient, setRecipient]       = useState("");
  const [merchant, setMerchant]         = useState("");
  const [merchantCat, setMerchantCat]   = useState<MerchantCategory>("grocery");
  const [country, setCountry]           = useState<AseanCountry>(DEFAULT_COUNTRY);
  const [city, setCity]                 = useState(DEFAULT_CITY);
  const [isNewDevice, setIsNewDevice]   = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading]           = useState(false);
  const [result, setResult]             = useState<TransactionResult | null>(null);
  const [error, setError]               = useState<string | null>(null);
  const [history, setHistory]           = useState<TransactionResult[]>([]);
  const [health, setHealth]             = useState<HealthStatus | null>(null);
  const [deviceId]                      = useState<string>(generateDeviceId);
  const { currency: displayCurrency, convert } = useCurrency();

  const displayBalance = convert(balance, country.currency);
  const showingConverted = displayCurrency.code !== country.currency;

  useEffect(() => { api.getHealth().then(setHealth).catch(() => null); }, []);

  const handleCountryChange = (code: string) => {
    const c = ASEAN_COUNTRIES.find(c => c.code === code);
    if (!c) return;
    setCountry(c);
    setCity(c.cities[0]);
  };

  const handleSubmit = useCallback(async () => {
    setError(null);
    const amt = parseFloat(amount);
    if (!amt || amt <= 0) { setError("Enter a valid amount."); return; }
    setLoading(true);
    try {
      const res = await api.submitTransaction({
        user_id: userId, amount: amt, transaction_type: txType,
        recipient_id: txType === "transfer" ? recipient || undefined : undefined,
        merchant: ["merchant", "payment"].includes(txType) ? merchant || undefined : undefined,
        merchant_category: ["merchant", "payment"].includes(txType) ? merchantCat : undefined,
        device_type: "mobile", device_id: deviceId, ip_address: "203.0.113.42",
        location: `${city}, ${country.code}`, is_new_device: isNewDevice,
      });
      setResult(res);
      setHistory(prev => [res, ...prev].slice(0, 10));
      if (res.decision === "APPROVE") {
        if (["transfer", "payment", "cashout", "merchant"].includes(txType))
          setBalance(prev => Math.max(0, prev - amt));
        else if (txType === "topup")
          setBalance(prev => prev + amt);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }, [userId, amount, txType, recipient, merchant, merchantCat, city, country, isNewDevice, deviceId]);

  const reset = () => { setResult(null); setAmount(""); setError(null); };
  const modelOffline = health && !health.model_loaded;

  return (
    <div className="min-h-screen bg-[#09090b] pb-20">
      <div className="max-w-md mx-auto px-4 pt-6 space-y-4">

        {/* Offline warning */}
        {modelOffline && (
          <div className="card p-4 border-[#FF9F0A]/20 bg-[#FF9F0A]/[0.05] flex items-start gap-3">
            <AlertTriangle size={15} className="text-[#FF9F0A] shrink-0 mt-0.5" />
            <p className="text-sm text-[#FF9F0A]/80">
              Fraud model not loaded — train the model first to enable live scoring.
            </p>
          </div>
        )}

        {/* ── Wallet Card ── */}
        <div className="relative rounded-2xl overflow-hidden shadow-2xl"
          style={{
            background: "linear-gradient(145deg, #111827 0%, #0d1520 50%, #0a1128 100%)",
            border: "1px solid rgba(255,255,255,0.09)",
          }}>
          {/* Background glow */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            <div className="absolute -top-12 -right-12 w-40 h-40 rounded-full opacity-20"
              style={{ background: "radial-gradient(circle, #0A84FF, transparent 70%)" }} />
            <div className="absolute -bottom-8 -left-8 w-32 h-32 rounded-full opacity-10"
              style={{ background: "radial-gradient(circle, #BF5AF2, transparent 70%)" }} />
          </div>

          <div className="relative p-6">
            <div className="flex items-start justify-between mb-7">
              <div>
                <p className="section-label mb-1">eWallet Balance</p>
                <p className="text-[11px] font-mono text-white/30">{userId}</p>
              </div>
              <div className="flex items-center gap-2.5">
                <span className="text-2xl">{country.flag}</span>
                <div className="w-8 h-8 rounded-xl flex items-center justify-center"
                  style={{ background: "rgba(10,132,255,0.15)", border: "1px solid rgba(10,132,255,0.25)" }}>
                  <Shield size={14} style={{ color: "#0A84FF" }} />
                </div>
              </div>
            </div>

            <div>
              <p className="text-[40px] font-bold text-white leading-none tabular-nums">
                {displayCurrency.symbol}{Math.round(displayBalance).toLocaleString()}
              </p>
              <div className="flex items-baseline gap-2 mt-1.5">
                <p className="text-white/40 text-sm">Available · {displayCurrency.code}</p>
                {showingConverted && (
                  <p className="text-white/20 text-xs">
                    ({country.symbol}{balance.toLocaleString()})
                  </p>
                )}
              </div>
            </div>

            {history[0] && (
              <div className="mt-5 pt-4 border-t border-white/[0.07]">
                <p className="text-[11px] text-white/25">
                  Last tx · {country.symbol}{history[0].amount.toLocaleString()}
                  {" "}— {history[0].transaction_type}
                  {" "}— <span className={
                    history[0].decision === "APPROVE" ? "text-[#30D158]"
                    : history[0].decision === "FLAG" ? "text-[#FF9F0A]"
                    : "text-[#FF453A]"
                  }>{history[0].decision}</span>
                </p>
              </div>
            )}
          </div>
        </div>

        {/* User ID */}
        <div>
          <label className="section-label mb-2 block">User ID</label>
          <input
            className="input-apple w-full px-4 py-2.5 text-sm"
            value={userId}
            onChange={e => setUserId(e.target.value)}
            placeholder="user_demo_001"
          />
        </div>

        {/* ── Transaction Form or Result ── */}
        {result ? (
          <DecisionCard result={result} onReset={reset} country={country} />
        ) : (
          <div className="card p-5 space-y-5">

            {/* TX type selector */}
            <div>
              <p className="section-label mb-3">Transaction Type</p>
              <div className="grid grid-cols-5 gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.06]">
                {(Object.keys(TX_META) as TxType[]).map(t => {
                  const m = TX_META[t];
                  const active = txType === t;
                  return (
                    <button
                      key={t}
                      onClick={() => setTxType(t)}
                      className="flex flex-col items-center gap-1.5 py-2.5 px-1 rounded-lg text-[10px] font-semibold transition-all"
                      style={active ? {
                        background: `${m.color}20`,
                        color: m.color,
                        border: `1px solid ${m.color}35`,
                      } : {
                        color: "rgba(255,255,255,0.35)",
                      }}
                    >
                      {m.icon}{m.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Amount input */}
            <div>
              <label className="section-label mb-2 block">Amount ({country.currency})</label>
              <div className="relative">
                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-xl font-bold text-white/20 pointer-events-none">
                  {country.symbol}
                </span>
                <input
                  type="number" min="1"
                  className="input-apple w-full pl-9 pr-4 py-4 text-3xl font-bold"
                  placeholder="0"
                  value={amount}
                  onChange={e => setAmount(e.target.value)}
                />
              </div>
            </div>

            {/* Conditional fields */}
            {txType === "transfer" && (
              <div>
                <label className="section-label mb-2 block">Recipient ID</label>
                <input
                  className="input-apple w-full px-4 py-2.5 text-sm"
                  placeholder="user_abc123"
                  value={recipient}
                  onChange={e => setRecipient(e.target.value)}
                />
              </div>
            )}

            {["merchant", "payment"].includes(txType) && (
              <div className="space-y-3">
                <div>
                  <label className="section-label mb-2 block">Merchant Name</label>
                  <input
                    className="input-apple w-full px-4 py-2.5 text-sm"
                    placeholder="SM Supermarket"
                    value={merchant}
                    onChange={e => setMerchant(e.target.value)}
                  />
                </div>
                <div>
                  <label className="section-label mb-2 block">Merchant Category</label>
                  <select
                    className="select-apple w-full px-4 py-2.5 text-sm"
                    value={merchantCat}
                    onChange={e => setMerchantCat(e.target.value as MerchantCategory)}
                  >
                    {(["grocery", "utility", "food", "transport", "entertainment", "other"] as MerchantCategory[]).map(c => (
                      <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            {/* Location */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="section-label mb-2 flex items-center gap-1">
                  <Globe size={8} />Country
                </label>
                <select
                  className="select-apple w-full px-3 py-2.5 text-sm"
                  value={country.code}
                  onChange={e => handleCountryChange(e.target.value)}
                >
                  {ASEAN_COUNTRIES.map(c => (
                    <option key={c.code} value={c.code}>{c.flag} {c.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="section-label mb-2 flex items-center gap-1">
                  <MapPin size={8} />City
                </label>
                <select
                  className="select-apple w-full px-3 py-2.5 text-sm"
                  value={city}
                  onChange={e => setCity(e.target.value)}
                >
                  {country.cities.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Advanced */}
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 section-label hover:text-white/50 transition-colors"
            >
              {showAdvanced ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              Advanced Options
            </button>

            {showAdvanced && (
              <div className="space-y-3 pt-1 animate-fade-in">
                <div>
                  <label className="section-label mb-1.5 block">Device ID (session)</label>
                  <input readOnly className="input-apple w-full px-4 py-2 text-xs font-mono text-white/30" value={deviceId} />
                </div>
                <div>
                  <label className="section-label mb-1.5 block">Computed Location</label>
                  <input readOnly className="input-apple w-full px-4 py-2 text-xs font-mono text-white/30" value={`${city}, ${country.code}`} />
                </div>

                {/* Toggle */}
                <label className="flex items-center gap-3 cursor-pointer">
                  <button
                    type="button"
                    onClick={() => setIsNewDevice(!isNewDevice)}
                    className="relative w-9 h-5 rounded-full transition-colors shrink-0"
                    style={{ background: isNewDevice ? "#0A84FF" : "rgba(255,255,255,0.12)" }}
                  >
                    <span className="absolute top-0.5 w-4 h-4 bg-white rounded-full shadow-sm transition-transform"
                      style={{ left: isNewDevice ? "20px" : "2px" }} />
                  </button>
                  <span className="text-sm text-white/55">Simulate new / unrecognized device</span>
                </label>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2.5 px-4 py-3 rounded-xl text-[#FF453A] text-sm"
                style={{ background: "rgba(255,69,58,0.08)", border: "1px solid rgba(255,69,58,0.18)" }}>
                <XCircle size={14} className="shrink-0" />
                {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading || !amount || modelOffline === true}
              className="btn-primary w-full py-4 text-base rounded-xl flex items-center justify-center gap-2.5"
            >
              {loading ? (
                <><Loader2 size={18} className="animate-spin" />Analyzing transaction…</>
              ) : (
                <><Shield size={16} />Submit Transaction</>
              )}
            </button>

            {modelOffline === true && (
              <p className="text-center text-white/25 text-xs">Train the model to enable fraud scoring</p>
            )}
          </div>
        )}

        {/* Transaction history */}
        {history.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <p className="section-label">Transaction History</p>
              <Link href="/" className="text-xs text-[#0A84FF] hover:opacity-75 transition-opacity flex items-center gap-1">
                Dashboard <ArrowRight size={11} />
              </Link>
            </div>
            {history.map((tx, i) => <HistoryRow key={i} tx={tx} symbol={country.symbol} />)}
          </div>
        )}
      </div>
    </div>
  );
}
