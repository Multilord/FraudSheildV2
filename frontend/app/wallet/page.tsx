"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  Send, CreditCard, ArrowDownLeft, ArrowUpRight, ShoppingBag,
  CheckCircle, AlertTriangle, XCircle, ChevronDown, ChevronUp,
  Loader2, Shield, Clock, Cpu, MapPin, Globe,
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
const DEFAULT_COUNTRY = ASEAN_COUNTRIES[6]; // Philippines
const DEFAULT_CITY    = "Manila";

type TxType = "transfer" | "payment" | "cashout" | "topup" | "merchant";
type MerchantCategory = "grocery" | "utility" | "food" | "transport" | "entertainment" | "other";

const TX_ICONS: Record<TxType, React.ReactNode> = {
  transfer: <Send size={16} />,
  payment:  <CreditCard size={16} />,
  cashout:  <ArrowDownLeft size={16} />,
  topup:    <ArrowUpRight size={16} />,
  merchant: <ShoppingBag size={16} />,
};

const TX_LABELS: Record<TxType, string> = {
  transfer: "Transfer",
  payment:  "Pay Bill",
  cashout:  "Cash Out",
  topup:    "Top Up",
  merchant: "Buy",
};

function generateDeviceId(): string {
  if (typeof window === "undefined") return "device-server";
  const stored = sessionStorage.getItem("fraudshield_device_id");
  if (stored) return stored;
  const id = "device-" + Math.random().toString(36).substring(2, 11);
  sessionStorage.setItem("fraudshield_device_id", id);
  return id;
}

// ─── Decision Result Card ─────────────────────────────────────────────────────

function DecisionCard({
  result, onReset, symbol,
}: {
  result: TransactionResult; onReset: () => void; symbol: string;
}) {
  const isApprove = result.decision === "APPROVE";
  const isFlag    = result.decision === "FLAG";
  const riskColor = result.risk_score >= 70 ? "#FF453A" : result.risk_score >= 40 ? "#FF9F0A" : "#30D158";

  const headerBg = isApprove
    ? "bg-[#30D158]/[0.08] border-[#30D158]/20"
    : isFlag
    ? "bg-[#FF9F0A]/[0.08] border-[#FF9F0A]/20"
    : "bg-[#FF453A]/[0.08] border-[#FF453A]/20";

  const iconEl = isApprove
    ? <CheckCircle size={44} style={{ color: "#30D158" }} />
    : isFlag
    ? <AlertTriangle size={44} style={{ color: "#FF9F0A" }} />
    : <XCircle size={44} style={{ color: "#FF453A" }} />;

  const title = isApprove ? "Approved" : isFlag ? "Under Review" : "Blocked";
  const sub   = isApprove
    ? "Transaction processed successfully."
    : isFlag
    ? result.action_required || "Additional verification required."
    : result.action_required || "Rejected due to fraud risk.";

  return (
    <div className="card overflow-hidden animate-fade-up">
      {/* Header */}
      <div className={`p-6 border-b ${headerBg} text-center space-y-3`}>
        {iconEl}
        <div>
          <p className="text-xl font-bold text-white">{title}</p>
          <p className="text-sm text-white/50 mt-1">{sub}</p>
        </div>
        <p className="text-3xl font-bold text-white">{symbol}{result.amount.toLocaleString()}</p>
        <p className="text-[11px] font-mono text-white/25">{result.transaction_id}</p>
      </div>

      <div className="p-5 space-y-5">
        {/* Risk score */}
        <div>
          <div className="flex justify-between items-baseline mb-2">
            <span className="text-[11px] text-white/40 uppercase tracking-widest font-medium">Risk Score</span>
            <span className="text-lg font-bold tabular-nums" style={{ color: riskColor }}>
              {result.risk_score}<span className="text-xs text-white/30 font-normal">/100</span>
            </span>
          </div>
          <div className="w-full bg-white/[0.07] rounded-full h-1.5">
            <div
              className="h-1.5 rounded-full"
              style={{ width: `${result.risk_score}%`, background: riskColor, boxShadow: `0 0 10px ${riskColor}50` }}
            />
          </div>
        </div>

        {/* Risk signals */}
        <div className="space-y-2">
          <p className="text-[11px] text-white/40 uppercase tracking-widest font-medium">Risk Signals</p>
          {result.reasons.map((r, i) => (
            <div key={i} className="flex gap-2.5 text-sm text-white/65 p-2.5 rounded-lg bg-white/[0.03]">
              <span className="text-white/25 mt-0.5 shrink-0">•</span>
              <span>{r}</span>
            </div>
          ))}
        </div>

        {/* 4-model ensemble */}
        <div className="space-y-3 pt-4 border-t border-white/[0.06]">
          <p className="text-[11px] text-white/40 uppercase tracking-widest font-medium flex items-center gap-1.5">
            <Cpu size={10} /> 4-Model Ensemble
          </p>
          {(["xgboost","lightgbm","isolation_forest","lof","behavioral"] as const).map(key => {
            const val = result.model_breakdown[key];
            if (val == null) return null;
            const barColor = val >= 70 ? "#FF453A" : val >= 40 ? "#FF9F0A" : "#30D158";
            const label: Record<string, string> = {
              xgboost: "XGBoost", lightgbm: "LightGBM",
              isolation_forest: "Isolation Forest", lof: "LOF",
              behavioral: "Behavioral",
            };
            return (
              <div key={key} className="flex items-center gap-3">
                <span className="text-xs text-white/35 w-28 shrink-0">{label[key]}</span>
                <div className="flex-1 bg-white/[0.07] rounded-full h-1">
                  <div className="h-1 rounded-full" style={{ width: `${Math.min(100, val)}%`, background: barColor }} />
                </div>
                <span className="text-xs font-mono text-white/50 w-8 text-right tabular-nums">{val.toFixed(0)}</span>
              </div>
            );
          })}
        </div>

        {/* XAI features */}
        {result.xai_top_features && result.xai_top_features.length > 0 && (
          <div className="space-y-2 pt-4 border-t border-white/[0.06]">
            <p className="text-[11px] text-white/40 uppercase tracking-widest font-medium">XAI Attribution</p>
            {result.xai_top_features.slice(0, 3).map((f: XaiFeature, i: number) => (
              <div key={i} className="flex items-center gap-2.5 text-xs">
                <span style={{ color: f.direction === "increases_risk" ? "#FF453A" : "#30D158" }}>
                  {f.direction === "increases_risk" ? "▲" : "▼"}
                </span>
                <span className="text-white/55 flex-1">{f.label}</span>
                <span className="text-white/30 font-mono tabular-nums">{Math.abs(f.contribution).toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}

        {/* Stats row */}
        <div className="flex gap-5 text-xs text-white/30 pt-3 border-t border-white/[0.06]">
          <span className="flex items-center gap-1.5">
            <Cpu size={11} /> {result.confidence ? (result.confidence * 100).toFixed(0) : "—"}% confidence
          </span>
          <span className="flex items-center gap-1.5">
            <Clock size={11} /> {result.latency_ms.toFixed(1)}ms
          </span>
          <span className="flex items-center gap-1.5">
            <Shield size={11} /> {Object.keys(result.model_breakdown).filter(k => k !== "ensemble" && k !== "behavioral").length} ML models
          </span>
        </div>

        <button
          onClick={onReset}
          className="w-full py-3 rounded-xl bg-white/[0.06] hover:bg-white/[0.10] text-white/80 hover:text-white font-medium text-sm transition-all border border-white/[0.07]"
        >
          New Transaction
        </button>
      </div>
    </div>
  );
}

// ─── History Row ──────────────────────────────────────────────────────────────

function HistoryRow({ tx, symbol }: { tx: TransactionResult; symbol: string }) {
  const [open, setOpen] = useState(false);
  const cls = {
    APPROVE: "badge-approve",
    FLAG:    "badge-flag",
    BLOCK:   "badge-block",
  }[tx.decision];

  return (
    <div className="card overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 p-3.5 hover:bg-white/[0.04] transition-colors text-left"
      >
        <span className="text-white/30">{TX_ICONS[tx.transaction_type as TxType] ?? <CreditCard size={16} />}</span>
        <span className="flex-1 text-sm text-white/80">{symbol}{tx.amount.toLocaleString()}</span>
        <span className={`text-[11px] px-2 py-0.5 rounded-full font-semibold ${cls}`}>{tx.decision}</span>
        <span className="text-xs text-white/30 font-mono tabular-nums">{tx.risk_score}</span>
        {open ? <ChevronUp size={13} className="text-white/25" /> : <ChevronDown size={13} className="text-white/25" />}
      </button>
      {open && (
        <div className="px-4 pb-4 space-y-1.5 border-t border-white/[0.06] pt-3">
          <p className="text-[11px] text-white/25 font-mono">{tx.transaction_id}</p>
          {tx.reasons.map((r, i) => (
            <p key={i} className="text-xs text-white/45">• {r}</p>
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

  useEffect(() => {
    api.getHealth().then(setHealth).catch(() => null);
  }, []);

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
        user_id: userId,
        amount: amt,
        transaction_type: txType,
        recipient_id: txType === "transfer" ? recipient || undefined : undefined,
        merchant: ["merchant", "payment"].includes(txType) ? merchant || undefined : undefined,
        merchant_category: ["merchant", "payment"].includes(txType) ? merchantCat : undefined,
        device_type: "mobile",
        device_id: deviceId,
        ip_address: "203.0.113.42",
        location: `${city}, ${country.code}`,
        is_new_device: isNewDevice,
      });

      setResult(res);
      setHistory(prev => [res, ...prev].slice(0, 10));

      if (res.decision === "APPROVE") {
        if (["transfer", "payment", "cashout", "merchant"].includes(txType)) {
          setBalance(prev => Math.max(0, prev - amt));
        } else if (txType === "topup") {
          setBalance(prev => prev + amt);
        }
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
    <div className="min-h-screen bg-[#09090b] pb-16">
      <div className="max-w-md mx-auto px-5 pt-7 space-y-5">

        {/* Offline banner */}
        {modelOffline && (
          <div className="card p-4 border-[#FF9F0A]/20 bg-[#FF9F0A]/[0.05] text-sm text-[#FF9F0A]">
            Fraud model not loaded — train the model first to enable live scoring.
          </div>
        )}

        {/* ── Premium Wallet Card ── */}
        <div className="relative rounded-3xl overflow-hidden shadow-2xl"
          style={{
            background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%)",
            border: "1px solid rgba(255,255,255,0.10)",
          }}
        >
          {/* Decorative circles */}
          <div className="absolute top-0 right-0 w-48 h-48 rounded-full opacity-10"
            style={{ background: "radial-gradient(circle, #0A84FF, transparent)", transform: "translate(30%, -30%)" }} />
          <div className="absolute bottom-0 left-0 w-32 h-32 rounded-full opacity-8"
            style={{ background: "radial-gradient(circle, #BF5AF2, transparent)", transform: "translate(-30%, 30%)" }} />

          <div className="relative p-6">
            <div className="flex justify-between items-start mb-8">
              <div>
                <p className="text-[11px] text-white/40 uppercase tracking-widest font-medium mb-1">eWallet Balance</p>
                <p className="text-[11px] text-white/30 font-mono">{userId}</p>
              </div>
              <div className="flex items-center gap-2.5">
                <span className="text-2xl">{country.flag}</span>
                <div className="w-8 h-8 rounded-xl bg-[#0A84FF]/20 flex items-center justify-center border border-[#0A84FF]/30">
                  <Shield size={14} className="text-[#0A84FF]" />
                </div>
              </div>
            </div>

            <p className="text-4xl font-bold text-white tracking-tight tabular-nums">
              {displayCurrency.symbol}{Math.round(displayBalance).toLocaleString()}
            </p>
            <div className="flex items-center gap-2 mt-1.5">
              <p className="text-white/40 text-xs">Available · {displayCurrency.code}</p>
              {showingConverted && (
                <p className="text-white/20 text-xs">
                  ({country.symbol}{balance.toLocaleString()} {country.currency})
                </p>
              )}
            </div>

            {history[0] && (
              <p className="text-white/25 text-xs mt-5 pt-4 border-t border-white/[0.07]">
                Last: {country.symbol}{history[0].amount.toLocaleString()} — {history[0].transaction_type} — {history[0].decision}
              </p>
            )}
          </div>
        </div>

        {/* User ID */}
        <div>
          <label className="text-[11px] text-white/40 mb-1.5 block uppercase tracking-widest font-medium">User ID</label>
          <input
            className="input-apple w-full px-4 py-2.5 text-sm"
            value={userId}
            onChange={e => setUserId(e.target.value)}
            placeholder="user_demo_001"
          />
        </div>

        {/* ── Transaction Form ── */}
        {result ? (
          <DecisionCard result={result} onReset={reset} symbol={country.symbol} />
        ) : (
          <div className="card p-5 space-y-5">

            {/* TX type selector */}
            <div className="grid grid-cols-5 gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.06]">
              {(["transfer", "payment", "cashout", "topup", "merchant"] as TxType[]).map(t => (
                <button
                  key={t}
                  onClick={() => setTxType(t)}
                  className={`flex flex-col items-center gap-1.5 py-2.5 px-1 rounded-lg text-[10px] font-medium transition-all ${
                    txType === t
                      ? "bg-[#0A84FF] text-white shadow-lg shadow-[#0A84FF]/25"
                      : "text-white/35 hover:text-white/65 hover:bg-white/[0.05]"
                  }`}
                >
                  {TX_ICONS[t]}
                  {TX_LABELS[t]}
                </button>
              ))}
            </div>

            {/* Amount input */}
            <div>
              <label className="text-[11px] text-white/40 mb-2 block uppercase tracking-widest font-medium">
                Amount ({country.currency})
              </label>
              <div className="relative">
                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-2xl font-bold text-white/20 pointer-events-none">
                  {country.symbol}
                </span>
                <input
                  type="number"
                  min="1"
                  className="input-apple w-full pl-10 pr-4 py-4 text-3xl font-bold"
                  placeholder="0"
                  value={amount}
                  onChange={e => setAmount(e.target.value)}
                />
              </div>
            </div>

            {/* Transfer recipient */}
            {txType === "transfer" && (
              <div>
                <label className="text-[11px] text-white/40 mb-1.5 block uppercase tracking-widest font-medium">Recipient ID</label>
                <input
                  className="input-apple w-full px-4 py-2.5 text-sm"
                  placeholder="user_abc123"
                  value={recipient}
                  onChange={e => setRecipient(e.target.value)}
                />
              </div>
            )}

            {/* Merchant fields */}
            {["merchant", "payment"].includes(txType) && (
              <div className="space-y-3">
                <div>
                  <label className="text-[11px] text-white/40 mb-1.5 block uppercase tracking-widest font-medium">Merchant</label>
                  <input
                    className="input-apple w-full px-4 py-2.5 text-sm"
                    placeholder="SM Supermarket"
                    value={merchant}
                    onChange={e => setMerchant(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-[11px] text-white/40 mb-1.5 block uppercase tracking-widest font-medium">Category</label>
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
            <div className="space-y-3">
              <div>
                <label className="text-[11px] text-white/40 mb-1.5 flex items-center gap-1.5 uppercase tracking-widest font-medium">
                  <Globe size={9} /> Country
                </label>
                <select
                  className="select-apple w-full px-4 py-2.5 text-sm"
                  value={country.code}
                  onChange={e => handleCountryChange(e.target.value)}
                >
                  {ASEAN_COUNTRIES.map(c => (
                    <option key={c.code} value={c.code}>
                      {c.flag} {c.name} ({c.currency})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-[11px] text-white/40 mb-1.5 flex items-center gap-1.5 uppercase tracking-widest font-medium">
                  <MapPin size={9} /> City
                </label>
                <select
                  className="select-apple w-full px-4 py-2.5 text-sm"
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
              className="flex items-center gap-1.5 text-xs text-white/30 hover:text-white/55 transition-colors"
            >
              {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              Advanced options
            </button>
            {showAdvanced && (
              <div className="space-y-3 pt-1">
                <div>
                  <label className="text-[11px] text-white/40 mb-1.5 block uppercase tracking-widest font-medium">Device ID</label>
                  <input
                    readOnly
                    className="input-apple w-full px-4 py-2 text-xs font-mono text-white/30"
                    value={deviceId}
                  />
                </div>
                <div>
                  <label className="text-[11px] text-white/40 mb-1.5 block uppercase tracking-widest font-medium">Location</label>
                  <input
                    readOnly
                    className="input-apple w-full px-4 py-2 text-xs font-mono text-white/30"
                    value={`${city}, ${country.code}`}
                  />
                </div>
                <label className="flex items-center gap-3 text-sm text-white/55 cursor-pointer">
                  <div className="relative">
                    <input
                      type="checkbox"
                      checked={isNewDevice}
                      onChange={e => setIsNewDevice(e.target.checked)}
                      className="sr-only"
                    />
                    <div className={`w-9 h-5 rounded-full transition-colors ${isNewDevice ? "bg-[#0A84FF]" : "bg-white/[0.12]"}`}>
                      <div className={`w-4 h-4 bg-white rounded-full absolute top-0.5 transition-transform ${isNewDevice ? "translate-x-4" : "translate-x-0.5"}`} />
                    </div>
                  </div>
                  Simulate new/unrecognized device
                </label>
              </div>
            )}

            {error && (
              <p className="text-[#FF453A] text-sm bg-[#FF453A]/10 border border-[#FF453A]/20 rounded-xl px-4 py-2.5">
                {error}
              </p>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading || !amount || modelOffline === true}
              className="btn-primary w-full py-4 text-base flex items-center justify-center gap-2.5"
            >
              {loading ? <><Loader2 size={18} className="animate-spin" /> Processing…</> : "Submit Transaction"}
            </button>
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <p className="text-xs text-white/40 uppercase tracking-widest font-medium">Recent Transactions</p>
              <Link href="/" className="text-xs text-[#0A84FF] hover:text-[#0A84FF]/70 transition-colors">
                Dashboard →
              </Link>
            </div>
            {history.map((tx, i) => <HistoryRow key={i} tx={tx} symbol={country.symbol} />)}
          </div>
        )}
      </div>
    </div>
  );
}
