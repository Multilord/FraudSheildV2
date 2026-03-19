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

// ─── Transaction types ────────────────────────────────────────────────────────

type TxType = "transfer" | "payment" | "cashout" | "topup" | "merchant";
type MerchantCategory = "grocery" | "utility" | "food" | "transport" | "entertainment" | "other";

const TX_ICONS: Record<TxType, React.ReactNode> = {
  transfer: <Send size={18} />,
  payment:  <CreditCard size={18} />,
  cashout:  <ArrowDownLeft size={18} />,
  topup:    <ArrowUpRight size={18} />,
  merchant: <ShoppingBag size={18} />,
};

const TX_LABELS: Record<TxType, string> = {
  transfer: "Transfer",
  payment:  "Pay Bill",
  cashout:  "Cash Out",
  topup:    "Top Up",
  merchant: "Buy",
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function generateDeviceId(): string {
  if (typeof window === "undefined") return "device-server";
  const stored = sessionStorage.getItem("fraudshield_device_id");
  if (stored) return stored;
  const id = "device-" + Math.random().toString(36).substring(2, 11);
  sessionStorage.setItem("fraudshield_device_id", id);
  return id;
}

// ─── Decision Card ────────────────────────────────────────────────────────────

function DecisionCard({
  result, onReset, symbol,
}: {
  result: TransactionResult; onReset: () => void; symbol: string;
}) {
  const isApprove = result.decision === "APPROVE";
  const isFlag    = result.decision === "FLAG";
  const isBlock   = result.decision === "BLOCK";

  const bg    = isApprove ? "bg-green-900/40 border-green-600" : isFlag ? "bg-yellow-900/40 border-yellow-600" : "bg-red-900/40 border-red-600";
  const icon  = isApprove ? <CheckCircle size={48} className="text-green-400" /> : isFlag ? <AlertTriangle size={48} className="text-yellow-400" /> : <XCircle size={48} className="text-red-400" />;
  const title = isApprove ? "Transaction Approved" : isFlag ? "Under Review" : "Transaction Blocked";
  const sub   = isApprove ? "Your transaction has been processed successfully." : isFlag ? result.action_required || "Additional verification required." : result.action_required || "Transaction rejected due to fraud risk.";

  const riskColor = result.risk_score >= 70 ? "text-red-400" : result.risk_score >= 40 ? "text-yellow-400" : "text-green-400";

  return (
    <div className={`rounded-2xl border p-6 ${bg} space-y-5`}>
      <div className="flex flex-col items-center text-center gap-3">
        {icon}
        <div>
          <p className="text-xl font-bold text-white">{title}</p>
          <p className="text-sm text-gray-300 mt-1">{sub}</p>
        </div>
        <p className="text-3xl font-bold text-white">{symbol}{result.amount.toLocaleString()}</p>
        <p className="text-xs text-gray-400 font-mono">{result.transaction_id}</p>
      </div>

      {/* Risk score */}
      <div>
        <div className="flex justify-between mb-1 text-xs text-gray-400">
          <span>Risk Score</span>
          <span className={`font-bold ${riskColor}`}>{result.risk_score}/100</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all ${result.risk_score >= 70 ? "bg-red-500" : result.risk_score >= 40 ? "bg-yellow-500" : "bg-green-500"}`}
            style={{ width: `${result.risk_score}%` }}
          />
        </div>
      </div>

      {/* Reasons */}
      <div className="space-y-1">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Risk Signals</p>
        {result.reasons.map((r, i) => (
          <div key={i} className="flex gap-2 text-sm text-gray-200">
            <span className="text-gray-500 mt-0.5">•</span>
            <span>{r}</span>
          </div>
        ))}
      </div>

      {/* 4-model ensemble breakdown */}
      <div className="space-y-1.5 pt-3 border-t border-white/10">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1">
          <Cpu size={10} /> 4-Model Ensemble
        </p>
        {(["xgboost","lightgbm","isolation_forest","lof","behavioral"] as const).map(key => {
          const val = result.model_breakdown[key];
          if (val == null) return null;
          const color = val >= 70 ? "bg-red-500" : val >= 40 ? "bg-yellow-500" : "bg-green-500";
          const label: Record<string,string> = {
            xgboost: "XGBoost", lightgbm: "LightGBM",
            isolation_forest: "Isolation Forest", lof: "LOF",
            behavioral: "Behavioral",
          };
          return (
            <div key={key} className="flex items-center gap-2">
              <span className="text-xs text-gray-500 w-28 shrink-0">{label[key]}</span>
              <div className="flex-1 bg-white/10 rounded-full h-1.5">
                <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${Math.min(100, val)}%` }} />
              </div>
              <span className="text-xs font-mono text-gray-300 w-8 text-right">{val.toFixed(0)}</span>
            </div>
          );
        })}
      </div>

      {/* XAI top features */}
      {result.xai_top_features && result.xai_top_features.length > 0 && (
        <div className="space-y-1 pt-2 border-t border-white/10">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">XAI Feature Attribution</p>
          {result.xai_top_features.slice(0, 3).map((f: XaiFeature, i: number) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className={f.direction === "increases_risk" ? "text-red-400" : "text-green-400"}>
                {f.direction === "increases_risk" ? "▲" : "▼"}
              </span>
              <span className="text-gray-300 flex-1">{f.label}</span>
              <span className="text-gray-500 font-mono">{Math.abs(f.contribution).toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Stats row */}
      <div className="flex gap-4 text-xs text-gray-400 pt-1 border-t border-white/10">
        <span className="flex items-center gap-1"><Cpu size={12} /> {result.confidence ? (result.confidence * 100).toFixed(0) : "—"}% confidence</span>
        <span className="flex items-center gap-1"><Clock size={12} /> {result.latency_ms.toFixed(1)}ms</span>
        <span className="flex items-center gap-1"><Shield size={12} /> {Object.keys(result.model_breakdown).filter(k => k !== "ensemble" && k !== "behavioral").length} ML models</span>
      </div>

      <button
        onClick={onReset}
        className="w-full py-3 rounded-xl bg-white/10 hover:bg-white/20 text-white font-medium text-sm transition"
      >
        New Transaction
      </button>
    </div>
  );
}

// ─── History Row ──────────────────────────────────────────────────────────────

function HistoryRow({ tx, symbol }: { tx: TransactionResult; symbol: string }) {
  const [open, setOpen] = useState(false);
  const badge =
    tx.decision === "APPROVE" ? "bg-green-900 text-green-300"
    : tx.decision === "FLAG"  ? "bg-yellow-900 text-yellow-300"
    : "bg-red-900 text-red-300";

  return (
    <div className="border border-gray-800 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 p-3 hover:bg-gray-800/50 transition text-left"
      >
        <span className="text-gray-400">{TX_ICONS[tx.transaction_type as TxType] ?? <CreditCard size={18} />}</span>
        <span className="flex-1 text-sm text-white">{symbol}{tx.amount.toLocaleString()}</span>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badge}`}>{tx.decision}</span>
        <span className="text-xs text-gray-500">{tx.risk_score}</span>
        {open ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
      </button>
      {open && (
        <div className="px-4 pb-3 space-y-1 border-t border-gray-800 pt-2">
          <p className="text-xs text-gray-400 font-mono">{tx.transaction_id}</p>
          {tx.reasons.map((r, i) => (
            <p key={i} className="text-xs text-gray-300">• {r}</p>
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

  // Balance converted to display currency
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
    <div className="min-h-screen bg-[#0a0e1a] text-white pb-16">
      <div className="max-w-md mx-auto px-4 pt-6 space-y-5">

        {/* Model offline banner */}
        {modelOffline && (
          <div className="bg-yellow-900/40 border border-yellow-600 rounded-xl p-3 text-sm text-yellow-200">
            Fraud model not loaded. Train the model first before submitting transactions.
          </div>
        )}

        {/* Wallet card */}
        <div className="rounded-2xl bg-gradient-to-br from-blue-700 via-blue-800 to-indigo-900 p-6 shadow-xl">
          <div className="flex justify-between items-start mb-6">
            <div>
              <p className="text-blue-200 text-xs font-medium uppercase tracking-widest">My eWallet</p>
              <p className="text-white/70 text-xs mt-1 font-mono">{userId}</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xl">{country.flag}</span>
              <Shield size={18} className="text-blue-300" />
            </div>
          </div>
          <p className="text-4xl font-bold text-white">
            {displayCurrency.symbol}{Math.round(displayBalance).toLocaleString()}
          </p>
          <div className="flex items-center gap-2 mt-1">
            <p className="text-blue-200 text-xs">Available Balance · {displayCurrency.code}</p>
            {showingConverted && (
              <span className="text-blue-300/50 text-xs">
                ({country.symbol}{balance.toLocaleString()} {country.currency})
              </span>
            )}
          </div>
          {history[0] && (
            <p className="text-blue-300/70 text-xs mt-4">
              Last: {country.symbol}{history[0].amount.toLocaleString()} — <span className="capitalize">{history[0].transaction_type}</span> — {history[0].decision}
            </p>
          )}
        </div>

        {/* User ID */}
        <div>
          <label className="text-xs text-gray-400 mb-1 block">User ID</label>
          <input
            className="w-full bg-gray-800 border border-gray-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            value={userId}
            onChange={e => setUserId(e.target.value)}
            placeholder="user_demo_001"
          />
        </div>

        {result ? (
          <DecisionCard result={result} onReset={reset} symbol={country.symbol} />
        ) : (
          <div className="bg-[#111827] border border-gray-800 rounded-2xl p-5 space-y-5">

            {/* Transaction type tabs */}
            <div className="grid grid-cols-5 gap-1 bg-gray-900 rounded-xl p-1">
              {(["transfer", "payment", "cashout", "topup", "merchant"] as TxType[]).map(t => (
                <button
                  key={t}
                  onClick={() => setTxType(t)}
                  className={`flex flex-col items-center gap-1 py-2 px-1 rounded-lg text-xs font-medium transition ${
                    txType === t ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"
                  }`}
                >
                  {TX_ICONS[t]}
                  <span className="text-[10px]">{TX_LABELS[t]}</span>
                </button>
              ))}
            </div>

            {/* Amount */}
            <div>
              <label className="text-xs text-gray-400 mb-1 block">Amount ({country.currency})</label>
              <div className="relative">
                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-2xl font-bold text-gray-500 pointer-events-none">
                  {country.symbol}
                </span>
                <input
                  type="number"
                  min="1"
                  className="w-full bg-gray-900 border border-gray-700 rounded-xl pl-10 pr-4 py-4 text-3xl font-bold text-white focus:outline-none focus:border-blue-500"
                  placeholder="0.00"
                  value={amount}
                  onChange={e => setAmount(e.target.value)}
                />
              </div>
            </div>

            {/* Conditional fields */}
            {txType === "transfer" && (
              <div>
                <label className="text-xs text-gray-400 mb-1 block">Recipient ID</label>
                <input
                  className="w-full bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                  placeholder="user_abc123"
                  value={recipient}
                  onChange={e => setRecipient(e.target.value)}
                />
              </div>
            )}

            {["merchant", "payment"].includes(txType) && (
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">Merchant</label>
                  <input
                    className="w-full bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                    placeholder="SM Supermarket"
                    value={merchant}
                    onChange={e => setMerchant(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">Category</label>
                  <select
                    className="w-full bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
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

            {/* Location — Country + City dropdowns */}
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                  <Globe size={10} /> Country
                </label>
                <select
                  className="w-full bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
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
                <label className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                  <MapPin size={10} /> City
                </label>
                <select
                  className="w-full bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                  value={city}
                  onChange={e => setCity(e.target.value)}
                >
                  {country.cities.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Advanced toggle */}
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300"
            >
              {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              Advanced options
            </button>
            {showAdvanced && (
              <div className="space-y-3 pt-1">
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">Device ID</label>
                  <input
                    readOnly
                    className="w-full bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-xs text-gray-500 font-mono"
                    value={deviceId}
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">Location (computed)</label>
                  <input
                    readOnly
                    className="w-full bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-xs text-gray-500 font-mono"
                    value={`${city}, ${country.code}`}
                  />
                </div>
                <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={isNewDevice}
                    onChange={e => setIsNewDevice(e.target.checked)}
                    className="rounded"
                  />
                  Simulate new/unrecognized device
                </label>
              </div>
            )}

            {error && (
              <p className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading || !amount || modelOffline === true}
              className="w-full py-4 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-base transition flex items-center justify-center gap-2"
            >
              {loading ? <><Loader2 size={18} className="animate-spin" /> Processing…</> : "Submit Transaction"}
            </button>
          </div>
        )}

        {/* Transaction history */}
        {history.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold text-gray-300">Recent Transactions</p>
              <Link href="/" className="text-xs text-blue-400 hover:text-blue-300">
                View Dashboard →
              </Link>
            </div>
            {history.map((tx, i) => <HistoryRow key={i} tx={tx} symbol={country.symbol} />)}
          </div>
        )}
      </div>
    </div>
  );
}
