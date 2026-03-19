"use client";

import { createContext, useContext, useState, ReactNode } from "react";

export interface CurrencyInfo {
  code: string;
  symbol: string;
  name: string;
  flag: string;
}

export const ASEAN_CURRENCIES: CurrencyInfo[] = [
  { code: "BND", symbol: "B$", name: "Brunei Dollar",       flag: "🇧🇳" },
  { code: "KHR", symbol: "₭",  name: "Cambodian Riel",      flag: "🇰🇭" },
  { code: "IDR", symbol: "Rp", name: "Indonesian Rupiah",   flag: "🇮🇩" },
  { code: "LAK", symbol: "₭",  name: "Lao Kip",             flag: "🇱🇦" },
  { code: "MYR", symbol: "RM", name: "Malaysian Ringgit",   flag: "🇲🇾" },
  { code: "MMK", symbol: "K",  name: "Myanmar Kyat",        flag: "🇲🇲" },
  { code: "PHP", symbol: "₱",  name: "Philippine Peso",     flag: "🇵🇭" },
  { code: "SGD", symbol: "S$", name: "Singapore Dollar",    flag: "🇸🇬" },
  { code: "THB", symbol: "฿",  name: "Thai Baht",           flag: "🇹🇭" },
  { code: "USD", symbol: "$",  name: "US Dollar",           flag: "🇹🇱" },
  { code: "VND", symbol: "₫",  name: "Vietnamese Dong",     flag: "🇻🇳" },
];

// Country code → currency code
export const COUNTRY_CURRENCY: Record<string, string> = {
  BN: "BND", KH: "KHR", ID: "IDR", LA: "LAK", MY: "MYR",
  MM: "MMK", PH: "PHP", SG: "SGD", TH: "THB", TL: "USD", VN: "VND",
};

// Units per 1 USD (approximate mid-2025 rates)
const TO_USD: Record<string, number> = {
  BND: 1.35,
  KHR: 4100,
  IDR: 15900,
  LAK: 21000,
  MYR: 4.45,
  MMK: 2100,
  PHP: 58,
  SGD: 1.34,
  THB: 35,
  USD: 1,
  VND: 25400,
};

export function codeFromLocation(location: string): string {
  const countryCode = location?.split(", ").pop()?.toUpperCase() ?? "";
  return COUNTRY_CURRENCY[countryCode] ?? "USD";
}

interface CurrencyContextValue {
  currency: CurrencyInfo;
  setCurrency: (c: CurrencyInfo) => void;
  /** Convert `amount` from `fromCode` currency into the currently selected display currency. */
  convert: (amount: number, fromCode: string) => number;
  /** Convert and format with the display currency symbol, with smart magnitude formatting. */
  display: (amount: number, fromCode: string) => string;
  /** Format a number that is already in the display currency. */
  formatRaw: (amount: number) => string;
}

const CurrencyContext = createContext<CurrencyContextValue | null>(null);

function formatNum(amount: number, symbol: string): string {
  const abs = Math.abs(amount);
  if (abs >= 1_000_000) {
    return `${symbol}${(amount / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 2 })}M`;
  }
  if (abs >= 1_000) {
    return `${symbol}${Math.round(amount).toLocaleString()}`;
  }
  return `${symbol}${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function CurrencyProvider({ children }: { children: ReactNode }) {
  const [currency, setCurrency] = useState<CurrencyInfo>(
    ASEAN_CURRENCIES.find(c => c.code === "PHP")!,
  );

  function convert(amount: number, fromCode: string): number {
    if (fromCode === currency.code) return amount;
    const fromRate = TO_USD[fromCode] ?? 1;
    const toRate   = TO_USD[currency.code] ?? 1;
    return (amount / fromRate) * toRate;
  }

  function display(amount: number, fromCode: string): string {
    return formatNum(convert(amount, fromCode), currency.symbol);
  }

  function formatRaw(amount: number): string {
    return formatNum(amount, currency.symbol);
  }

  return (
    <CurrencyContext.Provider value={{ currency, setCurrency, convert, display, formatRaw }}>
      {children}
    </CurrencyContext.Provider>
  );
}

export function useCurrency() {
  const ctx = useContext(CurrencyContext);
  if (!ctx) throw new Error("useCurrency must be used inside CurrencyProvider");
  return ctx;
}
