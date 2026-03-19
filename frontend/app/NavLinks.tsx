"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronDown } from "lucide-react";
import { useCurrency, ASEAN_CURRENCIES } from "./CurrencyContext";

const NAV = [
  { href: "/",       label: "Dashboard" },
  { href: "/triage", label: "Triage"    },
  { href: "/wallet", label: "eWallet"   },
];

function CurrencySwitcher() {
  const { currency, setCurrency } = useCurrency();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-white text-xs px-2.5 py-1.5 rounded-lg font-medium transition"
      >
        <span>{currency.flag}</span>
        <span className="font-mono">{currency.code}</span>
        <ChevronDown size={11} className={`text-gray-400 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-56 bg-[#1a2235] border border-gray-700 rounded-xl shadow-2xl z-50 overflow-hidden">
          <div className="px-3 py-2 border-b border-gray-700">
            <p className="text-xs text-gray-400 font-medium">Display Currency</p>
            <p className="text-[10px] text-gray-600 mt-0.5">Amounts are converted at indicative rates</p>
          </div>
          <div className="max-h-72 overflow-y-auto py-1">
            {ASEAN_CURRENCIES.map(c => (
              <button
                key={c.code}
                onClick={() => { setCurrency(c); setOpen(false); }}
                className={`w-full flex items-center gap-2.5 px-3 py-2 text-sm hover:bg-gray-700/50 transition text-left ${
                  c.code === currency.code ? "bg-blue-900/40 text-blue-300" : "text-gray-200"
                }`}
              >
                <span className="text-base">{c.flag}</span>
                <span className="font-mono font-medium w-8">{c.code}</span>
                <span className="text-gray-400 text-xs">{c.name}</span>
                {c.code === currency.code && (
                  <span className="ml-auto text-blue-400 text-xs">✓</span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function NavLinks() {
  const path = usePathname();
  return (
    <div className="flex items-center gap-2">
      <div className="hidden sm:flex items-center gap-1 text-sm mr-2">
        {NAV.map(({ href, label }) => {
          const active = href === "/" ? path === "/" : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`px-3 py-1.5 rounded-lg font-medium transition ${
                active
                  ? "bg-gray-800 text-white"
                  : "text-gray-400 hover:text-white hover:bg-gray-800/50"
              }`}
            >
              {label}
            </Link>
          );
        })}
      </div>
      <CurrencySwitcher />
    </div>
  );
}
