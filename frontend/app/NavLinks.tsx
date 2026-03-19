"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronDown, Check } from "lucide-react";
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
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-white/60 hover:text-white/90 hover:bg-white/[0.06] border border-white/[0.08] hover:border-white/[0.14] transition-all duration-150"
      >
        <span className="text-base leading-none">{currency.flag}</span>
        <span className="font-mono tracking-tight">{currency.code}</span>
        <ChevronDown
          size={12}
          className={`text-white/40 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="animate-slide-down absolute right-0 top-full mt-2 w-60 rounded-2xl overflow-hidden shadow-2xl shadow-black/60 z-50"
          style={{
            background: "rgba(28, 28, 30, 0.96)",
            backdropFilter: "blur(24px)",
            border: "1px solid rgba(255,255,255,0.10)",
          }}
        >
          <div className="px-4 py-3 border-b border-white/[0.07]">
            <p className="text-xs font-semibold text-white/80 tracking-wide">Display Currency</p>
            <p className="text-[11px] text-white/35 mt-0.5">Converted at indicative rates</p>
          </div>
          <div className="max-h-72 overflow-y-auto py-1.5">
            {ASEAN_CURRENCIES.map(c => (
              <button
                key={c.code}
                onClick={() => { setCurrency(c); setOpen(false); }}
                className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                  c.code === currency.code
                    ? "bg-[#0A84FF]/12 text-white"
                    : "text-white/65 hover:text-white hover:bg-white/[0.05]"
                }`}
              >
                <span className="text-lg leading-none">{c.flag}</span>
                <span className="font-mono text-xs font-semibold w-8 shrink-0">{c.code}</span>
                <span className="text-xs text-white/40 flex-1">{c.name}</span>
                {c.code === currency.code && (
                  <Check size={13} className="text-[#0A84FF] shrink-0" />
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
      {/* Nav pills */}
      <div className="hidden sm:flex items-center bg-white/[0.04] border border-white/[0.07] rounded-xl p-1 mr-1">
        {NAV.map(({ href, label }) => {
          const active = href === "/" ? path === "/" : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                active
                  ? "bg-white/[0.10] text-white shadow-sm"
                  : "text-white/50 hover:text-white/80 hover:bg-white/[0.05]"
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
