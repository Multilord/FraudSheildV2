import type { Metadata } from "next";
import Link from "next/link";
import { NavLinks } from "./NavLinks";
import { CurrencyProvider } from "./CurrencyContext";
import "./globals.css";

export const metadata: Metadata = {
  title: "FraudShield | V HACK 2026",
  description: "Real-time fraud detection for ASEAN digital wallets — V HACK 2026 Case Study 2",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#09090b] text-white/[0.92] min-h-screen antialiased">
        <CurrencyProvider>
          {/* Frosted-glass navigation */}
          <nav className="nav-glass sticky top-0 z-50 h-14">
            <div className="max-w-6xl mx-auto px-5 h-full flex items-center justify-between">
              <Link
                href="/"
                className="flex items-center gap-2.5 group"
              >
                {/* Hexagon logo mark */}
                <div className="w-7 h-7 rounded-lg bg-[#0A84FF] flex items-center justify-center shadow-lg shadow-[#0A84FF]/30 group-hover:shadow-[#0A84FF]/50 transition-shadow">
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path
                      d="M7 1L12.196 3.75V9.25L7 12L1.804 9.25V3.75L7 1Z"
                      stroke="white"
                      strokeWidth="1.2"
                      fill="rgba(255,255,255,0.15)"
                    />
                    <circle cx="7" cy="7" r="1.5" fill="white" />
                  </svg>
                </div>
                <span className="font-semibold text-[15px] tracking-tight text-white/90 group-hover:text-white transition-colors">
                  FraudShield
                </span>
              </Link>

              <NavLinks />
            </div>
          </nav>

          <div className="animate-fade-in">
            {children}
          </div>
        </CurrencyProvider>
      </body>
    </html>
  );
}
