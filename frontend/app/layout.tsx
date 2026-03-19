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
      <body
        style={{
          backgroundColor: "#0a0e1a",
          color: "#f1f5f9",
          minHeight: "100vh",
          fontFamily: "Inter, system-ui, sans-serif",
        }}
      >
        <CurrencyProvider>
          {/* Top navigation */}
          <nav className="border-b border-gray-800 bg-[#0d1117] sticky top-0 z-50">
            <div className="max-w-6xl mx-auto px-4 h-12 flex items-center justify-between">
              <div className="flex items-center gap-6">
                <Link href="/" className="flex items-center gap-2 font-bold text-white">
                  <span className="text-blue-400">⬡</span> FraudShield
                </Link>
              </div>
              <NavLinks />
            </div>
          </nav>

          {children}
        </CurrencyProvider>
      </body>
    </html>
  );
}
