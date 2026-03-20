import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { NavLinks } from "./NavLinks";
import { CurrencyProvider } from "./CurrencyContext";
import "./globals.css";

export const metadata: Metadata = {
  title: "FraudShield | V HACK 2026",
  description: "Real-time fraud detection for ASEAN digital wallets — V HACK 2026 Case Study 2",
  icons: { icon: "/logo.png" },
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
                <Image
                  src="/logo.png"
                  alt="FraudShield logo"
                  width={32}
                  height={32}
                  className="drop-shadow-lg"
                  priority
                />
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
