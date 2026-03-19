import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#09090b",
        surface: "rgba(255,255,255,0.04)",
        border: "rgba(255,255,255,0.07)",
        apple: {
          blue:   "#0A84FF",
          green:  "#30D158",
          amber:  "#FF9F0A",
          red:    "#FF453A",
          purple: "#BF5AF2",
        },
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Display",
          "SF Pro Text",
          "Inter",
          "system-ui",
          "sans-serif",
        ],
        mono: ["SF Mono", "JetBrains Mono", "Fira Code", "monospace"],
      },
      borderRadius: {
        card: "16px",
        pill: "999px",
      },
      backdropBlur: {
        nav: "24px",
      },
      animation: {
        "fade-up":    "fade-up 0.4s cubic-bezier(0.16,1,0.3,1) both",
        "fade-in":    "fade-in 0.3s ease both",
        "slide-down": "slide-down 0.25s cubic-bezier(0.16,1,0.3,1) both",
        "live-pulse": "live-pulse 2s ease-in-out infinite",
        shimmer:      "shimmer 1.6s ease infinite",
      },
      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(12px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          from: { opacity: "0" },
          to:   { opacity: "1" },
        },
        "slide-down": {
          from: { opacity: "0", transform: "translateY(-8px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        "live-pulse": {
          "0%, 100%": { transform: "scale(1)", opacity: "1" },
          "50%":       { transform: "scale(1.6)", opacity: "0.4" },
        },
        shimmer: {
          from: { backgroundPosition: "-200% 0" },
          to:   { backgroundPosition: "200% 0" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
