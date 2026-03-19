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
        bg: "#0a0e1a",
        card: "#111827",
        border: "#1e293b",
        primary: "#f1f5f9",
        muted: "#94a3b8",
        dim: "#64748b",
        accent: {
          blue: "#3b82f6",
          red: "#ef4444",
          yellow: "#eab308",
          green: "#22c55e",
          purple: "#a855f7",
          pink: "#ec4899",
          indigo: "#6366f1",
        },
      },
      borderRadius: {
        card: "12px",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
