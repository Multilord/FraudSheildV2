"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface WalletTransactionRequest {
  user_id: string;
  amount: number;
  transaction_type: "transfer" | "payment" | "cashout" | "topup" | "merchant";
  recipient_id?: string;
  merchant?: string;
  merchant_category?: string;
  device_type: "mobile" | "desktop" | "pos";
  device_id: string;
  ip_address: string;
  location: string;
  is_new_device: boolean;
  note?: string;
}

export interface XaiFeature {
  feature: string;
  label: string;
  contribution: number;
  direction: "increases_risk" | "reduces_risk";
}

export interface TransactionResult {
  // Core API spec fields
  risk_score: number;
  decision: "APPROVE" | "FLAG" | "BLOCK";
  confidence: number;
  explanation: string;
  top_risk_factors: string[];
  model_contributions: Record<string, number>;
  // Extended fields
  transaction_id: string;
  user_id: string;
  amount: number;
  transaction_type: string;
  reasons: string[];
  latency_ms: number;
  action_required: string | null;
  timestamp: string;
  /** Contribution breakdown — how many 0-100 points each component added to final_prob */
  model_breakdown: {
    // New composition keys (present from updated engine)
    ml_ensemble?: number;
    anomaly?: number;
    behavioral?: number;
    escalation?: number;
    ensemble?: number;  // = final_prob * 100, backward-compat
    // Legacy raw-probability keys (may be present in older stored transactions)
    xgboost?: number;
    lightgbm?: number;
    isolation_forest?: number;
    lof?: number;
  };
  /** Raw model outputs — honest probabilities separate from contribution points */
  model_raw_probabilities?: {
    xgboost?: number;
    lightgbm?: number;
    isolation_forest?: number;
    lof?: number;
    ml_ensemble?: number;
    behavioral?: number;
    final?: number;
  };
  xai_top_features?: XaiFeature[];
}

export interface DashboardStats {
  total: number;
  approved_count: number;
  flagged_count: number;
  blocked_count: number;
  fraud_rate: number;
  avg_risk_score: number;
  total_amount: number;
  total_blocked_amount: number;
  total_approved_amount: number;
  total_flagged_amount: number;
  demo_mode?: boolean;
}

export interface DashboardCharts {
  hourly_trend: Array<{
    /** ISO-8601 UTC prefix, e.g. "2026-03-20T14" */
    hour: string;
    total: number;
    blocked: number;
    flagged: number;
    approved: number;
  }>;
  risk_distribution: Array<{
    /** Human label, e.g. "40–49" */
    bucket: string;
    /** Numeric bucket start, e.g. 40 */
    bucket_start: number;
    count: number;
  }>;
}

export interface StoredTransaction {
  transaction_id: string;
  user_id: string;
  amount: number;
  transaction_type: string;
  device_type: string;
  ip_address: string;
  location: string;
  merchant: string;
  merchant_category: string;
  timestamp: string;
  risk_score: number;
  decision: "APPROVE" | "FLAG" | "BLOCK";
  reasons: string[] | string;
  confidence: number;
  latency_ms: number;
  features?: { model_breakdown?: Record<string, number> } | string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  artifact_version: string | null;
  demo_mode: boolean;
  database_connected: boolean;
  timestamp: string;
  thresholds?: { flag: number; block: number } | null;
}

export interface LiveAlert {
  type: string;
  transaction_id: string;
  user_id: string;
  amount: number;
  risk_score: number;
  decision: "APPROVE" | "FLAG" | "BLOCK";
  reasons: string[];
  timestamp: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function parseReasons(raw: string[] | string | undefined): string[] {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  try { return JSON.parse(raw) as string[]; } catch { return [String(raw)]; }
}

// ─── API ──────────────────────────────────────────────────────────────────────

export const api = {
  submitTransaction: (tx: WalletTransactionRequest): Promise<TransactionResult> =>
    fetchJSON("/api/wallet/transaction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tx),
    }),

  getDashboardStats: (): Promise<DashboardStats> =>
    fetchJSON("/api/dashboard/stats"),

  getDashboardTransactions: (
    limit = 50,
    offset = 0,
    decision?: string
  ): Promise<{ transactions: StoredTransaction[]; count: number }> =>
    fetchJSON(
      `/api/dashboard/transactions?limit=${limit}&offset=${offset}${
        decision ? `&decision=${decision}` : ""
      }`
    ),

  getCase: (id: string): Promise<StoredTransaction> =>
    fetchJSON(`/api/dashboard/cases/${id}`),

  getDashboardCharts: (): Promise<DashboardCharts> =>
    fetchJSON("/api/dashboard/charts"),

  getHealth: (): Promise<HealthStatus> =>
    fetchJSON("/api/health"),

  getMetrics: (): Promise<Record<string, unknown>> =>
    fetchJSON("/api/metrics"),
};

// ─── WebSocket hook ───────────────────────────────────────────────────────────

export function useAlertStream(onAlert: (alert: LiveAlert) => void) {
  const wsRef       = useRef<WebSocket | null>(null);
  const timerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const destroyedRef = useRef(false);
  // Always hold the latest onAlert without making it a WebSocket effect dep.
  // This prevents stale-closure bugs and React StrictMode race conditions.
  const onAlertRef  = useRef(onAlert);
  const [connected, setConnected] = useState(false);

  useEffect(() => { onAlertRef.current = onAlert; });

  useEffect(() => {
    destroyedRef.current = false;

    function connect() {
      if (destroyedRef.current) return;
      try {
        const ws = new WebSocket(`${WS_BASE}/ws/alerts`);
        wsRef.current = ws;

        ws.onopen = () => {
          if (destroyedRef.current) { ws.close(); return; }
          setConnected(true);
          ws.send("ping");
        };

        ws.onmessage = (e) => {
          if (destroyedRef.current) return;
          try {
            const msg = JSON.parse(e.data as string) as LiveAlert;
            onAlertRef.current(msg);
          } catch { /* ignore malformed frames */ }
        };

        ws.onclose = () => {
          if (destroyedRef.current) return;
          setConnected(false);
          timerRef.current = setTimeout(connect, 3_000);
        };

        ws.onerror = () => ws.close();
      } catch {
        if (!destroyedRef.current) {
          timerRef.current = setTimeout(connect, 3_000);
        }
      }
    }

    connect();

    return () => {
      destroyedRef.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // suppress reconnect on teardown
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []); // intentionally no deps — connection lifecycle managed internally

  return connected;
}

export { API_BASE, WS_BASE };
