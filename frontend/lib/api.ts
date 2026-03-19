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
  transaction_id: string;
  user_id: string;
  amount: number;
  transaction_type: string;
  risk_score: number;
  decision: "APPROVE" | "FLAG" | "BLOCK";
  reasons: string[];
  confidence: number;
  latency_ms: number;
  action_required: string | null;
  timestamp: string;
  model_breakdown: {
    xgboost?: number;
    lightgbm?: number;
    random_forest?: number;
    logistic_regression?: number;
    behavioral?: number;
    ensemble: number;
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
  demo_mode?: boolean;
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

  getHealth: (): Promise<HealthStatus> =>
    fetchJSON("/api/health"),

  getMetrics: (): Promise<Record<string, unknown>> =>
    fetchJSON("/api/metrics"),
};

// ─── WebSocket hook ───────────────────────────────────────────────────────────

export function useAlertStream(onAlert: (alert: LiveAlert) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    function connect() {
      try {
        const ws = new WebSocket(`${WS_BASE}/ws/alerts`);
        ws.onopen = () => { setConnected(true); ws.send("ping"); };
        ws.onmessage = (e) => {
          try { onAlert(JSON.parse(e.data as string) as LiveAlert); } catch { /* ignore */ }
        };
        ws.onclose = () => {
          setConnected(false);
          timerRef.current = setTimeout(connect, 3000);
        };
        ws.onerror = () => ws.close();
        wsRef.current = ws;
      } catch {
        timerRef.current = setTimeout(connect, 3000);
      }
    }
    connect();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return connected;
}

export { API_BASE, WS_BASE };
