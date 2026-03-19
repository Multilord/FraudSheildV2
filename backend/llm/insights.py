"""
InsightsEngine — Claude-powered fraud analysis
Uses Anthropic API for hypothesis generation and action advisory.
Falls back to deterministic responses when API key is not set.
"""

import os
import re
from typing import Any, Dict, List, Optional

try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False


SYSTEM_PROMPT = """You are an expert fraud analyst AI for an ASEAN digital wallet platform.
You specialise in detecting money laundering, synthetic identity fraud, account takeover, and card fraud.
When analysing transactions, provide concise, evidence-based assessments.
Focus on patterns, anomalies, and connections that indicate fraudulent behaviour.
Keep responses clear and actionable for compliance analysts."""


class InsightsEngine:
    """
    AI-powered fraud insights using Claude.
    Falls back to rule-based responses when ANTHROPIC_API_KEY is not set.
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.model = "claude-sonnet-4-6"
        self._client: Optional[Any] = None

        if ANTHROPIC_OK and self.api_key:
            try:
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except Exception as exc:
                print(f"[InsightsEngine] Failed to initialise Anthropic client: {exc}")
                print("[InsightsEngine] Running in offline/fallback mode.")

    # ─── Public API ───────────────────────────────────────────────────────────

    async def generate_hypothesis(self, transaction_data: Dict[str, Any]) -> str:
        """Generate a fraud hypothesis for a transaction."""
        if self._client is None:
            return self._fallback_hypothesis(transaction_data)

        prompt = self._build_hypothesis_prompt(transaction_data)
        try:
            message = await self._client.messages.create(
                model=self.model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as exc:
            print(f"[InsightsEngine] Claude API error: {exc}")
            return self._fallback_hypothesis(transaction_data)

    def extract_key_indicators(self, transaction_data: Dict[str, Any]) -> List[str]:
        """Extract key risk indicators from transaction data."""
        indicators: List[str] = []
        amount = transaction_data.get("amount", 0)
        risk_score = transaction_data.get("risk_score", 0)
        fraud_type = transaction_data.get("fraud_type", "")

        if amount > 10000:
            indicators.append(f"High-value transaction: ${amount:,.0f}")
        if risk_score and float(risk_score) > 0.8:
            indicators.append(f"Extreme risk score: {float(risk_score):.0%}")
        if fraud_type:
            indicators.append(f"Fraud pattern: {fraud_type}")

        # Behavioural signals
        vpn = transaction_data.get("vpn_usage_rate", 0)
        if vpn and float(vpn) > 0.5:
            indicators.append(f"High VPN usage: {float(vpn):.0%} of sessions")

        geo_hops = transaction_data.get("geo_hops", 0)
        if geo_hops and int(geo_hops) > 3:
            indicators.append(f"Multiple geographic IP hops: {geo_hops}")

        synthetic_prob = transaction_data.get("synthetic_id_prob", 0)
        if synthetic_prob and float(synthetic_prob) > 0.7:
            indicators.append(f"High synthetic identity probability: {float(synthetic_prob):.0%}")

        if not indicators:
            indicators.append("Anomalous transaction pattern detected by ML ensemble")

        return indicators

    def detect_patterns(self, transaction_data: Dict[str, Any]) -> List[str]:
        """Detect fraud patterns in transaction data."""
        patterns: List[str] = []
        fraud_type = str(transaction_data.get("fraud_type", "")).lower()

        if "laundering" in fraud_type or "launder" in fraud_type:
            patterns.extend([
                "Deposit-withdrawal cycle (smurfing/layering)",
                "Near-complete fund recovery (>95%)",
                "Minimal legitimate trading to simulate activity",
            ])
        elif "synthetic" in fraud_type or "identity" in fraud_type:
            patterns.extend([
                "Synthetic identity creation indicators",
                "Velocity spike — multiple transactions in short window",
                "Document quality failure",
            ])
        elif "takeover" in fraud_type or "ato" in fraud_type:
            patterns.extend([
                "Geographic anomaly — login from new region",
                "New device fingerprint",
                "Credential change followed by immediate withdrawal",
            ])
        else:
            patterns.extend([
                "Behavioural deviation from account baseline",
                "ML ensemble anomaly score elevated",
            ])

        return patterns

    async def suggest_actions(
        self,
        risk_score: float,
        fraud_type: str,
        case_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Suggest remediation actions based on risk level and fraud type."""
        if self._client is None:
            return self._fallback_actions(risk_score, fraud_type)

        prompt = (
            f"Fraud case analysis:\n"
            f"- Risk score: {risk_score:.0%}\n"
            f"- Fraud type: {fraud_type}\n"
            f"- Case ID: {case_id or 'N/A'}\n\n"
            "List exactly 3 recommended remediation actions as a numbered list.\n"
            "For each action include: action name, priority (HIGH/MEDIUM/LOW), and a one-sentence rationale.\n"
            "Format each line as: N. [PRIORITY] Action Name — Rationale"
        )
        try:
            message = await self._client.messages.create(
                model=self.model,
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_actions(message.content[0].text)
        except Exception as exc:
            print(f"[InsightsEngine] Claude API error: {exc}")
            return self._fallback_actions(risk_score, fraud_type)

    # ─── Fallbacks ────────────────────────────────────────────────────────────

    def _fallback_hypothesis(self, transaction_data: Dict[str, Any]) -> str:
        fraud_type = str(transaction_data.get("fraud_type", "unknown")).lower()
        amount = transaction_data.get("amount", 0)
        risk = transaction_data.get("risk_score", 0.5)

        if "laundering" in fraud_type:
            return (
                f"Transaction analysis indicates a structured money laundering scheme. "
                f"The deposit of ${amount:,.0f} followed by rapid withdrawal is consistent "
                f"with layering techniques. Risk score of {float(risk):.0%} from ML ensemble "
                f"confirms high suspicion of criminal fund movement."
            )
        if "synthetic" in fraud_type or "identity" in fraud_type:
            return (
                f"Account exhibits strong indicators of synthetic identity fraud. "
                f"ML ensemble risk score of {float(risk):.0%} driven by document quality issues, "
                f"face match failure, and transaction velocity anomalies consistent with "
                f"fabricated identity patterns."
            )
        if "takeover" in fraud_type or "ato" in fraud_type:
            return (
                f"Account takeover indicators detected. Geographic and device anomalies "
                f"suggest unauthorised access. Credential change followed immediately by "
                f"high-value withdrawal attempt (${amount:,.0f}) is a classic ATO pattern. "
                f"ML ensemble confidence: {float(risk):.0%}."
            )
        return (
            f"ML ensemble model flagged this transaction with a risk score of {float(risk):.0%}. "
            f"Behavioural and financial anomalies detected. Manual review recommended."
        )

    def _fallback_actions(
        self, risk_score: float, fraud_type: str
    ) -> List[Dict[str, Any]]:
        fraud_lower = fraud_type.lower()

        if "laundering" in fraud_lower:
            return [
                {"action": "Freeze Withdrawals", "priority": "HIGH", "confidence": 0.95,
                 "rationale": "Prevent further fund movement pending investigation."},
                {"action": "Escalate to Compliance", "priority": "HIGH", "confidence": 0.91,
                 "rationale": "AML regulations require SAR filing within 30 days."},
                {"action": "Lock Account", "priority": "MEDIUM", "confidence": 0.88,
                 "rationale": "Restrict account access to preserve evidence chain."},
            ]
        if "synthetic" in fraud_lower or "identity" in fraud_lower:
            return [
                {"action": "Request Re-KYC", "priority": "HIGH", "confidence": 0.93,
                 "rationale": "Identity verification failed — re-verification mandatory."},
                {"action": "Limit Transactions", "priority": "HIGH", "confidence": 0.89,
                 "rationale": "Cap transaction volume until identity is confirmed."},
                {"action": "Flag for Review", "priority": "MEDIUM", "confidence": 0.82,
                 "rationale": "Assign to fraud analyst queue for manual assessment."},
            ]
        if "takeover" in fraud_lower or "ato" in fraud_lower:
            return [
                {"action": "Force Re-authentication", "priority": "HIGH", "confidence": 0.96,
                 "rationale": "Invalid session must be terminated immediately."},
                {"action": "Block Withdrawals", "priority": "HIGH", "confidence": 0.94,
                 "rationale": "Prevent financial loss from unauthorised withdrawal attempt."},
                {"action": "Send Security Alert", "priority": "MEDIUM", "confidence": 0.87,
                 "rationale": "Notify account holder of suspicious activity via secondary channel."},
            ]
        # Generic high-risk
        priority = "HIGH" if risk_score >= 0.75 else "MEDIUM"
        return [
            {"action": "Manual Review", "priority": priority, "confidence": risk_score,
             "rationale": "Risk score exceeds automated decision threshold."},
            {"action": "Restrict Account", "priority": "MEDIUM", "confidence": 0.75,
             "rationale": "Limit account activity pending investigation outcome."},
            {"action": "Monitor Closely", "priority": "LOW", "confidence": 0.65,
             "rationale": "Increase transaction monitoring frequency for 30 days."},
        ]

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _build_hypothesis_prompt(self, data: Dict[str, Any]) -> str:
        lines = ["Analyse this fraud case and provide a concise hypothesis (2-3 paragraphs):\n"]
        for key, value in data.items():
            lines.append(f"- {key}: {value}")
        lines.append(
            "\nExplain the likely fraud mechanism, key evidence, and confidence level."
        )
        return "\n".join(lines)

    def _parse_actions(self, text: str) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        priority_map = {"HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"}
        priority_confidence = {"HIGH": 0.92, "MEDIUM": 0.78, "LOW": 0.65}

        for line in lines:
            # Match pattern: "1. [HIGH] Freeze Account — Rationale here"
            match = re.match(r"^\d+\.\s+\[?(\w+)\]?\s+([^—–-]+)[—–-]+(.+)$", line)
            if match:
                raw_priority = match.group(1).upper()
                priority = priority_map.get(raw_priority, "MEDIUM")
                action_name = match.group(2).strip()
                rationale = match.group(3).strip()
                actions.append({
                    "action": action_name,
                    "priority": priority,
                    "confidence": priority_confidence[priority],
                    "rationale": rationale,
                })
            elif actions is not None and len(actions) < 3 and len(line) > 10:
                # Fallback: treat any non-empty line as an action
                actions.append({
                    "action": line[:60],
                    "priority": "MEDIUM",
                    "confidence": 0.75,
                    "rationale": "AI-recommended action based on case analysis.",
                })

        # Ensure we always return exactly 3 actions
        while len(actions) < 3:
            actions.append({
                "action": "Manual Review Required",
                "priority": "MEDIUM",
                "confidence": 0.70,
                "rationale": "Automated analysis recommends human review.",
            })

        return actions[:3]
