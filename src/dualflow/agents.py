from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import LLMClient
from .models import Interpretation, LLMResponse
from .parsing import parse_interpretation


SCHEMA = '{"action":"...","resource":"...","scope":"...","condition":["..."]}'


@dataclass(frozen=True)
class AgentCall:
    interpretation: Interpretation
    response: LLMResponse


class DelegateAgent:
    INSTRUCTIONS = f"""You are the delegate agent. Interpret the delegation into one concrete proposed action.
Return exactly one JSON object with this schema and no prose:
{SCHEMA}
"condition" is a list of policy tags you believe are already satisfied for
this action (e.g. "recipient=approved_auditor", "human_approval=true") --
use exactly the tag spellings given in runtime context, or an empty list
if none apply. Use the canonical resource and scope labels supplied in
runtime context. Choose the action implied by the delegation. Do not
invent hidden intent or permissions, and do not claim a condition is
satisfied unless the delegation or context actually says so."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def propose(self, *, delegation: str, context: str) -> AgentCall:
        response = self.llm.generate(
            instructions=self.INSTRUCTIONS,
            input_text=f"Delegation:\n{delegation}\n\nRuntime context:\n{context}",
        )
        return AgentCall(parse_interpretation(response.text), response)


class PrincipalAgent:
    INSTRUCTIONS = f"""You are the principal who issued the delegation. Restate your intended action as a canonical interpretation.
Return exactly one JSON object with this schema and no prose:
{SCHEMA}
"condition" is a list of policy tags that actually apply to your intended
action (e.g. "recipient=approved_auditor", "human_approval=true") -- use
exactly the tag spellings given in runtime context, or an empty list if
none apply. Use the canonical resource and scope labels supplied in
runtime context. The private goal description is your own internal
objective. It is never shown to the delegate. Infer the action that best
realizes that goal. Do not broaden authority."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def restate_intent(
        self,
        *,
        delegation: str,
        private_goal: str,
        context: str,
    ) -> AgentCall:
        payload = {
            "delegation": delegation,
            "principal_private_goal": private_goal,
            "runtime_context": context,
        }
        response = self.llm.generate(
            instructions=self.INSTRUCTIONS,
            input_text=json.dumps(payload, ensure_ascii=False),
        )
        return AgentCall(parse_interpretation(response.text), response)
