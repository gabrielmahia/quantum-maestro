# Governance — Who May Do What

Doctrine: **one executor per account, one auditor per executor, one human above all.**
Two AIs with write access to the same account produce duplicate orders, ambiguous
position state, and un-reconcilable journals. So write access is exclusive.

## Roles

| Account | Executor (write) | Auditor (read-only) | Final authority |
|---|---|---|---|
| Robinhood agentic account | Claude (via Robinhood MCP) | ChatGPT (read-only connection or exported activity feed) | Human |
| Tradier account | Quantum Maestro (this repo, deterministic pipeline) | Claude and/or ChatGPT (read APIs, journal review) | Human |

## Rules
1. An auditor NEVER holds order-placing credentials for the account it audits.
   ChatGPT's Robinhood connection, if made at all, is configured read-only:
   no order tools approved, no "remember approval" on any write action.
2. The executor journals every decision (including no-trades). The auditor's
   weekly job: reconcile journal vs. broker activity feed vs. guardrails.yaml.
3. Any mismatch => executor is disconnected FIRST, investigated second.
4. Role changes (e.g., swapping executor) require a human-authored commit to
   this file. No agent may edit GOVERNANCE.md.
5. Cross-account trades are prohibited: no agent may reason about "moving"
   exposure between Robinhood and Tradier. Each system is fenced.

## Why an auditor at all
Executors grade their own homework badly — LLM or human. A separate model,
with separate context and no execution stake, reviewing the same evidence,
is the cheapest institutional control available to a retail-scale operation.
