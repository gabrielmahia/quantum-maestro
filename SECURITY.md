# Security Policy

## Reporting a Vulnerability

If you discover an error:

**DO NOT open a public issue.**

Email directly to:
contact@aikungfu.dev

## Supported Versions

Only the latest `main` branch is supported.

## Scope

Quantum Maestro is an analytical simulation tool — it does not execute trades,
hold funds, or connect to brokerage APIs. Security scope covers:
- Data integrity of journal/position exports (CSV)
- Accuracy of risk calculation formulas
- yFinance data handling (third-party, read-only)

Mathematical errors in position sizing or risk calculation logic are treated
as security-equivalent issues because they could affect capital at risk decisions.
