# PATCH NOTES — Quantum Maestro

## v4.1.0 — 2026-06-30 (Claude Sonnet 5 upgrade)

### Added
- `_call_sonnet5()` helper in app.py — Claude Sonnet 5 available for
  Research tab skills when ANTHROPIC_API_KEY is set in Streamlit secrets
- Sonnet 5 routing comment in .streamlit/config.toml
- Gemini remains default for all cost-sensitive features (free tier)
- Sonnet 5 rationale: ties Opus 4.8 on knowledge work (1618 vs 1615 GDPval-AA v2)

### Background
Claude Sonnet 5 (released June 30, 2026) is the first Sonnet-class model
to approach Opus 4.8 performance on knowledge work benchmarks. For the
Research tab (6 analyst skills: equity research, earnings review, sector
comparison, comparable companies, pre-earnings prep, daily brief),
this means deeper, more nuanced analysis at Sonnet pricing.

Introductory pricing: $2/$10 per MTok through August 31, 2026.

# V4.0.0 — Full Instrument Suite (2026-05-10)

Auto-computed IVR from live VIX data, Instrument Advisor, Futures Calculator (14 contracts),
Beginner-to-Professional language engine, live commodity prices.

See PATCH_NOTES_V3.md and git history for full detail.
