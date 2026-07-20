# Tradier Setup — Quantum Maestro

## Key hygiene (do this FIRST if keys were ever pasted anywhere)
1. Tradier dashboard → API Access → **Regenerate** Production key, then Sandbox key.
   Regeneration invalidates old keys instantly. Any key that has touched a chat,
   email, screenshot, or doc is compromised by definition — rotate it.
2. Fresh keys go ONLY into the two places below. Never into code, git, or chat.

## Streamlit Cloud (easystocktrader.streamlit.app)
App → ⋮ → Settings → **Secrets** → paste:

    TRADIER_TOKEN = "fresh-SANDBOX-token"
    TRADIER_ENV = "sandbox"
    TRADIER_ACCOUNT_ID = "VAxxxxxxxx"

Save → app reboots with sandbox trading enabled. Repo is private: if deploys
stall, reauthorize the Streamlit GitHub app for private-repo access.

## Local development
Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` (gitignored),
or export TRADIER_ENV / TRADIER_TOKEN / TRADIER_ACCOUNT_ID as env vars.

## Going live (deliberate, later)
Doctrine: sandbox until 30–50 documented trades show positive after-cost
expectancy. Then: set TRADIER_ENV="production" + live token + live account id
in secrets, AND set QM_LIVE_CONFIRM=I_UNDERSTAND_LIVE_RISK per session.
Both switches are required on purpose. The permission engine still gates
every order regardless of environment.

## Sandbox limitations (per Tradier)
15-min delayed data · no account activity feed · no streaming.
