# Connecting Tradier to EasyStockTrader

## What you get when connected
- **Broker-computed Greeks** — delta, gamma, theta, vega from Tradier's own vol surface model (smv_vol), not BSM approximation
- **Real-time or 15-min delayed quotes** (depending on your account tier)
- **Better IV data** — Tradier's smoothed market vol (smv_vol) is more accurate than yfinance for liquid options

Without Tradier, the app uses yfinance (delayed IV) + exact BSM formula — still accurate, just slightly behind real-time.

---

## Step 1 — Get your Tradier API token

1. Log in at [tradier.com](https://tradier.com)
2. Click **My Account** (top-right corner)
3. Click **API Access** in the left sidebar
4. Under **API Tokens**, click **Generate Token** (or copy your existing token)
5. Copy the token — it looks like: `abc123def456...` (long alphanumeric string)

**Which environment to use:**
| Environment | Token source | Data quality |
|------------|-------------|-------------|
| **Sandbox** | Paper trading account | Simulated/delayed — use only for testing |
| **Production** | Live or developer account | Real market data |

→ If your account was just created, start with **sandbox** to verify the connection works, then switch to production.

---

## Step 2 — Add the token to Streamlit

### Option A: Streamlit Cloud (easystocktrader.streamlit.app)
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Find the **easystocktrader** app
3. Click the **⋮ menu** → **Settings** → **Secrets**
4. Paste this (replace with your actual token and environment):

```toml
TRADIER_TOKEN = "your_token_here"
TRADIER_ENV = "production"
```

→ Click **Save**. The app restarts automatically.

### Option B: Local development
Create or edit `.streamlit/secrets.toml` in the project root:

```toml
TRADIER_TOKEN = "your_token_here"
TRADIER_ENV = "production"
```

> ⚠️ **Never commit secrets.toml to GitHub.** It's already in `.gitignore`.

---

## Step 3 — Verify the connection

After adding the token:
1. Open [easystocktrader.streamlit.app](https://easystocktrader.streamlit.app)
2. Look at the **top of the sidebar**
3. You should see: **✅ Tradier connected — broker Greeks active**

If you see the grey "📡 Greeks: yfinance+BSM" message instead:
- Check that the token is correct (no extra spaces)
- Check that TRADIER_ENV matches your account type (`sandbox` or `production`)
- Check Streamlit Cloud secrets were saved and the app restarted

---

## What changes in the app when connected

| Feature | Without Tradier | With Tradier |
|---------|----------------|-------------|
| Implied Volatility | yfinance delayed bid/ask | Tradier smv_vol (smoothed surface) |
| Delta | BSM from IV | Broker-computed |
| Gamma | BSM from IV | Broker-computed |
| Theta | BSM from IV | Broker-computed |
| Vega | BSM from IV | Broker-computed |
| Quote latency | 15-20 min delay | Real-time (production) |
| IVR | Realized vol proxy | Realized vol proxy (same — historical IV still best-effort) |
| Source label in app | "yfinance delayed IV + BSM exact (scipy)" | "Tradier broker-computed Greeks (smv_vol surface)" |

---

## Tradier API limits (free developer tier)

| Limit | Value |
|-------|-------|
| Requests per second | 1 req/sec |
| Options chain requests | ~200/day on free tier |
| Quote requests | Higher limit |
| Historical data | Limited on free tier |

The app caches options chain data for **60 seconds** to stay well within rate limits.

---

## Troubleshooting

**"HTTP Error 401: Unauthorized"**
→ Token is wrong or expired. Re-generate at tradier.com → API Access.

**"HTTP Error 403: Forbidden"**
→ Your account tier doesn't have access to that endpoint. Try sandbox first.

**"HTTP Error 429: Too Many Requests"**
→ Rate limit hit. The app's 60-second cache should prevent this, but if you're running locally and refreshing frequently, wait 30 seconds.

**App shows yfinance even after adding token**
→ Streamlit Cloud sometimes takes 1-2 minutes to restart. Hard-refresh the browser tab.

---

## Sandbox vs Production

```toml
# For testing (immediately after signup):
TRADIER_TOKEN = "your_sandbox_token"
TRADIER_ENV = "sandbox"

# For live use (after account is funded or developer access granted):
TRADIER_TOKEN = "your_production_token"
TRADIER_ENV = "production"
```

Sandbox tokens and production tokens are **different** — don't mix them.

---

## Future upgrade path

Once Tradier historical IV data becomes available (higher tier or via accumulation):
1. Daily ATM IV stored in a local file (auto-builds over time)
2. True IVR computed from stored history
3. Replaces the current realized vol proxy

See `RESOURCES.md` for the full data source roadmap.

---
*contact@aikungfu.dev | [github.com/gabrielmahia](https://github.com/gabrielmahia)*
