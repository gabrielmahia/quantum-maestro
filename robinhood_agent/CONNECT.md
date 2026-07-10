# Connecting Claude to Robinhood Agentic Trading

Claude cannot add this connector to itself — it's a user-level setting.

## Claude Desktop / claude.ai (recommended)
1. Settings → Connectors → Add custom connector
2. Name: robinhood-trading
3. URL: https://agent.robinhood.com/mcp/trading
4. Complete the Robinhood OAuth screen when prompted.
5. Start a NEW conversation (a Claude Project is best). Paste the DIRECTIVE
   section of AGENT_DIRECTIVE.md into the project instructions.
6. The Robinhood tools will now appear to Claude in that conversation.

## Claude Code (terminal)
    claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
    # then: /mcp  → select robinhood-trading → authenticate

## Before first session
- Fund the dedicated agentic account ONLY with disposable capital.
- Turn on trade notifications in the Robinhood app.
- Locate the disconnect control (Agentic → Disconnect) BEFORE you need it.
- Phase 1 is ADVISORY: the agent proposes; you place trades manually.
