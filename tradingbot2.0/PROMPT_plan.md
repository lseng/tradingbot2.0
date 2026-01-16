0a. Study `specs/*` with up to 250 parallel Sonnet subagents to learn the application specifications.
0b. Study @IMPLEMENTATION_PLAN.md (if present) to understand the plan so far.
0c. Study `src/lib/*` with up to 250 parallel Sonnet subagents to understand shared utilities & components.
0d. For reference, the application source code is in `src/*`.

1. Study @IMPLEMENTATION_PLAN.md (if present; it may be incorrect) and use up to 500 Sonnet subagents to study existing source code in `src/*` and compare it against `specs/*`. Use an Opus subagent to analyze findings, prioritize tasks, and create/update @IMPLEMENTATION_PLAN.md as a bullet point list sorted in priority of items yet to be implemented. Ultrathink. Consider searching for TODO, minimal implementations, placeholders, skipped/flaky tests, and inconsistent patterns. Study @IMPLEMENTATION_PLAN.md to determine starting point for research and keep it up to date with items considered complete/incomplete using subagents.

IMPORTANT: Plan only. Do NOT implement anything. Do NOT assume functionality is missing; confirm with code search first. Treat `src/lib` as the project's standard library for shared utilities and components. Prefer consolidated, idiomatic implementations there over ad-hoc copies.

ULTIMATE GOAL: Build a profitable MES futures day trading scalper with the following requirements:

1. DAY TRADING ONLY: Flatten all positions by 4:30 PM NY time (21:30 UTC). No overnight positions.

2. CAPITAL PROTECTION: Starting capital is $1,000. The account CANNOT be blown up. Implement strict risk management with max 5% daily loss limit, 2.5% risk per trade, and circuit breakers.

3. POSITION MANAGEMENT: Trade only one position at a time. Scale contracts based on confidence and available capital (start with 1 contract, scale up as profitable).

4. ML MODEL: Train on 3 years of 1-second MES data (33M rows in data/historical/MES/MES_1s_2years.parquet) to predict price action for scalping entries/exits. Use walk-forward validation to prevent overfitting.

5. STRATEGY OPTIMIZATION: Determine optimal R:R ratios, whether to adjust stops/targets during trades, and whether to reverse positions. No limit on number of trades.

6. PROFITABILITY: The bot must consistently generate high returns through disciplined scalping with proper risk management.

Consider missing elements and plan accordingly. If an element is missing, search first to confirm it doesn't exist, then if needed author the specification at specs/FILENAME.md. If you create a new element then document the plan to implement it in @IMPLEMENTATION_PLAN.md using a subagent.
