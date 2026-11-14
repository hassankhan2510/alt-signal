# Alt Signal

A daily altcoin **spot** signal (long-only, cash in USDT) that runs for free on GitHub Actions and publishes a page on GitHub Pages.
Research output, **not financial advice**.

## Rules (fixed)
1. **Switch** — ON only when BTC's daily close is above its 50-day average **and** total stablecoin supply grew over the last 7 days (DefiLlama). OFF → stay in USDT.
2. **Universe** — the 30 most-traded altcoins on Binance spot (USDT pairs, last 30 days). BTC, stablecoins, fiat, wrapped, leveraged and tokenized-stock tokens are excluded (stablecoins and stock tokens are detected by behaviour).
3. **Money score** — average rank of: volume share rising (7d vs 30d), net aggressive buying (3d vs 60d normal), money balance (up-day vs down-day volume, 14d).
4. **Size** — full when the alts' share of buy volume (vs BTC) is rising (7d vs 30d), otherwise half.
5. **Model portfolio** — 5 slots; re-check every 2 days; buy the top-5; sell when a coin leaves the top-5, after 3 days, when the switch turns OFF, or at a -20% disaster stop. Fees 0.1% per side.

## Files
- `alt_signal.py` — the whole system (Binance public market data via `data-api.binance.vision`, DefiLlama stablecoins).
- `.github/workflows/daily.yml` — runs every day at 00:20 UTC and commits the results.
- `docs/index.html` — the page (GitHub Pages serves the `docs/` folder). `docs/signal.json`, `docs/history.csv` — machine-readable output and the daily log.
- `state/portfolio.json` — the paper model portfolio (the live forward record).

## Background (research, 2021–2026, survivorship-free, fees included)
Switch + money-score top-5: about +81%/yr with a worst drop around −63% (2025: −24%); the switch alone turned "all alts" from −18%/yr into +33%/yr. Tight stops and take-profits made results worse. Past results do not guarantee future results.
