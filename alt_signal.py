"""ALT SIGNAL — daily altcoin spot system (research, not financial advice). Runs on GitHub Actions after the UTC daily close.
Rules (fixed, from STATUS 9.29-9.37 research, survivorship-free 2021-2026 tests):
  SWITCH  ON only if BTC daily close > its 50-day average AND total stablecoin supply grew over the last 7 days
          (DefiLlama, yesterday's reading). OFF -> everything in USDT.
  UNIVERSE  the 30 most-traded ALTS on Binance spot (USDT pairs, last 30 days' USDT volume). BTC, stablecoins, fiat,
          wrapped and leveraged tokens excluded. ETH counts as an alt.
  MONEY SCORE (rank 0-1 within the 30, average of three):
          volume share  : coin's share of the 30 alts' volume, last 7 days vs its 30-day average share
          net buying    : (aggressive buy $ - aggressive sell $) / volume over 3 days, vs its 60-day normal
          money balance : volume on up-days minus down-days over 14 days / total volume
  SIZE    full if the alts' share of BUY volume (vs BTC) is higher over 7 days than over 30 days, else half
  MODEL PORTFOLIO (paper): 5 slots, re-check every 2 days: buy the top-5 not held; sell a coin when it leaves the top-5,
          after 3 days held, when the switch turns OFF, or at the -20% disaster stop. Fees 0.1% per side.
Outputs (run: python alt_signal.py): docs/index.html (the page), docs/signal.json, docs/history.csv, state/portfolio.json
"""
import json, os, math, time, datetime as dt, urllib.request
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
API = 'https://data-api.binance.vision/api/v3'                 # Binance public market data (works from GitHub's US servers)
STABLES = 'https://stablecoins.llama.fi/stablecoincharts/all'
FEE, SLOTS, MAX_HOLD, STOP = 0.001, 5, 3, 0.20
EXCLUDE = {'BTC', 'USDC', 'FDUSD', 'TUSD', 'BUSD', 'USDP', 'DAI', 'EUR', 'TRY', 'BRL', 'GBP', 'AUD', 'USD1', 'RLUSD', 'USDE', 'PYUSD',
           'XUSD', 'BFUSD', 'AEUR', 'EURI', 'WBTC', 'WBETH', 'BETH', 'STETH', 'PAXG', 'XAUT', 'USTC', 'UST', 'SUSD', 'USDS', 'FRAX'}

def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'alt-signal'}), timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(3 + 5 * k)

def klines(sym, n=100):
    k = get(f'{API}/klines?symbol={sym}&interval=1d&limit={n}')
    d = pd.DataFrame(k, columns=['t', 'o', 'h', 'l', 'c', 'v', 'ct', 'qv', 'n', 'tbv', 'tbq', 'x'])
    for col in ('o', 'h', 'l', 'c', 'qv', 'tbq'): d[col] = d[col].astype(float)
    d['date'] = pd.to_datetime(d.t, unit='ms'); d = d.set_index('date')
    now_ms = int(time.time() * 1000)
    return d[d.ct < now_ms][['o', 'h', 'l', 'c', 'qv', 'tbq']]                    # complete days only

def main():
    # ---------------- data
    tick = get(f'{API}/ticker/24hr')
    cands = []
    for t in tick:
        s = t['symbol']
        if not s.endswith('USDT'): continue
        base = s[:-4]
        if base in EXCLUDE or base.endswith(('UP', 'DOWN', 'BULL', 'BEAR')) or float(t['quoteVolume']) <= 0: continue
        cands.append((float(t['quoteVolume']), base))
    cands = [b for _, b in sorted(cands, reverse=True)[:60]]
    K = {}
    for b in cands + ['BTC']:
        try: K[b] = klines(f'{b}USDT')
        except Exception: pass
    days = K['BTC'].index; D = days[-1]                                                 # last COMPLETE UTC day
    C = pd.DataFrame({b: k.c for b, k in K.items()}).reindex(days); L = pd.DataFrame({b: k.l for b, k in K.items()}).reindex(days)
    QV = pd.DataFrame({b: k.qv for b, k in K.items()}).reindex(days); TB = pd.DataFrame({b: k.tbq for b, k in K.items()}).reindex(days)
    alts = [b for b in K if b != 'BTC' and C[b].notna().sum() >= 60]
    # behaviour filters (robust to new listings): stablecoins barely move; tokenized STOCKS (Binance names end in 'B') barely move on weekends
    r_ = C[alts].pct_change(fill_method=None).iloc[-60:]; wkend = r_.index.dayofweek >= 5
    excluded = {}
    for b in list(alts):
        sd = r_[b].iloc[-30:].std(); ratio = r_[b][wkend].abs().mean() / max(r_[b][~wkend].abs().mean(), 1e-9)
        if not np.isfinite(sd) or sd < 0.006: excluded[b] = 'stablecoin (price barely moves)'
        elif b.endswith('B') and len(b) >= 4 and ratio < 0.45: excluded[b] = f'tokenized stock (name ends in B, weekend moves {ratio * 100:.0f}% of weekday)'
    alts = [b for b in alts if b not in excluded]
    vol30 = QV[alts].iloc[-30:].sum().sort_values(ascending=False); U = list(vol30.index[:30])
    # ---------------- switch
    btc, ma50 = C['BTC'].iloc[-1], C['BTC'].iloc[-50:].mean()
    st = get(STABLES); ST = pd.Series({pd.to_datetime(int(x['date']), unit='s').normalize(): x['totalCirculatingUSD'].get('peggedUSD', np.nan) for x in st}).sort_index()
    ST = ST[ST.index <= D - pd.Timedelta(days=1)]                                      # yesterday's reading (as tested)
    sg7 = math.log(ST.iloc[-1] / ST.asof(ST.index[-1] - pd.Timedelta(days=7)))
    ON = bool(btc > ma50 and sg7 > 0)
    # ---------------- money score
    R = C[U].pct_change(fill_method=None)
    sh = QV[U].rolling(7).sum().div(QV[U].rolling(7).sum().sum(1), axis=0)
    m1 = np.log(sh.iloc[-1] / sh.iloc[-30:].mean())
    nb = (2 * TB[U] - QV[U]).rolling(3).sum() / QV[U].rolling(3).sum(); m2 = nb.iloc[-1] - nb.iloc[-60:].mean()
    m3 = (np.sign(R) * QV[U]).iloc[-14:].sum() / QV[U].iloc[-14:].sum()
    ranks = pd.DataFrame({'volume share': m1.rank(pct=True), 'net buying': m2.rank(pct=True), 'money balance': m3.rank(pct=True)})
    ranks['money score'] = ranks.mean(1); ranks = ranks.sort_values('money score', ascending=False)
    top5 = list(ranks.index[:SLOTS])
    tb_alt = TB[U].sum(1); bshare = tb_alt / (tb_alt + TB['BTC'])
    size_full = bool(tb_alt.iloc[-7:].sum() / (tb_alt.iloc[-7:].sum() + TB['BTC'].iloc[-7:].sum()) > tb_alt.iloc[-30:].sum() / (tb_alt.iloc[-30:].sum() + TB['BTC'].iloc[-30:].sum()))
    size = 1.0 if size_full else 0.5
    # ---------------- model portfolio (paper)
    sp = os.path.join(ROOT, 'state', 'portfolio.json'); os.makedirs(os.path.dirname(sp), exist_ok=True)
    S = json.load(open(sp)) if os.path.exists(sp) else {'start': str(D.date()), 'equity': 1.0, 'cash': 1.0, 'holdings': {}, 'last_day': None, 'log': [], 'curve': []}
    dstr = str(D.date()); actions = []
    if S['last_day'] != dstr:
        recheck = (D - pd.Timestamp('1970-01-01')).days % 2 == 0
        # mark to market + disaster stop
        for b, h in list(S['holdings'].items()):
            px = C[b].iloc[-1] if b in C else np.nan; lo = L[b].iloc[-1] if b in L else np.nan
            if not np.isfinite(px):                                                       # delisted / no data: exit at last price
                S['cash'] += h['value'] * (1 - FEE); actions.append(f"SELL {b} (no data)"); del S['holdings'][b]; continue
            if np.isfinite(lo) and lo <= h['entry'] * (1 - STOP):
                exit_px = h['entry'] * (1 - STOP); h['value'] *= exit_px / h['last']; S['cash'] += h['value'] * (1 - FEE)
                actions.append(f"SELL {b} — disaster stop -20%"); del S['holdings'][b]; continue
            h['value'] *= px / h['last']; h['last'] = px; h['days'] += 1
        def sell(b, why):
            h = S['holdings'].pop(b); S['cash'] += h['value'] * (1 - FEE); actions.append(f"SELL {b} — {why} ({(h['last'] / h['entry'] - 1) * 100:+.1f}%)")
        for b in list(S['holdings']):
            if not ON: sell(b, 'switch OFF')
            elif recheck and b not in top5: sell(b, 'left the top-5')
            elif S['holdings'][b]['days'] >= MAX_HOLD: sell(b, f'held {MAX_HOLD} days')
        equity = S['cash'] + sum(h['value'] for h in S['holdings'].values())
        if ON and recheck:
            for b in top5:
                if b in S['holdings'] or len(S['holdings']) >= SLOTS: continue
                amt = min(S['cash'], equity / SLOTS * size)
                if amt <= equity * 0.01: continue
                S['cash'] -= amt; px = C[b].iloc[-1]
                S['holdings'][b] = {'entry': px, 'last': px, 'value': amt * (1 - FEE), 'date': dstr, 'days': 0}
                actions.append(f"BUY {b} at {px:.6g} ({'full' if size == 1 else 'half'} size)")
        S['equity'] = S['cash'] + sum(h['value'] for h in S['holdings'].values()); S['last_day'] = dstr
        S['curve'].append([dstr, round(S['equity'], 6), round(float(C['BTC'].iloc[-1]), 2)])
        for a in actions: S['log'].append([dstr, a])
        S['log'] = S['log'][-200:]
        json.dump(S, open(sp, 'w'), indent=1)
    # ---------------- outputs
    out = dict(date=dstr, switch='ON' if ON else 'OFF', btc=round(float(btc), 2), btc_ma50=round(float(ma50), 2), stablecoins_7d_pct=round(sg7 * 100, 3),
               size='full' if size_full else 'half', top5=top5, ranking={b: {k: round(float(v), 3) for k, v in ranks.loc[b].items()} for b in ranks.index[:10]},
               universe=sorted(U), excluded=excluded, model_equity=round(S['equity'], 4), holdings=S['holdings'], actions_today=actions,
               generated_utc=dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M'))
    os.makedirs(os.path.join(ROOT, 'docs'), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, 'docs', 'signal.json'), 'w'), indent=1)
    hp = os.path.join(ROOT, 'docs', 'history.csv'); new = not os.path.exists(hp)
    with open(hp, 'a') as f:
        if new: f.write('date,switch,size,btc,btc_ma50,stablecoins_7d_pct,top5,model_equity\n')
        f.write(f"{dstr},{out['switch']},{out['size']},{out['btc']},{out['btc_ma50']},{out['stablecoins_7d_pct']},{'|'.join(top5)},{out['model_equity']}\n")
    open(os.path.join(ROOT, 'docs', 'index.html'), 'w', encoding='utf-8').write(page(out, ranks, S))
    print(json.dumps({k: out[k] for k in ('date', 'switch', 'size', 'top5', 'model_equity', 'actions_today')}, indent=1))

def page(o, ranks, S):
    on = o['switch'] == 'ON'
    rows = ''.join(f"<tr><td>{i + 1}</td><td><b>{b}</b></td><td>{r['volume share']:.2f}</td><td>{r['net buying']:.2f}</td><td>{r['money balance']:.2f}</td><td><b>{r['money score']:.2f}</b></td></tr>"
                   for i, (b, r) in enumerate(ranks.head(10).iterrows()))
    hold = ''.join(f"<tr><td><b>{b}</b></td><td>{h['date']}</td><td>{h['days']}</td><td>{h['entry']:.6g}</td><td>{(h['last'] / h['entry'] - 1) * 100:+.1f}%</td><td>{h['entry'] * (1 - STOP):.6g}</td></tr>"
                   for b, h in S['holdings'].items()) or '<tr><td colspan="6">No positions — 100% USDT</td></tr>'
    log = ''.join(f"<li><span>{d}</span> {a}</li>" for d, a in reversed(S['log'][-25:])) or '<li>No trades yet.</li>'
    cv = S['curve'][-180:]; svg = ''
    if len(cv) >= 2:
        ys = [c[1] for c in cv]; lo, hi = min(ys), max(ys); hi = hi if hi > lo else lo + 1e-9
        pts = ' '.join(f"{i / (len(cv) - 1) * 600:.1f},{110 - (y - lo) / (hi - lo) * 100:.1f}" for i, y in enumerate(ys))
        svg = f'<svg viewBox="0 0 600 120" class="curve" role="img" aria-label="model portfolio equity"><polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="2.5"/></svg>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alt Signal</title><style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#14181f;--mute:#5d6673;--line:#e3e6eb;--accent:#2f6fde;--on:#17864a;--off:#b4331f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1217;--card:#171b22;--ink:#e8ebf0;--mute:#9aa3af;--line:#262c36;--accent:#6ea0ff;--on:#3fbf7a;--off:#ff7a64}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:860px;margin:0 auto;padding:20px 16px 48px}}h1{{font-size:22px;margin:0 0 4px}}.sub{{color:var(--mute);font-size:13px;margin-bottom:18px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}}
.state{{font-size:30px;font-weight:700;color:{'var(--on)' if on else 'var(--off)'}}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:10px}}
.k{{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}.v{{font-size:18px;font-weight:600}}
.ok{{color:var(--on)}}.no{{color:var(--off)}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:7px 6px;border-bottom:1px solid var(--line)}}
th{{color:var(--mute);font-weight:600;font-size:12px}}.scroll{{overflow-x:auto}}h2{{font-size:16px;margin:0 0 10px}}ul{{margin:0;padding-left:18px}}li span{{color:var(--mute);margin-right:6px}}
.curve{{width:100%;height:120px}}.note{{color:var(--mute);font-size:12px}}.picks b{{display:inline-block;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:4px 10px;margin:3px 4px 0 0}}
</style></head><body><main>
<h1>Alt Signal</h1><div class="sub">Daily altcoin spot system · data up to the {o['date']} UTC close · updated {o['generated_utc']} UTC</div>
<div class="card"><div class="k">Switch</div><div class="state">{'ON — buy alts' if on else 'OFF — stay in USDT'}</div>
<div class="grid"><div><div class="k">BTC vs 50-day avg</div><div class="v {'ok' if o['btc'] > o['btc_ma50'] else 'no'}">{o['btc']:,.0f} vs {o['btc_ma50']:,.0f}</div></div>
<div><div class="k">Stablecoins, 7 days</div><div class="v {'ok' if o['stablecoins_7d_pct'] > 0 else 'no'}">{o['stablecoins_7d_pct']:+.2f}%</div></div>
<div><div class="k">Size</div><div class="v">{'Full' if o['size'] == 'full' else 'Half'}</div></div></div></div>
<div class="card"><h2>Top 5 by money score{'' if on else ' (not bought while OFF)'}</h2><div class="picks">{''.join(f'<b>{b}</b>' for b in o['top5'])}</div></div>
<div class="card"><h2>Money-score ranking (top 10 of the 30 most-traded alts)</h2><div class="scroll"><table><tr><th>#</th><th>Coin</th><th>Volume share</th><th>Net buying</th><th>Money balance</th><th>Score</th></tr>{rows}</table></div>
<p class="note">Each part is a rank from 0 (worst) to 1 (best) among the 30 alts. Score = average of the three.
Excluded automatically: {', '.join(f'{b} ({w})' for b, w in o['excluded'].items()) or 'none'}.</p></div>
<div class="card"><h2>Model portfolio (paper) · value {S['equity']:.3f} since {S['start']} (started at 1.000)</h2>{svg}
<div class="scroll"><table><tr><th>Coin</th><th>Bought</th><th>Days</th><th>Entry</th><th>Now</th><th>Safety stop (-20%)</th></tr>{hold}</table></div></div>
<div class="card"><h2>Recent actions</h2><ul>{log}</ul></div>
<div class="card note"><b>Rules:</b> switch ON = BTC above its 50-day average AND stablecoin supply up over 7 days. When ON, hold the top 5 alts by money score,
re-check every 2 days, sell when a coin leaves the top 5, after 3 days, when the switch turns OFF, or at -20%. Full size when alts take a rising share of buy volume, else half.
Research output from survivorship-free 2021–2026 tests — not financial advice. Past results do not guarantee future results.</div>
</main></body></html>"""

if __name__ == '__main__':
    main()
