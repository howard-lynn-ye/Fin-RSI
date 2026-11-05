// Figure 1, left half: an exploded perspective view of three aligned planes, rendered with CSS 3D
// (the method of the Project Progress Desk teaser). Plane 0 holds the information feeds, plane 1
// the skills, rule cards, checks and investor templates, plane 2 the agent's program. Each plane is
// written as its own transparent PNG, and anchors.json records where chosen items land on screen,
// so compose.py can draw the threads between planes.
//
//   node scene.cjs            writes ../overview/l0.png, l1.png, l2.png and anchors.json
//
// Every name shown is a real adapter, collector, skill, rule card, guard, judge check or API in the
// library. Sparklines are drawn shapes, not vendor prices.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const OUT = path.resolve(__dirname, '../overview');
const P = { rx: 58, rz: -20, gap: 220, persp: 2600, W: 1500, H: 1000, ox: 250, oy: 430, PW: 984, PH: 340 };
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const rnd = (i, a) => { const v = Math.sin(i * 12.9898 + a * 78.233) * 43758.5453; return v - Math.floor(v); };

// ------------------------------------------------------------------ plane 0: information
const spark = {
  split20: 'M2,6 L14,5 L24,7 L34,4 L44,6 L52,5 L53,17 L64,16 L76,17 L88,15 L100,16 L112,15',
  split3: 'M2,7 L14,5 L26,8 L38,6 L50,7 L62,5 L74,6 L75,14 L86,13 L98,15 L112,13',
  rate: 'M2,15 L16,14 L30,13 L44,11 L58,10 L72,8 L86,7 L100,5 L112,4',
  vol: 'M2,10 L10,6 L18,12 L26,4 L34,13 L42,8 L50,15 L58,5 L66,11 L74,7 L82,14 L90,6 L98,10 L106,5 L112,9',
  flat: 'M2,11 L20,10 L38,12 L56,9 L74,11 L92,10 L112,11',
};
const feeds = [
  // id, title, kind (spark name or text), accent, x, y
  ['amzn', 'yfinance · AMZN', 'split20', '#5AA2E6', 330, 318],
  ['form4', 'EDGAR · Form 4', 'insider transactions', '#9AA7B5', 60, 30],
  ['fred', 'FRED · DGS10', 'rate', '#6CC3A0', 440, 40],
  ['gdelt', 'GDELT · events', 'tone, themes, sources', '#E0A458', 640, 26],
  ['tsla', 'yfinance · TSLA', 'split3', '#5AA2E6', 820, 52],
  ['full', 'EDGAR · full text', 'filings search', '#9AA7B5', 130, 140],
  ['ccxt', 'ccxt · BTC/USDT', 'vol', '#5AA2E6', 330, 128],
  ['rss', 'RSS · headlines', 'watchlist feeds', '#E0A458', 520, 150],
  ['f13', 'SEC · 13F', 'holdings, 45-day lag', '#9AA7B5', 715, 136],
  ['cftc', 'CFTC · COT', 'positions by trader type', '#6CC3A0', 40, 250],
  ['googl', 'yfinance · GOOGL', 'split20', '#5AA2E6', 235, 240],
  ['aaii', 'AAII · sentiment', 'weekly survey', '#6CC3A0', 430, 262],
  ['house', 'US House · trades', 'disclosures', '#9AA7B5', 620, 248],
  ['sky', 'Bluesky · posts', 'cashtags', '#E0A458', 810, 262],
  ['umich', 'UMich · sentiment', 'consumer survey', '#6CC3A0', 120, 300],
  ['stooq', 'Stooq · daily bars', 'flat', '#5AA2E6', 560, 290],
  ['fredc', 'FRED · CPIAUCSL', 'rate', '#6CC3A0', 760, 300],
];
const HL = 'amzn';
const FAINT = new Set(['umich', 'stooq', 'fredc', 'sky', 'cftc']);
function feedTile([id, title, kind, accent, x, y], i) {
  const hl = id === HL;
  const dx = (rnd(i, 1) - 0.5) * 50, dy = (rnd(i, 2) - 0.5) * 30, z = (rnd(i, 6) - 0.7) * 110;
  const tilt = 18 + rnd(i, 3) * 30, spin = (rnd(i, 4) - 0.5) * 36, sc = 1.0 + rnd(i, 5) * 0.15;
  const t = hl ? `translate3d(0px,20px,6px) rotateZ(${-P.rz}deg) rotateX(-44deg) scale(1.22)`
               : `translate3d(${dx}px,${dy}px,${z}px) rotateZ(${-P.rz + spin}deg) rotateX(${-tilt}deg) scale(${sc})`;
  const body = spark[kind]
    ? `<svg width="116" height="20" viewBox="0 0 116 20"><path d="${spark[kind]}" fill="none" stroke="${accent}" stroke-width="1.6" stroke-linejoin="round"/>${
        kind === 'split20' && hl ? '<circle cx="52.5" cy="11" r="6.5" fill="none" stroke="#FF8A65" stroke-width="1.4"/><text x="61" y="9" font-size="8" fill="#FF8A65" font-family="DejaVu Sans Mono">20:1</text>' : ''}</svg>`
    : `<div class="t2"><span class="pr">&rsaquo;</span> ${esc(kind)}</div>`;
  return `<div class="tile ${hl ? 'hl' : ''} ${FAINT.has(id) ? 'faint' : ''}" style="left:${x}px;top:${y}px;border-left-color:${accent};transform:${t}">
    <div class="t1">${esc(title)}</div>${body}<i style="background:${accent}"></i>
    <b class="anc" id="t-${id}" style="left:70px;top:22px"></b></div>`;
}

// ------------------------------------------------------------------ plane 1: skills and checks
const rows = [
  { kind: 'skill', y: 112, label: 'skills', items: ['fin-core', 'fin-china', 'fin-crypto', 'fin-macro', 'fin-alt-data', 'fin-credit', 'fin-ml'], on: ['fin-core'] },
  { kind: 'card', y: 166, label: 'rule cards', items: ['split_once', 'execution_lag', 'warmup', 'same_convention', 'feature_causality', 'first_return'], on: ['split_once', 'execution_lag', 'warmup'] },
  { kind: 'check', y: 220, label: 'checks', items: ['adjustment_check', 'execution_lag', 'warmup_probe', 'assert_causal', 'survivorship_audit', 'safe_asof'], on: ['adjustment_check', 'execution_lag', 'warmup_probe'] },
  { kind: 'tmpl', y: 274, label: 'investor templates', items: ['Buffett', 'Soros', 'Simons', 'Druckenmiller', 'Duan Yongping', 'Seykota'], on: ['Buffett'] },
];
const COL = { skill: '#2A78D6', card: '#2A78D6', check: '#EB6834', tmpl: '#C2408B' };
function glassItems() {
  let h = '';
  for (const r of rows) {
    let x = 150;
    for (const it of r.items) {
      const on = r.on.includes(it), c = COL[r.kind];
      const w = Math.max(76, 22 + it.length * 7.6);
      const hlid = (r.kind === 'card' && it === 'split_once') ? 'g-card' : (r.kind === 'check' && it === 'adjustment_check') ? 'g-check' : '';
      const icon = r.kind === 'check' ? (on ? '&#10003;' : '') : r.kind === 'tmpl' ? '&#9679;' : '';
      h += `<div class="it ${r.kind} ${on ? 'on' : ''} ${hlid ? 'hl' : ''}" style="left:${x}px;top:${r.y}px;width:${w}px;--c:${c}">
        ${r.kind === 'check' ? `<span class="box">${icon}</span>` : ''}${r.kind === 'tmpl' ? '<span class="who"></span>' : ''}${esc(it)}
        ${hlid ? `<b class="anc" id="${hlid}" style="left:${w / 2}px;top:14px"></b>` : ''}
        ${on && !hlid ? `<b class="anc" id="o-${r.kind}-${it}" style="left:${w / 2}px;top:14px"></b>` : ''}</div>`;
      x += w + 12;
    }
  }
  return h;
}

// ------------------------------------------------------------------ plane 2: the agent
const code = [
  ['kw', 'from', ' fin_skills.algorithms ', 'kw', 'import', ' positions'],
  ['kw', 'def', ' ', 'fn', 'solve', '(prices, split_factors):'],
  ['cm', '    # playbook: apply each split once'],
  ['kw', '    return', ' positions(', 'st', '"momentum"', ', prices,'],
  ['', '                     ', 'hl', 'split_factors', ', lag=', 'nu', '1', ')'],
];
function codeLines() {
  return code.map((ln, i) => {
    let s = '';
    for (let k = 0; k < ln.length; k++) {
      const tok = ln[k];
      if (['kw', 'fn', 'cm', 'st', 'nu', 'hl'].includes(tok) && k + 1 < ln.length) { s += `<span class="${tok}">${esc(ln[++k])}</span>`; }
      else if (tok !== '') s += esc(tok);
    }
    const hl = i >= 3;
    return `<div class="ln ${hl ? 'hlline' : ''}"><span class="no">${i + 1}</span>${s || '&nbsp;'}${i === 4 ? '<b class="anc" id="c-line" style="left:205px;top:9px"></b>' : ''}</div>`;
  }).join('');
}

const html = `<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent}
#scene{position:relative;width:${P.W}px;height:${P.H}px;perspective:${P.persp}px;perspective-origin:50% 30%}
#stack{position:absolute;left:${P.ox}px;top:${P.oy}px;width:${P.PW}px;height:${P.PH}px;transform-style:preserve-3d;transform:rotateX(${P.rx}deg) rotateZ(${P.rz}deg)}
.plane{position:absolute;left:0;top:0;width:${P.PW}px;height:${P.PH}px;border-radius:12px;transform-style:preserve-3d}
#p0{transform:translateZ(0px);background:radial-gradient(ellipse at 50% 55%, rgba(20,30,45,.18), rgba(0,0,0,0) 70%)}
#p1{transform:translateZ(${P.gap}px);background:rgba(234,240,249,.62);border:1.5px solid rgba(42,120,214,.32);box-sizing:border-box}
#p2{transform:translateZ(${2 * P.gap}px);border-radius:12px}
.tile{position:absolute;box-sizing:border-box;width:150px;height:46px;background:#16202C;border-radius:6px;border-left:4px solid;
  box-shadow:0 6px 14px rgba(0,0,0,.35);transform-origin:50% 50%;padding:4px 8px;font:600 11px/1.35 "DejaVu Sans Mono",monospace;color:#D9E3EE}
.tile.faint{filter:blur(1.3px);opacity:.85}
.tile .t2{font-weight:400;color:#8395A8;font-size:10px;white-space:nowrap}.tile .pr{color:#7FB3E8}
.tile svg{display:block;margin-top:2px}
.tile i{position:absolute;right:8px;top:8px;width:7px;height:7px;border-radius:50%}
.tile.hl{color:#fff;background:#1B3550;box-shadow:0 0 0 2px #5AA2E6, 0 0 22px rgba(90,162,230,.75)}
.p1head,.p2head{position:absolute;left:18px;top:12px;font:600 13px Inter,sans-serif;color:#3F5F86;letter-spacing:.02em}
.rowlab{position:absolute;left:22px;font:600 12px Inter,sans-serif;color:#6A7F98}
.it{position:absolute;height:28px;box-sizing:border-box;border-radius:6px;background:#fff;border:1.3px solid #C8D1DD;
  font:500 12px/26px "DejaVu Sans Mono",monospace;color:#4A5668;padding:0 8px;white-space:nowrap;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.it.skill::before{content:"";display:inline-block;width:9px;height:11px;margin-right:6px;vertical-align:-1px;border:1.3px solid #9FB0C4;border-radius:1.5px}
.it.card{border-style:solid;border-radius:3px}
.it .box{display:inline-block;width:12px;height:12px;line-height:12px;text-align:center;margin-right:6px;border:1.3px solid #AEB8C6;border-radius:2.5px;font-size:10px;vertical-align:-1px;color:#fff}
.it .who{display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:50%;background:#C9D0DA;vertical-align:-1px}
.it.on{border-color:var(--c);color:#1F2335;font-weight:700;box-shadow:0 0 0 1px var(--c)}
.it.on.skill::before{border-color:var(--c);background:var(--c)}
.it.on .box{background:var(--c);border-color:var(--c)}
.it.on .who{background:var(--c)}
.it.hl{box-shadow:0 0 0 2px var(--c), 0 0 18px color-mix(in srgb, var(--c) 70%, transparent)}
.win{position:absolute;background:#fff;border-radius:10px;border:1.2px solid #C9D2DE;box-shadow:0 14px 34px rgba(0,0,0,.22);overflow:hidden}
.bar{height:26px;background:#F1F4F8;border-bottom:1px solid #DCE2EA;font:600 12px/26px Inter,sans-serif;color:#40506A;padding-left:52px;position:relative}
.bar::before{content:"";position:absolute;left:12px;top:9px;width:8px;height:8px;border-radius:50%;background:#E8A0A0;box-shadow:12px 0 #E8CC90,24px 0 #A8D4A8}
.code{padding:8px 0;font:500 13.5px/22px "DejaVu Sans Mono",monospace;color:#1F2335}
.ln{position:relative;padding-left:44px;white-space:pre}.no{position:absolute;left:12px;width:20px;text-align:right;color:#A6B0BE}
.kw{color:#7A3EA8;font-weight:700}.fn{color:#2A63B5;font-weight:700}.cm{color:#2E8B62;font-style:italic}.st{color:#A6532A}.nu{color:#A6532A}
.hlline{background:#EAF2FC}.hlline::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:#2A78D6}
.hl{color:#1556A8;font-weight:700;background:#D6E7FB;border-radius:3px;box-shadow:0 0 0 2px #D6E7FB}
.note{padding:10px 14px;font:500 12.5px/21px Inter,sans-serif;color:#1F2335}
.note .k{display:inline-block;width:18px;height:18px;line-height:18px;border-radius:50%;background:#1BAF7A;color:#fff;font-size:10.5px;font-weight:700;text-align:center;margin-right:8px}
.judge{padding:8px 14px;font:500 12px/22px "DejaVu Sans Mono",monospace;color:#1F2335}
.judge .ok{color:#1BAF7A;font-weight:700;margin-right:4px}
.judge .acc{margin-top:4px;display:inline-block;background:#1BAF7A;color:#fff;border-radius:4px;padding:0 8px;font:700 11.5px/20px Inter,sans-serif}
.anc{position:absolute;width:2px;height:2px;margin:-1px 0 0 -1px}
</style></head><body><div id="scene"><div id="stack">
<div class="plane" id="p0">${feeds.map(feedTile).join('')}</div>
<div class="plane" id="p1">${glassItems()}</div>
<div class="plane" id="p2">
  <div class="win" style="left:0;top:0;width:560px;height:236px"><div class="bar">agent.py — written by your model</div><div class="code">${codeLines()}</div>
    <div class="judge" style="position:absolute;left:0;right:0;bottom:0;border-top:1px solid #E1E6ED;background:#FAFBFC">
      judge&nbsp; <span class="ok">&#10003;</span>finite_output&nbsp; <span class="ok">&#10003;</span>split_representation&nbsp; <span class="ok">&#10003;</span>execution_lag<br>
      <span class="acc">accepted</span></div></div>
  <div class="win" style="left:592px;top:0;width:392px;height:236px"><div class="bar">playbook — PlaybookLearning</div>
    <div class="note"><div><span class="k">1</span>apply split factors once</div><div><span class="k">2</span>decide on the previous close</div>
      <div><span class="k">3</span>warm-up rows hold no position</div>
      <div style="margin-top:6px;color:#6A7F98;font-size:11.5px">learned on development tasks; an edit is kept only if neither dev half gets worse</div></div></div>
</div>
</div></div></body></html>`;

fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(__dirname, 'scene.html'), html);

(async () => {
  const b = await chromium.launch({ executablePath: '/usr/bin/google-chrome' });
  const page = await b.newPage({ viewport: { width: P.W, height: P.H }, deviceScaleFactor: 4 });
  await page.goto('file://' + path.join(__dirname, 'scene.html'));
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(400);
  const anchors = await page.evaluate(([PW, PH]) => {
    const c = el => { const r = el.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; };
    const proj = (pl, x, y) => { const d = document.createElement('div'); d.className = 'anc'; d.style.left = x + 'px'; d.style.top = y + 'px';
      pl.appendChild(d); const v = c(d); d.remove(); return v; };
    const o = { corners: {}, pts: {} };
    for (const id of ['p0', 'p1', 'p2']) {
      const pl = document.getElementById(id);
      o.corners[id] = [[0, 0], [PW, 0], [PW, PH], [0, PH]].map(([x, y]) => proj(pl, x, y));
    }
    for (const e of document.querySelectorAll('.anc[id]')) o.pts[e.id] = c(e);
    const p2 = document.getElementById('p2');
    o.win = [[984, 0], [984, 236], [592, 236], [0, 236], [0, 0]].map(([x, y]) => proj(p2, x, y));
    o.lab2 = proj(document.getElementById('p1'), 14, 150);
    return o;
  }, [P.PW, P.PH]);
  for (const keep of ['p0', 'p1', 'p2']) {
    await page.evaluate(k => { for (const id of ['p0', 'p1', 'p2']) document.getElementById(id).style.visibility = id === k ? 'visible' : 'hidden'; }, keep);
    await page.waitForTimeout(120);
    await page.screenshot({ path: path.join(OUT, `raw_${keep}.png`), omitBackground: true });
  }
  fs.writeFileSync(path.join(__dirname, 'anchors.json'), JSON.stringify({ P, ...anchors }, null, 1));
  console.log('ok', Object.keys(anchors.pts).join(' '));
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
