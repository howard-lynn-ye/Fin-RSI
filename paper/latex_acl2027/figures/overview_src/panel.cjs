// Figure 1, right half: the composed agent laid flat, drawn as the screen a user would set it up on.
// Counts come from ../../numbers.tex (written by make_numbers.py), never typed here.
//
//   node panel.cjs            writes ../overview/panel.png and panel_anchors.json
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const OUT = path.resolve(__dirname, '../overview');
const N = {};
for (const m of fs.readFileSync(path.resolve(__dirname, '../../numbers.tex'), 'utf8').matchAll(/\\newcommand\{\\(\w+)\}\{([^}]*)\}/g)) N[m[1]] = m[2];
const need = ['NMethods', 'NGuards', 'NSkills', 'NPlugins', 'NLensSkills', 'NCards', 'NAlgorithms', 'NTools', 'NTasks', 'BudgetAttempts', 'BudgetCalls', 'BudgetTokens', 'NBridgeLibraries'];
for (const k of need) if (!(k in N)) throw new Error('numbers.tex has no ' + k);
const more = (total, shown) => `+${Number(String(total).replace(/,/g, '')) - shown}`;

const W = 780;
const methods = ['NoImprove', 'SelfDebug', 'CheckFeedback', 'Reflexion', 'PlaybookLearning', 'Team', 'Evolve'];
const checks = [['adjustment_check', 1, 'hl'], ['execution_lag', 1], ['warmup_probe', 1], ['assert_causal', 0], ['survivorship_audit', 0]];
const skills = [['fin-core', 1], ['fin-china', 0], ['fin-crypto', 0], ['fin-macro', 0]];
const tmpls = [['Buffett', 1], ['Soros', 0], ['Simons', 0], ['Duan Yongping', 0]];
const uses = ['MCP server', 'OpenAI tools', 'Anthropic tools', 'Python API', `${N.NBridgeLibraries} bridged libraries`, 'Backtester', 'Benchmark'];
const pill = (t, on, cls, id) => `<span class="pill ${cls} ${on ? 'on' : ''}" ${id ? `id="${id}"` : ''}>${cls === 'chk' ? `<b class="box">${on ? '&#10003;' : ''}</b>` : ''}${cls === 'tpl' ? '<b class="who"></b>' : ''}${cls === 'skl' ? '<b class="doc"></b>' : ''}${t}</span>`;
const html = `<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent}
#ui{position:relative;width:${W}px;box-sizing:border-box;background:#fff;border-radius:12px;overflow:hidden;font-family:Inter,sans-serif;color:#1F2335}
.top{height:46px;display:flex;align-items:center;padding:0 18px;border-bottom:1px solid #E3E7ED;background:#FBFCFD}
.logo{width:22px;height:22px;border-radius:6px;background:#1F2335;display:grid;place-items:center;margin-right:10px}
.logo i{width:10px;height:10px;border:2px solid #fff;border-radius:50%;border-right-color:transparent;transform:rotate(45deg)}
.title{font-weight:700;font-size:16px}
.budget{margin-left:auto;font-size:12.5px;color:#5B6679}
.budget b{font-weight:600;color:#1F2335}
.rows{padding:6px 18px 0}
.row{display:flex;align-items:flex-start;padding:10px 0;border-bottom:1px solid #EEF1F5}
.lab{width:150px;flex:none;padding-top:4px}
.lab .n{font-weight:700;font-size:13.5px}.lab .s{font-size:11.5px;color:#7A8597;margin-top:1px}
.val{flex:1;display:flex;flex-wrap:wrap;gap:6px}
.pill{display:inline-flex;align-items:center;height:26px;padding:0 9px;border-radius:7px;border:1.3px solid #D5DBE4;font:500 12.5px/1 "DejaVu Sans Mono",monospace;color:#4A5668;background:#fff;white-space:nowrap}
.pill.more{border-style:dashed;color:#8A94A6;background:#F7F8FA;font-family:Inter,sans-serif}
.pill .box{width:12px;height:12px;border:1.3px solid #AEB8C6;border-radius:3px;margin-right:6px;font-size:10px;line-height:12px;text-align:center;color:#fff}
.pill .who{width:10px;height:10px;border-radius:50%;background:#C9D0DA;margin-right:6px}
.pill .doc{width:9px;height:11px;border:1.3px solid #A9B6C8;border-radius:1.5px;margin-right:6px}
.met .pill{font-family:Inter,sans-serif;font-weight:600}
.met .pill.on{background:#1BAF7A;border-color:#1BAF7A;color:#fff}
.chk .pill.on,.pill.chk.on{border-color:#EB6834;color:#1F2335;font-weight:700}.pill.chk.on .box{background:#EB6834;border-color:#EB6834}
.pill.skl.on{border-color:#2A78D6;color:#1F2335;font-weight:700}.pill.skl.on .doc{background:#2A78D6;border-color:#2A78D6}
.pill.tpl.on{border-color:#C2408B;color:#1F2335;font-weight:700}.pill.tpl.on .who{background:#C2408B}
.mono{font:600 13px "DejaVu Sans Mono",monospace;padding-top:5px}.mono span{font:400 12px Inter,sans-serif;color:#7A8597;margin-left:10px}
.sub{font-size:11.5px;color:#7A8597;width:100%;margin-top:1px}
.foot{margin-top:8px;height:52px;display:flex;align-items:center;gap:6px;padding:0 18px;background:#F5F7FA;border-top:1px solid #E3E7ED}
.foot .k{font-weight:700;font-size:12.5px;color:#4C6378;margin-right:6px;white-space:nowrap}
.foot .pill{font-family:Inter,sans-serif;font-weight:600;font-size:11.5px;color:#4C6378;border-color:#CBD3DD;height:24px;padding:0 7px}
</style></head><body><div id="ui">
<div class="top"><div class="logo"><i></i></div><div class="title">Your agent</div>
  <div class="budget">per task: <b>${N.BudgetAttempts}</b> runs · <b>${N.BudgetCalls}</b> model calls · <b>${N.BudgetTokens}</b> tokens</div></div>
<div class="rows">
 <div class="row"><div class="lab"><div class="n">Model</div><div class="s">yours</div></div>
   <div class="val mono" id="r-model">generate(messages)<span>any provider, every call metered</span></div></div>
 <div class="row met"><div class="lab"><div class="n">Self-improvement</div><div class="s">pick 1 of ${N.NMethods}</div></div>
   <div class="val" id="r-method">${methods.map(m => pill(m, m === 'PlaybookLearning', '', m === 'PlaybookLearning' ? 'p-method' : '')).join('')}</div></div>
 <div class="row"><div class="lab"><div class="n">Checks</div><div class="s">any of ${N.NGuards}</div></div>
   <div class="val" id="r-checks">${checks.map(([t, on, h]) => pill(t, on, 'chk', h ? 'p-check' : '')).join('')}<span class="pill more">${more(N.NGuards, checks.length)}</span></div></div>
 <div class="row"><div class="lab"><div class="n">Skills</div><div class="s">any of ${N.NSkills}</div></div>
   <div class="val" id="r-skills">${skills.map(([t, on]) => pill(t, on, 'skl', t === 'fin-core' ? 'p-skill' : '')).join('')}<span class="pill more">${more(N.NPlugins, skills.length)} plugins</span>
   <div class="sub">with ${N.NCards} rule cards and ${N.NAlgorithms} algorithms behind them</div></div></div>
 <div class="row" style="border-bottom:0"><div class="lab"><div class="n">Investor templates</div><div class="s">any of ${N.NLensSkills}, off by default</div></div>
   <div class="val" id="r-tmpl">${tmpls.map(([t, on]) => pill(t, on, 'tpl', t === 'Buffett' ? 'p-tmpl' : '')).join('')}<span class="pill more">more</span></div></div>
</div>
<div class="foot" id="r-use"><span class="k">Use it from</span>${uses.map(u => `<span class="pill">${u}</span>`).join('')}</div>
</div></body></html>`;
fs.writeFileSync(path.join(__dirname, 'panel.html'), html);

(async () => {
  const b = await chromium.launch({ executablePath: '/usr/bin/google-chrome' });
  const page = await b.newPage({ viewport: { width: W, height: 700 }, deviceScaleFactor: 4 });
  await page.goto('file://' + path.join(__dirname, 'panel.html'));
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(300);
  const boxes = await page.evaluate(() => {
    const o = {};
    for (const e of document.querySelectorAll('[id]')) { const r = e.getBoundingClientRect(); o[e.id] = [r.x, r.y, r.width, r.height]; }
    return o;
  });
  await page.locator('#ui').screenshot({ path: path.join(OUT, 'panel.png'), omitBackground: true });
  fs.writeFileSync(path.join(__dirname, 'panel_anchors.json'), JSON.stringify({ W, H: boxes.ui[3], boxes }, null, 1));
  console.log('ok', Object.keys(boxes).join(' '));
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
