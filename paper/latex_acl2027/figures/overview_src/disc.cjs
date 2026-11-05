// Figure 1: the library as a dial. A tilted disc with the user's model at its centre and four rings
// of pins around it: the self-improvement methods, the checks, the skills and the investor
// templates. Pin counts are the real counts from ../../numbers.tex. The chosen pieces stand up as
// lit pegs; compose_disc.py draws the threads from them to the centre, the pedestal, the
// information coming in and the interfaces going out.
//
//   node disc.cjs        writes ../overview/disc.png and disc_anchors.json
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const OUT = path.resolve(__dirname, '../overview');
const N = {};
for (const m of fs.readFileSync(path.resolve(__dirname, '../../numbers.tex'), 'utf8').matchAll(/\\newcommand\{\\(\w+)\}\{([^}]*)\}/g)) N[m[1]] = m[2];
const num = k => { if (!(k in N)) throw new Error('numbers.tex has no ' + k); return Number(String(N[k]).replace(/,/g, '')); };

const V = { W: 1600, H: 900, rx: 63, persp: 2600 };
const D = 1000, C = D / 2, RIM = 486;
// ring: name, count, radius, pin radius, colour, lit angles (degrees, 90 = nearest the viewer), peg height
const rings = [
  ['method', num('NMethods'), 138, 12, '#1BAF7A', [52], 70],
  ['check', num('NGuards'), 222, 7.5, '#EB6834', [30, 55, 134], 58],
  ['skill', num('NSkills'), 326, 4.6, '#2A78D6', [40, 66, 144], 50],
  ['template', num('NLensSkills'), 436, 5.6, '#C2408B', [120], 54],
];
const rad = d => d * Math.PI / 180;
let svg = `<svg width="${D}" height="${D}" viewBox="0 0 ${D} ${D}" xmlns="http://www.w3.org/2000/svg">
<defs><radialGradient id="g" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#FFFFFF"/><stop offset="0.82" stop-color="#F3F6FA"/><stop offset="1" stop-color="#E7ECF3"/></radialGradient></defs>
<circle cx="${C}" cy="${C}" r="${RIM}" fill="url(#g)" stroke="#CDD5E0" stroke-width="3"/>`;
const pegs = [];
for (const [name, n, r, pr, col, lit, h] of rings) {
  svg += `<circle cx="${C}" cy="${C}" r="${r}" fill="none" stroke="${col}" stroke-opacity="0.22" stroke-width="${pr * 3.2}"/>`;
  const off = lit[0];
  const chosen = new Set(lit.map(a => ((Math.round((a - off) / (360 / n)) % n) + n) % n));
  for (let k = 0; k < n; k++) {
    const a = rad(off + k * 360 / n), x = C + r * Math.cos(a), y = C + r * Math.sin(a);
    if (chosen.has(k)) { svg += `<circle cx="${x}" cy="${y}" r="${pr * 3.4 + 8}" fill="${col}" opacity="0.16"/><circle cx="${x}" cy="${y}" r="${pr + 3}" fill="${col}"/>`; pegs.push({ name, k, x, y, pr, col, h }); }
    else svg += `<circle cx="${x}" cy="${y}" r="${pr}" fill="#BCC5D2"/>`;
  }
}
svg += `</svg>`;

const pegHtml = pegs.map((p, i) => {
  const w = Math.max(13, p.pr * 2 + 6);
  return `<div class="peg" style="left:${p.x}px;top:${p.y}px;width:${w}px;height:${p.h}px;--c:${p.col};margin-left:${-w / 2}px;margin-top:${-p.h}px">
    <b class="anc" id="peg-${i}" style="left:${w / 2}px;top:0px"></b></div>`;
}).join('');

const html = `<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent}
#scene{position:relative;width:${V.W}px;height:${V.H}px;perspective:${V.persp}px;perspective-origin:50% 20%}
#disc{position:absolute;left:${(V.W - D) / 2}px;top:${(V.H - D) / 2 + 40}px;width:${D}px;height:${D}px;transform-style:preserve-3d;transform:rotateX(${V.rx}deg)}
#disc svg{position:absolute;left:0;top:0}
.peg{position:absolute;transform-origin:50% 100%;transform:rotateX(-90deg);border-radius:${'999px 999px 3px 3px'};
  background:linear-gradient(to right, color-mix(in srgb, var(--c) 70%, white) 0%, var(--c) 55%, color-mix(in srgb, var(--c) 80%, black) 100%)}
.anc{position:absolute;width:2px;height:2px;margin:-1px 0 0 -1px}
.mk{position:absolute;width:2px;height:2px;margin:-1px 0 0 -1px}
</style></head><body><div id="scene"><div id="disc">${svg}${pegHtml}</div></div></body></html>`;
fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(__dirname, 'disc.html'), html);

(async () => {
  const b = await chromium.launch({ executablePath: '/usr/bin/google-chrome' });
  const page = await b.newPage({ viewport: { width: V.W, height: V.H }, deviceScaleFactor: 4 });
  await page.goto('file://' + path.join(__dirname, 'disc.html'));
  await page.waitForTimeout(300);
  const a = await page.evaluate(([C, RIM, rings]) => {
    const disc = document.getElementById('disc');
    const c = el => { const r = el.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; };
    const proj = (x, y) => { const d = document.createElement('div'); d.className = 'mk'; d.style.left = x + 'px'; d.style.top = y + 'px';
      disc.appendChild(d); const v = c(d); d.remove(); return v; };
    const polar = (r, deg) => proj(C + r * Math.cos(deg * Math.PI / 180), C + r * Math.sin(deg * Math.PI / 180));
    const o = { centre: proj(C, C), rim: {}, ring: {}, pegs: {} };
    for (const d of [0, 90, 180, 270, 150, 210, 330, 30]) o.rim[d] = polar(RIM, d);
    o.axis = { x: polar(76, 0), y: polar(76, 90), yb: polar(76, 270) };
    for (const [name, , r] of rings) o.ring[name] = { r, label: polar(r, 148), left: polar(r, 180), front: polar(r, 90) };
    for (const e of document.querySelectorAll('.anc[id]')) o.pegs[e.id] = c(e);
    return o;
  }, [C, RIM, rings]);
  a.pegs_meta = pegs.map((p, i) => ({ id: `peg-${i}`, ring: p.name, col: p.col }));
  await page.screenshot({ path: path.join(OUT, 'raw_disc.png'), omitBackground: true });
  fs.writeFileSync(path.join(__dirname, 'disc_anchors.json'), JSON.stringify({ V, ...a }, null, 1));
  console.log('ok pegs', pegs.length, 'pins', rings.map(r => r[1]).join('/'));
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
