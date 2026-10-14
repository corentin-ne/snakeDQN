// node web/parity_test.js  -> greedy score of the embedded weights using the page's own JS core
const fs = require("fs"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const core = html.match(/<script id="core">([\s\S]*?)<\/script>/)[1];
const w = JSON.parse(html.match(/<script id="weights" type="application\/json">([\s\S]*?)<\/script>/)[1]);
global.atob = s => Buffer.from(s, "base64").toString("binary");
const m = { exports: {} };
new Function("module", core)(m);
const { SnakeEnv, Net, Trainer, argmax, NF } = m.exports;

const games = +(process.argv[2] || 100);
if (w) {
  const net = new Net(w.in, w.hidden, w.out).load(w.w), env = new SnakeEnv(12, 12), s = new Float32Array(NF);
  const scores = [];
  let t = Date.now(), steps = 0;
  for (let g = 0; g < games; g++) {
    env.reset(); env.features(s);
    while (!env.done) { env.step(argmax(net.forward(s))); env.features(s); steps++; }
    env.done = false; scores.push(env.score);
  }
  const mean = scores.reduce((a, b) => a + b) / games;
  console.log(`pretrained JS: mean ${mean.toFixed(1)} max ${Math.max(...scores)} (${Math.round(steps / ((Date.now() - t) / 1000))} steps/s)`);
}
// in-browser trainer smoke test
const net = new Net(NF, w ? w.hidden : 8), tr = new Trainer(net), env = new SnakeEnv(12, 12);
let s = env.features(), s2 = new Float32Array(NF), scores = [], t = Date.now();
while (tr.steps < +(process.argv[3] || 30000)) {
  let a = Math.random() < tr.eps ? (Math.random() * 3) | 0 : argmax(net.forward(s));
  const r = env.step(a); env.features(s2); tr.observe(s, a, r, s2, env.done); tr.learn();
  if (env.done) { scores.push(env.score); s = env.reset(); } else s.set(s2);
}
const last = scores.slice(-50);
console.log(`trainer: ${tr.steps} steps in ${((Date.now() - t) / 1000).toFixed(1)}s, last50 avg ${(last.reduce((a, b) => a + b, 0) / last.length).toFixed(1)}`);
