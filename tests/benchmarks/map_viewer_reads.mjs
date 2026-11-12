// Time to the first drawn map and bytes fetched by the map viewer, cold and warm.
// Needs Google Chrome and `npm install playwright-core`; run against a live instance:
//   [LATENCY_MS=100] [MBPS=10] node map_viewer_reads.mjs http://127.0.0.1:9000 5 <dataset id>...
// To compare two read paths, run it against an instance of each. Not collected by pytest.
// Each run: a fresh browser context (cold cache), then a reload in it (warm cache). A view has
// loaded when its data requests (to /icechunk or /zarr) have started and then stopped.
import { chromium } from "playwright-core";

const [base, runsArg, ...datasets] = process.argv.slice(2);
const runs = Number(runsArg);
const QUIET_MS = 750;
const TIMEOUT_MS = 60_000;

const browser = await chromium.launch({ channel: "chrome", headless: true });

async function load(page, url, traffic) {
  traffic.length = 0;
  await page.goto(url, { waitUntil: "load" });
  const began = Date.now();
  let stableSince = null;
  let lastCount = 0;
  while (Date.now() - began < TIMEOUT_MS) {
    const pending = traffic.some((t) => !t.done);
    if (traffic.length === 0 || pending || traffic.length !== lastCount) {
      lastCount = traffic.length;
      stableSince = Date.now();
    } else if (Date.now() - stableSince >= QUIET_MS) {
      break;
    }
    await new Promise((r) => setTimeout(r, 50));
  }
  if (traffic.length === 0) throw new Error(`no data requested for ${url}`);
  const done = traffic.filter((t) => t.done);
  return {
    lastData: Math.max(...done.map((t) => t.end)), // ms since navigation
    requests: done.length,
    fromCache: done.filter((t) => t.cached).length,
    bytes: done.reduce((sum, t) => sum + t.bytes, 0),
    failed: done.filter((t) => t.failed).length,
  };
}

async function run(dataset) {
  const context = await browser.newContext();
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  // LATENCY_MS emulates a remote server, added to every request; MBPS limits the download
  // bandwidth, which is where a large manifest shows. Unset, both are unlimited.
  const latency = Number(process.env.LATENCY_MS ?? 0);
  const download = process.env.MBPS ? (Number(process.env.MBPS) * 1e6) / 8 : -1;
  if (latency || download > 0) {
    await cdp.send("Network.emulateNetworkConditions", {
      offline: false,
      latency,
      downloadThroughput: download,
      uploadThroughput: -1,
    });
  }
  const traffic = [];
  const byId = new Map();
  let navStart = 0;
  cdp.on("Network.requestWillBeSent", (e) => {
    if (e.type === "Document") navStart = e.timestamp;
    if (!/\/(zarr|icechunk)\//.test(e.request.url)) return;
    const entry = { done: false, bytes: 0, cached: false, failed: false, end: 0 };
    byId.set(e.requestId, entry);
    traffic.push(entry);
  });
  cdp.on("Network.requestServedFromCache", (e) => {
    const entry = byId.get(e.requestId);
    if (entry) entry.cached = true;
  });
  cdp.on("Network.responseReceived", (e) => {
    const entry = byId.get(e.requestId);
    if (entry && e.response.fromDiskCache) entry.cached = true;
  });
  const finish = (failed) => (e) => {
    const entry = byId.get(e.requestId);
    if (!entry) return;
    Object.assign(entry, { done: true, failed, end: (e.timestamp - navStart) * 1000 });
    if (!failed) entry.bytes = e.encodedDataLength;
  };
  cdp.on("Network.loadingFinished", finish(false));
  cdp.on("Network.loadingFailed", finish(true));
  const url = `${base}/map?dataset=${dataset}`;
  const cold = await load(page, url, traffic);
  const warm = await load(page, url, traffic);
  await context.close();
  return { cold, warm };
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

for (const dataset of datasets) {
  const results = [];
  for (let i = 0; i < runs; i++) results.push(await run(dataset));
  for (const phase of ["cold", "warm"]) {
    const rs = results.map((r) => r[phase]);
    const field = (name) => median(rs.map((r) => r[name]));
    const failed = rs.reduce((sum, r) => sum + r.failed, 0);
    console.log(
      `${dataset.padEnd(40)} ${phase}  data done ${field("lastData").toFixed(0)} ms, ` +
        `${field("requests")} req (${field("fromCache")} cached), ${(field("bytes") / 1024).toFixed(0)} KB` +
        (failed ? `, ${failed} failed` : ""),
    );
  }
}
await browser.close();
