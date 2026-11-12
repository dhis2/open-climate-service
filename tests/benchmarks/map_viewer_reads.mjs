// Time to first drawn map and bytes fetched, /zarr against native Icechunk, in the map viewer.
// Needs Google Chrome and `npm install playwright-core`; run against a live instance:
//   [LATENCY_MS=100] node map_viewer_reads.mjs http://127.0.0.1:9000 5 <dataset id>...
// Not collected by pytest; run by hand.
// Each run: a fresh browser context (cold cache), then a reload in it (warm cache).
import { chromium } from "playwright-core";

const [base, runsArg, ...datasets] = process.argv.slice(2);
const runs = Number(runsArg);
const QUIET_MS = 750; // no data request in flight for this long: the view has finished loading
const TIMEOUT_MS = 60_000;

const browser = await chromium.launch({ channel: "chrome", headless: true });

async function load(page, url, traffic) {
  traffic.length = 0;
  await page.goto(url, { waitUntil: "load" });
  const began = Date.now();
  // Ready: the layer committed a level (or failed).
  await page.waitForFunction(() => window.trialTiming?.ready !== null || window.trialTiming?.error !== null, null, {
    timeout: TIMEOUT_MS,
  });
  // Then wait for the data requests to stop.
  let lastCount = -1;
  let stableSince = Date.now();
  while (Date.now() - began < TIMEOUT_MS) {
    const inFlight = traffic.filter((t) => t.done === false).length;
    if (traffic.length !== lastCount || inFlight > 0) {
      lastCount = traffic.length;
      stableSince = Date.now();
    } else if (Date.now() - stableSince >= QUIET_MS) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  const timing = await page.evaluate(() => window.trialTiming);
  const data = traffic.filter((t) => t.done);
  const lastData = Math.max(0, ...data.map((t) => t.end));
  return {
    ready: timing.ready,
    drawn: timing.drawn,
    lastData, // ms since navigation when the last data response finished
    requests: data.length,
    fromCache: data.filter((t) => t.cached).length,
    bytes: data.reduce((sum, t) => sum + t.bytes, 0),
    error: timing.error,
  };
}

async function run(dataset, reader) {
  const context = await browser.newContext();
  const page = await context.newPage();
  const traffic = [];
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  // LATENCY_MS emulates a remote server: added to every request, bandwidth left unlimited.
  const latency = Number(process.env.LATENCY_MS ?? 0);
  if (latency) await cdp.send("Network.emulateNetworkConditions", { offline: false, latency, downloadThroughput: -1, uploadThroughput: -1 });
  const byId = new Map();
  let navStart = 0;
  cdp.on("Network.requestWillBeSent", (e) => {
    if (e.type === "Document") navStart = e.timestamp;
    if (!/\/(zarr|icechunk)\//.test(e.request.url)) return;
    const entry = { url: e.request.url, done: false, bytes: 0, cached: false, end: 0 };
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
  cdp.on("Network.loadingFinished", (e) => {
    const entry = byId.get(e.requestId);
    if (!entry) return;
    entry.done = true;
    entry.bytes = e.encodedDataLength;
    entry.end = (e.timestamp - navStart) * 1000;
  });
  cdp.on("Network.loadingFailed", (e) => {
    const entry = byId.get(e.requestId);
    if (!entry) return;
    entry.done = true;
    entry.end = (e.timestamp - navStart) * 1000;
  });
  const url = `${base}/map?dataset=${dataset}&reader=${reader}`;
  const cold = await load(page, url, traffic);
  const warm = await load(page, url, traffic);
  await context.close();
  return { cold, warm };
}

const fmt = (r) =>
  `ready ${r.ready?.toFixed(0) ?? "-"} ms, data done ${r.lastData.toFixed(0)} ms, ` +
  `${r.requests} req (${r.fromCache} cached), ${(r.bytes / 1024).toFixed(0)} KB${r.error ? `, ERROR ${r.error}` : ""}`;
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

for (const dataset of datasets) {
  for (const reader of ["zarr", "icechunk"]) {
    const results = [];
    for (let i = 0; i < runs; i++) results.push(await run(dataset, reader));
    for (const phase of ["cold", "warm"]) {
      const rs = results.map((r) => r[phase]);
      const summary = {
        ready: median(rs.map((r) => r.ready ?? NaN)),
        lastData: median(rs.map((r) => r.lastData)),
        requests: median(rs.map((r) => r.requests)),
        fromCache: median(rs.map((r) => r.fromCache)),
        bytes: median(rs.map((r) => r.bytes)),
        error: rs.find((r) => r.error)?.error ?? null,
      };
      console.log(`${dataset.padEnd(40)} ${reader.padEnd(8)} ${phase}  ${fmt(summary)}`);
    }
  }
}
await browser.close();
