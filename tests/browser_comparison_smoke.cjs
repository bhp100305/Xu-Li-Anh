// Local Flask :5000 and an isolated Chromium CDP :9224 are required.
// node tests/browser_comparison_smoke.cjs <document photo> <artifact folder>
const assert = require('node:assert/strict');
const { mkdirSync, writeFileSync } = require('node:fs');
const { join, resolve } = require('node:path');

async function main() {
    const source = resolve(process.argv[2]), folder = resolve(process.argv[3]);
    mkdirSync(folder, { recursive: true });
    const targets = await (await fetch('http://127.0.0.1:9224/json/list')).json();
    const socket = new WebSocket(targets.find(item => item.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    const pending = new Map(), errors = [];
    let nextId = 0;
    socket.onmessage = event => {
        const message = JSON.parse(event.data);
        if (message.id) {
            const item = pending.get(message.id); pending.delete(message.id);
            if (message.error) item.reject(new Error(JSON.stringify(message.error)));
            else item.resolve(message.result);
        } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
    };
    const send = (method, params = {}) => new Promise((resolve, reject) => {
        const id = ++nextId; pending.set(id, { resolve, reject });
        socket.send(JSON.stringify({ id, method, params }));
    });
    async function evaluate(expression) {
        const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
        if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
        return result.result.value;
    }
    async function waitFor(expression) {
        const deadline = Date.now() + 30000;
        while (Date.now() < deadline) {
            try { if (await evaluate(expression)) return; } catch { /* Navigation replaces the JS context. */ }
            await new Promise(resolve => setTimeout(resolve, 150));
        }
        throw new Error(`Timed out: ${expression}`);
    }
    async function screenshot(name) {
        const result = await send('Page.captureScreenshot', { format: 'png' });
        writeFileSync(join(folder, name), Buffer.from(result.data, 'base64'));
    }
    async function upload(file) {
        const doc = await send('DOM.getDocument');
        const input = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '#imageInput' });
        await send('DOM.setFileInputFiles', { nodeId: input.nodeId, files: [file] });
        await waitFor('document.getElementById("fileName").textContent && document.getElementById("imageInput").validity.valid');
        await evaluate('document.getElementById("processButton").click()');
        await waitFor('document.querySelectorAll("[data-comparison-card]").length === 6 && document.readyState === "complete" && !document.getElementById("processButton").disabled && !document.getElementById("imageInput").value');
    }
    try {
        await send('Page.enable'); await send('Runtime.enable'); await send('DOM.enable');
        await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
        await send('Page.navigate', { url: 'http://127.0.0.1:5000/?tab=compare' });
        await waitFor('document.readyState === "complete" && document.getElementById("imageInput")');
        await upload(source);
        assert.equal(await evaluate('document.querySelectorAll("[data-scan-status=ready]").length'), 6);
        assert.equal(await evaluate('document.querySelectorAll(".result-download").length'), 6);
        assert.equal(await evaluate('document.querySelectorAll("[data-comparison-view]").length'), 0);
        assert.ok(await evaluate('document.getElementById("comparisonOverview").src.endsWith("_canny_scan.png")'));
        await waitFor('document.getElementById("comparisonOverview").complete && document.getElementById("comparisonOverview").naturalWidth > 0');
        await evaluate('document.getElementById("result-heading").scrollIntoView({block:"start"})');
        await screenshot('comparison-overview.png');
        const card = 'document.querySelector("[data-comparison-key=canny]")';
        const edgeCards = await evaluate('Array.from(document.querySelectorAll("[data-comparison-card]"), card => ({key:card.dataset.comparisonKey,image:card.querySelector("img").src,link:card.querySelector("[data-comparison-preview]").href}))');
        assert.ok(edgeCards.every(item => item.image.endsWith(`_${item.key}.jpg`) && item.link === item.image));
        await evaluate(`${card}.scrollIntoView({block:"start"})`);
        await waitFor(`${card}.querySelector('img').complete && ${card}.querySelector('img').naturalWidth > 0`);
        await screenshot('comparison-canny-desktop.png');
        assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
        await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
        await evaluate(`${card}.scrollIntoView({block:"start"})`);
        await screenshot('comparison-canny-mobile.png');
        assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
        const downloads = await evaluate('Promise.all(Array.from(document.querySelectorAll(".result-download"), async link => {const r=await fetch(link.href);return {status:r.status,type:r.headers.get("content-type"),bytes:(await r.arrayBuffer()).byteLength}}))');
        assert.equal(downloads.length, 6);
        assert.ok(downloads.every(item => item.status === 200 && item.type.includes('image/jpeg') && item.bytes > 1000));
        assert.equal(await evaluate('(async()=>{const r=await fetch(document.querySelector(".preview-panel .download-button").href);return r.status===200 && r.headers.get("content-type").includes("image/png")})()'), true);

        // Submit a blank image through the actual upload and navigation path.
        const blank = await evaluate('(async()=>{const c=document.createElement("canvas");c.width=80;c.height=100;return c.toDataURL("image/png").split(",")[1]})()');
        const blankPath = join(folder, 'black.png');
        writeFileSync(blankPath, Buffer.from(blank, 'base64'));
        await upload(blankPath);
        assert.equal(await evaluate('document.querySelectorAll("[data-scan-status=failed]").length'), 6);
        assert.equal(await evaluate('document.querySelectorAll(".scan-download").length'), 0);
        assert.equal(await evaluate('document.querySelectorAll("[data-comparison-view]").length'), 0);
        assert.ok(await evaluate('document.getElementById("comparisonSummary").textContent.includes("0 / 6")'));
        assert.deepEqual(errors, []);
        console.log(JSON.stringify({ passed: true, scans: 6,
            checks: ['real form upload', 'six original edge cards', 'Canny scanned overview preserved', 'six edge downloads', 'overview PNG download', 'mobile layout', 'all-failed results keep edge images'],
            screenshots: ['comparison-overview.png', 'comparison-canny-desktop.png', 'comparison-canny-mobile.png'].map(name => join(folder, name)) }, null, 2));
    } finally { socket.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
