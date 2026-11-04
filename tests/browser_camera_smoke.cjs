// Run against a local Flask server and a Chromium CDP endpoint with a fake camera.
// node tests/browser_camera_smoke.cjs <temporary artifact folder>
const assert = require('node:assert/strict');
const { writeFileSync } = require('node:fs');
const { join } = require('node:path');

async function main() {
    const folder = process.argv[2];
    if (!folder) throw new Error('Pass the temporary folder containing document.png');
    const targets = await (await fetch('http://127.0.0.1:9224/json/list')).json();
    const target = targets.find(item => item.type === 'page');
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    let nextId = 0;
    const pending = new Map();
    const errors = [];
    const requests = [];
    socket.onmessage = event => {
        const message = JSON.parse(event.data);
        if (message.id) {
            const item = pending.get(message.id); pending.delete(message.id);
            if (message.error) item.reject(new Error(JSON.stringify(message.error)));
            else item.resolve(message.result);
        } else if (message.method === 'Runtime.exceptionThrown') {
            errors.push(message.params.exceptionDetails.text);
        } else if (message.method === 'Network.requestWillBeSent') {
            requests.push(message.params.request.url);
        }
    };
    function send(method, params = {}) {
        const id = ++nextId;
        return new Promise((resolve, reject) => {
            pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params }));
        });
    }
    async function evaluate(expression) {
        const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
        if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
        return result.result.value;
    }
    async function waitFor(expression, timeout = 20000) {
        const deadline = Date.now() + timeout;
        while (Date.now() < deadline) {
            if (await evaluate(expression)) return;
            await new Promise(resolve => setTimeout(resolve, 150));
        }
        throw new Error(`Timed out: ${expression}; hint=${await evaluate('document.getElementById("cameraHint")?.textContent')}`);
    }
    async function screenshot(name) {
        const result = await send('Page.captureScreenshot', { format: 'png' });
        writeFileSync(join(folder, name), Buffer.from(result.data, 'base64'));
    }
    try {
        await send('Page.enable'); await send('Runtime.enable'); await send('Network.enable');
        await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
        await send('Page.navigate', { url: 'http://127.0.0.1:5051/multipage' });
        await waitFor('document.readyState === "complete" && typeof startCamera === "function"');
        await evaluate('document.getElementById("openCamera").click()');
        await waitFor('stream && !openingCamera && liveCorners !== null');
        assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
        await screenshot('camera-live.png');
        await evaluate('document.getElementById("captureButton").click()');
        await waitFor('pages.length === 1 && screen === "live" && !busy');
        assert.equal(await evaluate('stream !== null && !document.getElementById("camera").hidden'), true);
        assert.equal(requests.filter(url => url.endsWith('/api/multipage/scan')).length, 1);
        await evaluate('document.getElementById("captureButton").click()');
        await waitFor('pages.length === 2 && screen === "live" && !busy');
        assert.equal(requests.filter(url => url.endsWith('/api/multipage/scan')).length, 2);
        await evaluate('document.getElementById("showPages").click()');
        await waitFor('!document.getElementById("pageViewer").hidden');
        assert.equal(await evaluate('document.getElementById("viewerImage").src === document.getElementById("lastPage").src'), true);
        assert.equal(await evaluate('stream !== null'), true);
        await screenshot('camera-result.png');
        await evaluate('document.getElementById("closeViewer").click(); document.getElementById("closeScanner").click()');

        // Uploaded photos also use the original automatic scan pipeline.
        await send('DOM.enable');
        const doc = await send('DOM.getDocument');
        const input = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '#imageFiles' });
        await send('DOM.setFileInputFiles', { nodeId: input.nodeId, files: [join(folder, 'document.png')] });
        await waitFor('pages.length === 3 && !busy');
        assert.equal(requests.filter(url => url.endsWith('/api/multipage/scan')).length, 3);
        assert.equal(await evaluate('document.getElementById("exportPdf").disabled'), false);
        assert.deepEqual(errors, []);
        console.log(JSON.stringify({ passed: true, pages: 3, mobileViewport: '390x844',
            checks: ['live document guide', 'automatic shutter scan', 'continuous camera', 'processed image viewer', 'original image upload'],
            screenshots: ['camera-live.png', 'camera-result.png'].map(name => join(folder, name)) }, null, 2));
    } finally { socket.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
