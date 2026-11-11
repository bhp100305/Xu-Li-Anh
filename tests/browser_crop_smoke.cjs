// Local Flask :5000 and an isolated headless Chromium CDP :9224 are required.
// node tests/browser_crop_smoke.cjs <original image> <artifact folder>
const assert = require('node:assert/strict');
const { mkdirSync, writeFileSync } = require('node:fs');
const { resolve, join } = require('node:path');

async function main() {
    const source = resolve(process.argv[2]);
    const folder = resolve(process.argv[3]);
    mkdirSync(folder, { recursive: true });
    const targets = await (await fetch('http://127.0.0.1:9224/json/list')).json();
    const socket = new WebSocket(targets.find(item => item.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    const pending = new Map(), errors = [], requests = [];
    let nextId = 0;
    socket.onmessage = event => {
        const message = JSON.parse(event.data);
        if (message.id) {
            const item = pending.get(message.id); pending.delete(message.id);
            if (message.error) item.reject(new Error(JSON.stringify(message.error)));
            else item.resolve(message.result);
        } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
        else if (message.method === 'Network.requestWillBeSent') requests.push(message.params.request.url);
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
        const deadline = Date.now() + 20000;
        while (Date.now() < deadline) {
            if (await evaluate(expression)) return;
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        throw new Error(`Timed out: ${expression}`);
    }
    async function screenshot(name) {
        const result = await send('Page.captureScreenshot', { format: 'png' });
        writeFileSync(join(folder, name), Buffer.from(result.data, 'base64'));
    }
    try {
        await send('Page.enable'); await send('Runtime.enable'); await send('Network.enable'); await send('DOM.enable');
        await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
        await send('Page.navigate', { url: 'http://127.0.0.1:5000/multipage' });
        await waitFor('document.readyState === "complete" && typeof beginCropEditor === "function"');
        const doc = await send('DOM.getDocument');
        const input = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '#imageFiles' });
        await send('DOM.setFileInputFiles', { nodeId: input.nodeId, files: [source, resolve('test_image.jpg')] });
        await waitFor('pages.length === 2 && !busy');
        const initialCorners = await evaluate('pages[0].corners');
        if (process.argv.includes('--expect-full-frame'))
            assert.deepEqual(initialCorners, [[0, 0], [1, 0], [1, 1], [0, 1]], 'Automatic region must preserve the entire rectified page');
        await evaluate('openPageViewer(0)');
        await waitFor('document.getElementById("viewerImage").complete');
        await screenshot('crop-result-mobile.png');
        const initialUrl = await evaluate('pages[0].url');
        await evaluate('document.getElementById("editCrop").click()');
        await waitFor('editingCrop && !busy && document.getElementById("cropImage").complete');
        assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
        const point = await evaluate('(()=>{const r=document.querySelector("#cropHandles button").getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()');
        const before = await evaluate('cropCorners[0].slice()');
        await screenshot('crop-editor-mobile.png');
        const dx = before[0] > 0.05 ? -8 : 8;
        await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: point.x, y: point.y, button: 'left', clickCount: 1 });
        await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: point.x + dx, y: point.y + 3, button: 'left', buttons: 1 });
        await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: point.x + dx, y: point.y + 3, button: 'left', clickCount: 1 });
        const moved = await evaluate('cropCorners[0].slice()');
        assert.ok((moved[0] - before[0]) * dx > 0 && moved[1] > before[1], 'Real pointer dragging must update normalized corners');
        await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowRight', code: 'ArrowRight' });
        await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowRight', code: 'ArrowRight' });
        assert.ok(await evaluate(`cropCorners[0][0] > ${moved[0]}`));
        await evaluate('document.getElementById("selectWholeImage").click()');
        assert.deepEqual(await evaluate('cropCorners'), [[0, 0], [1, 0], [1, 1], [0, 1]]);
        await evaluate('document.getElementById("applyCrop").click()');
        await waitFor('!busy && !editingCrop');
        assert.equal(await evaluate('pages.length'), 2);
        assert.notEqual(await evaluate('pages[0].url'), initialUrl);
        assert.equal(requests.filter(url => url.endsWith('/api/multipage/rectify')).length, 1);
        await evaluate('showViewerPage(1); document.getElementById("editCrop").click()');
        await waitFor('editingCrop && !busy');
        const secondUrl = await evaluate('pages[1].url');
        await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
        await evaluate('fitCropSurface()');
        await screenshot('crop-editor-desktop.png');
        await evaluate('document.getElementById("cancelCrop").click()');
        assert.equal(await evaluate('pages[1].url'), secondUrl);
        assert.deepEqual(errors, []);
        console.log(JSON.stringify({ passed: true, pages: 2, initialCorners,
            checks: ['original photo batch upload', 'real pointer drag', 'keyboard adjustment', 'select whole original', 'original-pixel rectification', 'replace existing page', 'cancel preserves result', 'mobile and desktop layout'],
            screenshots: ['crop-result-mobile.png', 'crop-editor-mobile.png', 'crop-editor-desktop.png'].map(name => join(folder, name)) }, null, 2));
    } finally { socket.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
