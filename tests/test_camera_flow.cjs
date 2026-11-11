// Run with: node --test tests/test_camera_flow.cjs
// Exercise the page's actual event handlers with controlled camera/network APIs.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const template = readFileSync(join(__dirname, '../templates/multipage.html'), 'utf8');
const script = readFileSync(join(__dirname, '../static/js/multipage.js'), 'utf8');

class Element {
    constructor(tag = 'div') {
        this.tag = tag;
        this.children = [];
        this.handlers = {};
        this.style = {};
        this.disabled = false;
        this.hidden = false;
        this.textContent = '';
        this.firstChild = { textContent: '' };
        this.classList = { add() {}, remove() {}, toggle() {} };
    }
    addEventListener(type, handler) { (this.handlers[type] ??= []).push(handler); }
    async click() {
        if (this.disabled) return;
        await Promise.all((this.handlers.click ??= []).map(handler => handler({ target: this })));
    }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); }
    replaceChildren(...children) { this.children = children; }
    removeAttribute(name) { delete this[name]; }
    setAttribute(name, value) { this[name] = value; }
    toggleAttribute(name, force) { this[name] = force; }
    setPointerCapture() {}
    getBoundingClientRect() { return { left: 0, top: 0, width: 1000, height: 1000 }; }
    getClientRects() { return [{}]; }
    querySelectorAll(tag) {
        return this.children.flatMap(child => [
            ...(child.tag === tag ? [child] : []), ...child.querySelectorAll(tag),
        ]);
    }
    focus() { this.focused = true; }
    remove() {}
}

function setup() {
    const elements = {};
    for (const [, id] of template.matchAll(/\bid="([^"]+)"/g)) elements[id] = new Element();
    elements.pageViewer.hidden = true;
    elements.scanner.hidden = true;
    const events = {};
    const streams = [];
    const requests = [];
    const detectRequests = [];
    const mediaRequests = [];
    const encodings = [];
    const blobs = new Map();
    const controls = { failScan: false, nullBlob: false, playError: false, noDetection: false };
    let mediaRequest = async () => {
        const track = {
            stopped: false, constraints: [], stop() { this.stopped = true; },
            getCapabilities() { return controls.capabilities || {}; },
            async applyConstraints(options) {
                this.constraints.push(options);
                if (controls.focusError) throw new Error('Focus is not controllable');
            },
        };
        const stream = { getTracks: () => [track], getVideoTracks: () => [track], track };
        streams.push(stream);
        return stream;
    };
    Object.assign(elements.camera, {
        videoWidth: 1280, videoHeight: 720,
        async play() { if (controls.playError) throw new Error('Playback failed'); },
    });
    let draws = 0;
    elements.captureCanvas.getContext = () => ({ drawImage() { draws++; } });
    elements.captureCanvas.toBlob = (callback, type, quality) => {
        encodings.push({ type, quality });
        callback(controls.nullBlob ? null : new Blob([`frame-${draws}`], { type: 'image/jpeg' }));
    };
    let nextUrl = 0;
    let nextTimer = 0;
    const timers = new Map();
    const context = vm.createContext({
        document: {
            getElementById: id => elements[id],
            createElement: tag => {
                const element = new Element(tag);
                if (tag === 'canvas') {
                    element.getContext = () => ({ drawImage() {} });
                    element.toBlob = callback => callback(new Blob(['preview'], { type: 'image/jpeg' }));
                }
                return element;
            },
            body: new Element('body'),
            addEventListener(type, handler) { events[`document:${type}`] = handler; },
        },
        window: { innerWidth: 400, innerHeight: 800, addEventListener(type, handler) { events[type] = handler; } },
        navigator: { mediaDevices: { getUserMedia: options => { mediaRequests.push(options); return mediaRequest(options); } } },
        Blob, File, FormData, AbortController,
        setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
        clearTimeout(id) { timers.delete(id); },
        Image: class {
            naturalWidth = controls.imageWidth || 1280;
            naturalHeight = controls.imageHeight || 720;
            async decode() { if (controls.decodeError) throw new Error('Cannot decode photo'); }
        },
        URL: {
            createObjectURL(blob) { const url = `blob:test-${++nextUrl}`; blobs.set(url, blob); return url; },
            revokeObjectURL(url) { blobs.delete(url); },
        },
        async fetch(url, options) {
            if (url.endsWith('/detect')) {
                detectRequests.push({ url, options });
                return { ok: true, json: async () => ({ corners: controls.noDetection ? null : [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]] }) };
            }
            requests.push({ url, options });
            if (controls.scanGate) await controls.scanGate;
            return {
                ok: !controls.failScan,
                headers: { get: () => controls.cornerHeader ?? JSON.stringify([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]) },
                json: async () => ({ error: 'Không tìm thấy trang giấy' }),
                blob: async () => new Blob([url.endsWith('/rectify') ? 'rectified' : 'scanned'], { type: 'image/png' }),
            };
        },
    });
    vm.runInContext(script, context);
    return {
        elements, events, streams, requests, detectRequests, mediaRequests, encodings, controls, blobs, context,
        setMediaRequest(callback) { mediaRequest = callback; },
        click: id => elements[id].click(),
        value: expression => vm.runInContext(expression, context),
    };
}


async function uploadPhotos(app, names = ['document']) {
    const files = names.map(name => new File([name], `${name}.png`, { type: 'image/png' }));
    await app.elements.imageFiles.handlers.change[0]({ target: { files, value: 'selected' } });
}

test('one shutter click scans immediately and keeps the same camera stream open', async () => {
    const app = setup();
    await app.click('openCamera');
    await app.click('captureButton');
    assert.equal(app.requests.length, 1);
    assert.equal(app.requests[0].url, '/api/multipage/scan');
    const uploaded = app.requests[0].options.body.get('image');
    assert.equal(await uploaded.text(), 'frame-1');
    assert.equal(uploaded.type, 'image/jpeg');
    assert.equal(app.elements.pageCount.textContent, '1 trang');
    assert.equal(app.elements.capturedCount.textContent, 1);
    assert.equal(app.elements.lastPage.hidden, false);
    assert.equal(app.streams.length, 1, 'Do not request camera permission again');
    assert.equal(app.streams[0].track.stopped, false);
    assert.equal(app.elements.camera.srcObject, app.streams[0]);
    assert.equal(app.elements.camera.hidden, false);
    assert.equal(app.elements.captureReview.hidden, true);
    assert.equal(app.elements.scanner.hidden, false);
    assert.equal(app.elements.captureButton.disabled, false);
});

test('crop editor shows the original and applying replaces that page and its PDF image', async () => {
    const app = setup();
    await uploadPhotos(app, ['first', 'second']);
    app.value('openPageViewer(1)');
    const oldUrl = app.elements.viewerImage.src;
    await app.click('editCrop');
    assert.equal(app.value('editingCrop'), true);
    assert.equal(await app.blobs.get(app.elements.cropImage.src).text(), 'second');
    assert.equal(app.elements.viewerImage.hidden, true);
    assert.equal(app.elements.viewerNavigation.hidden, true);
    assert.equal(app.elements.cropHandles.children.length, 4);
    assert.equal(app.elements.cropHandles.children[0].style.left, '10%');
    app.value('moveCropCorner(0, [0.15, 0.12])');
    await app.click('applyCrop');
    const request = app.requests.at(-1);
    assert.equal(request.url, '/api/multipage/rectify');
    assert.equal(await request.options.body.get('image').text(), 'second', 'Never rescan the processed PNG');
    assert.deepEqual(JSON.parse(request.options.body.get('corners'))[0], [0.15, 0.12]);
    assert.equal(app.elements.pageCount.textContent, '2 trang', 'Replace; do not add a duplicate page');
    assert.equal(app.elements.viewerTitle.textContent, 'Trang 2');
    assert.equal(await app.blobs.get(app.elements.downloadPage.href).text(), 'rectified');
    assert.equal(app.blobs.has(oldUrl), false);
    assert.equal(app.value('editingCrop'), false);
    await app.click('closeViewer');
    await app.click('exportPdf');
    assert.equal(await app.requests.at(-1).options.body.getAll('pages')[1].text(), 'rectified');
});

test('canceling or escaping a crop keeps the old scan and releases the temporary original URL', async () => {
    const app = setup();
    await uploadPhotos(app);
    app.value('openPageViewer(0)');
    const oldUrl = app.elements.viewerImage.src;
    for (const method of ['cancel', 'escape']) {
        await app.click('editCrop');
        const originalUrl = app.elements.cropImage.src;
        app.value('moveCropCorner(0, [0.2, 0.2])');
        if (method === 'cancel') await app.click('cancelCrop');
        else app.events['document:keydown']({ key: 'Escape', preventDefault() {} });
        assert.equal(app.elements.pageViewer.hidden, false);
        assert.equal(app.value('editingCrop'), false);
        assert.equal(app.elements.viewerImage.src, oldUrl);
        assert.equal(app.blobs.has(originalUrl), false);
    }
    assert.equal(app.requests.length, 1, 'Cancel must not send a correction');
});

test('select whole image replaces an inner selection with the four original boundaries', async () => {
    const app = setup();
    await uploadPhotos(app);
    app.value('openPageViewer(0)');
    await app.click('editCrop');
    await app.click('selectWholeImage');
    assert.equal(app.elements.cropHandles.children[0].style.left, '0%');
    assert.equal(app.elements.cropHandles.children[2].style.left, '100%');
    assert.equal(app.elements.cropHandles.children[2].style.top, '100%');
    await app.click('applyCrop');
    assert.deepEqual(JSON.parse(app.requests.at(-1).options.body.get('corners')),
        [[0, 0], [1, 0], [1, 1], [0, 1]]);
    assert.equal(await app.requests.at(-1).options.body.get('image').text(), 'document');
    assert.equal(app.elements.pageCount.textContent, '1 trang');
});

test('failed correction retains the scan and original for retry and blocks duplicate submissions', async () => {
    const app = setup();
    await uploadPhotos(app);
    app.value('openPageViewer(0)');
    const oldUrl = app.elements.viewerImage.src;
    await app.click('editCrop');
    app.controls.failScan = true;
    let release;
    app.controls.scanGate = new Promise(resolve => { release = resolve; });
    const pending = app.click('applyCrop');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(app.elements.closeViewer.disabled, true);
    assert.equal(app.elements.cropHandles.children[0].disabled, true);
    await app.click('applyCrop');
    assert.equal(app.requests.length, 2);
    release();
    await pending;
    assert.equal(app.elements.viewerImage.src, oldUrl);
    assert.equal(app.value('editingCrop'), true);
    assert.match(app.elements.viewerMessage.textContent, /Không tìm thấy/);
    app.controls.failScan = false;
    await app.click('applyCrop');
    assert.equal(app.value('editingCrop'), false);
    assert.equal(await app.blobs.get(app.elements.downloadPage.href).text(), 'rectified');
});

test('corner dragging and keyboard move within the original image and reject crossed polygons', async () => {
    const app = setup();
    await uploadPhotos(app);
    app.value('openPageViewer(0)');
    await app.click('editCrop');
    const handle = app.elements.cropHandles.children[0];
    const pointer = { pointerId: 1, clientX: 200, clientY: 150, preventDefault() {} };
    handle.handlers.pointerdown[0](pointer);
    handle.handlers.pointermove[0](pointer);
    assert.equal(app.value('cropCorners[0][0]'), 0.2);
    assert.equal(app.value('cropCorners[0][1]'), 0.15);
    handle.handlers.pointerup[0]({ ...pointer, clientX: -100 });
    assert.equal(app.value('cropCorners[0][0]'), 0);
    const before = app.value('JSON.stringify(cropCorners)');
    app.value('moveCropCorner(0, [1, 1])');
    assert.equal(app.value('JSON.stringify(cropCorners)'), before);
    handle.handlers.keydown[0]({ key: 'ArrowRight', preventDefault() {}, stopPropagation() {} });
    assert.equal(app.value('cropCorners[0][0]'), 1 / 1280);
    app.events['document:keydown']({ key: 'ArrowRight', preventDefault() {} });
    assert.equal(app.elements.viewerTitle.textContent, 'Trang 1', 'Editing arrows must not navigate pages');
});

test('missing corner metadata and unreadable originals leave the saved scan intact', async () => {
    const app = setup();
    app.controls.cornerHeader = 'invalid JSON';
    await uploadPhotos(app);
    app.value('openPageViewer(0)');
    const oldUrl = app.elements.viewerImage.src;
    app.controls.decodeError = true;
    await app.click('editCrop');
    assert.equal(app.value('editingCrop'), false);
    assert.equal(app.elements.viewerImage.src, oldUrl);
    assert.equal(app.elements.closeViewer.disabled, false);
    app.controls.decodeError = false;
    await app.click('editCrop');
    assert.equal(app.value('cropCorners[0][0]'), 0.03);
    assert.equal(app.value('cropCorners[2][1]'), 0.97);
});

test('successive shutter clicks scan distinct pages without confirming or reopening camera', async () => {
    const app = setup();
    await app.click('openCamera');
    await app.click('captureButton');
    await app.click('captureButton');
    assert.equal(app.requests.length, 2);
    assert.equal(await app.requests[0].options.body.get('image').text(), 'frame-1');
    assert.equal(await app.requests[1].options.body.get('image').text(), 'frame-2');
    assert.equal(app.elements.pageCount.textContent, '2 trang');
    assert.equal(app.streams.length, 1);
    assert.equal(app.elements.camera.hidden, false);
});

test('captured frame stays visible until processing finishes and duplicate clicks are ignored', async () => {
    const app = setup();
    await app.click('openCamera');
    let release;
    app.controls.scanGate = new Promise(resolve => { release = resolve; });
    const pending = app.click('captureButton');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(app.requests.length, 1);
    assert.equal(app.elements.captureButton.disabled, true);
    assert.equal(app.elements.camera.hidden, true);
    assert.equal(app.elements.captureCanvas.hidden, false);
    assert.equal(app.value('screen'), 'processing');
    assert.equal(app.elements.captureCanvas.width, 1280);
    assert.equal(app.elements.captureCanvas.height, 720);
    app.elements.camera.videoWidth = 720;
    app.elements.camera.videoHeight = 1280;
    app.value('fitSurface()');
    assert.equal(app.elements.captureCanvas.width, 1280, 'Moving camera cannot replace the frozen frame');
    assert.equal(app.streams[0].track.stopped, false);
    await app.click('captureButton');
    assert.equal(app.requests.length, 1);
    release();
    await pending;
    assert.equal(app.elements.captureButton.disabled, false);
    assert.equal(app.elements.camera.hidden, false);
    assert.equal(app.elements.captureCanvas.hidden, true);
    assert.equal(app.value('screen'), 'live');
    assert.equal(app.elements.pageCount.textContent, '1 trang');
});

test('failed automatic scan keeps the original photo for retry and resumes camera', async () => {
    const app = setup();
    app.controls.failScan = true;
    await app.click('openCamera');
    await app.click('captureButton');
    assert.match(app.elements.status.textContent, /Không tìm thấy trang giấy/);
    assert.equal(app.elements.captureReview.hidden, false);
    assert.equal(app.elements.scanCapture.disabled, false);
    assert.equal(app.elements.saveCapture.hidden, false);
    assert.equal(app.streams[0].track.stopped, true);
    const photo = app.blobs.get(app.elements.saveCapture.href);
    assert.equal(await photo.text(), 'frame-1');
    app.controls.failScan = false;
    await app.click('scanCapture');
    assert.equal(app.elements.pageCount.textContent, '1 trang');
    assert.equal(app.elements.captureReview.hidden, true);
    assert.equal(app.elements.camera.hidden, false);
    assert.equal(app.streams.length, 2);
});

test('retake after an automatic scan error replaces the photo and scans immediately', async () => {
    const app = setup();
    app.controls.failScan = true;
    await app.click('openCamera');
    await app.click('captureButton');
    const oldUrl = app.elements.saveCapture.href;
    app.controls.failScan = false;
    await app.click('retakePhoto');
    await app.click('captureButton');
    assert.equal(app.blobs.has(oldUrl), false);
    assert.equal(app.requests.length, 2);
    assert.equal(await app.requests[1].options.body.get('image').text(), 'frame-2');
    assert.equal(app.elements.pageCount.textContent, '1 trang');
});

test('encoding failure keeps camera ready to take another photo', async () => {
    const app = setup();
    await app.click('openCamera');
    app.controls.nullBlob = true;
    await app.click('captureButton');
    assert.equal(app.requests.length, 0);
    assert.equal(app.streams[0].track.stopped, false);
    assert.equal(app.elements.camera.hidden, false);
    assert.equal(app.elements.captureButton.disabled, false);
    assert.match(app.elements.status.textContent, /Không thể lưu ảnh/);
    app.controls.nullBlob = false;
    await app.click('captureButton');
    assert.equal(app.requests.length, 1);
});

test('camera playback failure releases tracks and restores controls', async () => {
    const app = setup();
    app.controls.playError = true;
    await app.click('openCamera');
    assert.equal(app.streams[0].track.stopped, true);
    assert.equal(app.elements.openCamera.disabled, false);
    assert.equal(app.elements.camera.srcObject, null);
});

test('leaving while permission is pending stops the late camera stream', async () => {
    const app = setup();
    let resolve;
    app.setMediaRequest(() => new Promise(done => { resolve = done; }));
    const opening = app.click('openCamera');
    app.events.pagehide();
    const track = { stopped: false, stop() { this.stopped = true; } };
    resolve({ getTracks: () => [track] });
    await opening;
    assert.equal(track.stopped, true);
    assert.equal(app.elements.camera.srcObject, null);
    assert.equal(app.requests.length, 0);
});

test('leaving during a scan never reopens camera when the upload finishes', async () => {
    const app = setup();
    await app.click('openCamera');
    let release;
    app.controls.scanGate = new Promise(resolve => { release = resolve; });
    const pending = app.click('captureButton');
    await new Promise(resolve => setImmediate(resolve));
    app.events.pagehide();
    release(); await pending;
    assert.equal(app.streams[0].track.stopped, true);
    assert.equal(app.streams.length, 1);
    assert.equal(app.elements.camera.srcObject, null);
});

test('live guide coordinates never replace the original full-photo scan request', async () => {
    const app = setup();
    await app.click('openCamera');
    await app.value('detectLive(cameraRequest)');
    assert.equal(app.elements.cornerOverlay.hidden, false);
    const hintsBeforeCapture = app.detectRequests.length;
    await app.click('captureButton');
    const body = app.requests[0].options.body;
    assert.deepEqual(Array.from(body.keys()), ['image']);
    assert.equal(await body.get('image').text(), 'frame-1');
    assert.equal(app.elements.captureCanvas.width, 1280);
    assert.equal(app.elements.captureCanvas.height, 720);
    assert.equal(app.detectRequests.length, hintsBeforeCapture, 'No separate preview-crop request during capture');
});

test('live detection paints the shape without scanning until shutter is pressed', async () => {
    const app = setup();
    await app.click('openCamera');
    await app.value('detectLive(cameraRequest)');
    assert.equal(app.elements.cornerOverlay.hidden, false);
    assert.match(app.elements.documentPolygon.points, /100,100/);
    assert.equal(app.requests.length, 0);
});

test('uploaded photos use the original automatic scanner with only original image data', async () => {
    const app = setup();
    await uploadPhotos(app, ['one', 'two']);
    assert.equal(app.requests.length, 2);
    assert.equal(await app.requests[0].options.body.get('image').text(), 'one');
    assert.equal(await app.requests[1].options.body.get('image').text(), 'two');
    for (const request of app.requests) assert.deepEqual(Array.from(request.options.body.keys()), ['image']);
    assert.equal(app.elements.pageCount.textContent, '2 trang');
});

test('missed preview detection uses the full scanner instead of a guessed crop', async () => {
    const app = setup();
    app.controls.noDetection = true;
    await app.click('openCamera');
    await app.click('captureButton');
    assert.equal(app.requests.length, 1);
    assert.equal(app.requests[0].options.body.has('corners'), false);
    assert.equal(app.elements.camera.hidden, false);
    assert.equal(app.elements.pageCount.textContent, '1 trang');
});

test('reaching 15 pages ends capture and prevents a sixteenth scan', async () => {
    const app = setup();
    await app.click('openCamera');
    for (let index = 0; index < 15; index++) await app.click('captureButton');
    assert.equal(app.requests.length, 15);
    assert.equal(app.elements.pageCount.textContent, '15 trang');
    assert.equal(app.elements.scanner.hidden, true);
    assert.equal(app.streams[0].track.stopped, true);
    await app.click('captureButton');
    assert.equal(app.requests.length, 15);
});

test('a scanned page thumbnail opens the processed image and its download', async () => {
    const app = setup();
    await uploadPhotos(app);
    const preview = app.elements.pageGrid.children[0].children[0];
    assert.equal(preview.tag, 'button');
    assert.equal(preview['aria-label'], 'Mở ảnh trang 1');
    await preview.click();
    assert.equal(app.elements.pageViewer.hidden, false);
    assert.equal(app.elements.viewerTitle.textContent, 'Trang 1');
    assert.equal(app.elements.viewerImage.src, app.elements.lastPage.src);
    assert.equal(app.elements.downloadPage.href, app.elements.viewerImage.src);
    assert.equal(app.elements.downloadPage.download, 'DocuScan-trang-1.png');
    assert.equal(app.elements.previousPage.disabled, true);
    assert.equal(app.elements.nextPage.disabled, true);
    assert.equal(app.requests.length, 1, 'Viewing must not scan the image again');
});

test('camera thumbnail previews the latest scan and closing resumes the same camera', async () => {
    const app = setup();
    await app.click('openCamera');
    await app.click('captureButton');
    await app.click('showPages');
    assert.equal(app.elements.pageViewer.hidden, false);
    assert.equal(app.streams[0].track.stopped, false);
    assert.equal(app.elements.scanner.inert, true);
    await app.click('captureButton');
    assert.equal(app.requests.length, 1, 'The covered camera cannot capture behind the viewer');
    await app.click('closeViewer');
    assert.equal(app.elements.pageViewer.hidden, true);
    assert.equal(app.elements.scanner.inert, false);
    assert.equal(app.elements.scanner.hidden, false);
    assert.equal(app.elements.camera.srcObject, app.streams[0]);
    await app.click('captureButton');
    assert.equal(app.requests.length, 2);
    assert.equal(app.streams.length, 1);
});

test('viewer navigation updates the image, count and download filename', async () => {
    const app = setup();
    await uploadPhotos(app, ['one', 'two']);
    await app.elements.pageGrid.children[0].children[0].click();
    const firstUrl = app.elements.viewerImage.src;
    await app.click('nextPage');
    assert.equal(app.elements.viewerCount.textContent, '2 / 2');
    assert.notEqual(app.elements.viewerImage.src, firstUrl);
    assert.equal(app.elements.downloadPage.download, 'DocuScan-trang-2.png');
    assert.equal(app.elements.nextPage.disabled, true);
    await app.click('previousPage');
    assert.equal(app.elements.viewerImage.src, firstUrl);
    assert.equal(app.elements.viewerCount.textContent, '1 / 2');
});

test('zoom resets for the next image and Escape closes the viewer only', async () => {
    const app = setup();
    await app.click('openCamera');
    await app.click('captureButton');
    await app.click('captureButton');
    await app.click('showPages');
    await app.click('zoomViewer');
    assert.equal(app.elements.zoomViewer['aria-pressed'], 'true');
    assert.equal(app.elements.zoomViewer.textContent, 'Vừa màn hình');
    app.events['document:keydown']({ key: 'ArrowLeft', preventDefault() {} });
    assert.equal(app.elements.viewerCount.textContent, '1 / 2');
    assert.equal(app.elements.zoomViewer['aria-pressed'], 'false');
    app.events['document:keydown']({ key: 'Escape', preventDefault() {} });
    assert.equal(app.elements.pageViewer.hidden, true);
    assert.equal(app.elements.library.inert, false);
    assert.equal(app.elements.scanner.hidden, false);
    assert.equal(app.streams[0].track.stopped, false);
});

test('viewer numbering follows reordered pages and deleted pages cannot open', async () => {
    const app = setup();
    await uploadPhotos(app, ['one', 'two']);
    const firstUrl = app.elements.pageGrid.children[0].children[0].children[0].src;
    app.value('movePage(0, 1)');
    await app.elements.pageGrid.children[1].children[0].click();
    assert.equal(app.elements.viewerImage.src, firstUrl);
    assert.equal(app.elements.viewerTitle.textContent, 'Trang 2');
    await app.click('closeViewer');
    await app.click('clearPages');
    assert.equal(app.elements.showPages.disabled, true);
    app.value('openPageViewer(0)');
    assert.equal(app.elements.pageViewer.hidden, true);
});

test('live camera requests high resolution and encodes with higher JPEG quality', async () => {
    const app = setup();
    await app.click('openCamera');
    assert.equal(app.mediaRequests[0].video.width.ideal, 2560);
    assert.equal(app.mediaRequests[0].video.height.ideal, 1920);
    assert.equal(app.mediaRequests[0].video.facingMode.ideal, 'environment');
    await app.click('captureButton');
    assert.equal(app.encodings[0].type, 'image/jpeg');
    assert.equal(app.encodings[0].quality, 0.98);
});

test('continuous focus is requested when supported and a rejection does not break camera', async () => {
    for (const focusError of [false, true]) {
        const app = setup();
        app.controls.capabilities = { focusMode: ['continuous', 'manual'] };
        app.controls.focusError = focusError;
        await app.click('openCamera');
        assert.equal(app.streams[0].track.constraints[0].advanced[0].focusMode, 'continuous');
        assert.equal(app.elements.captureButton.disabled, false);
    }
});

test('native HD keeps original full-resolution photo bytes and resumes live camera', async () => {
    const app = setup();
    app.controls.imageWidth = 4032;
    app.controls.imageHeight = 3024;
    await app.click('openCamera');
    await app.click('nativeCameraButton');
    assert.equal(app.streams[0].track.stopped, true);
    assert.equal(app.elements.nativeCameraInput.capture, 'environment');
    assert.equal(app.requests.length, 0);
    const file = new File(['full-resolution-original'], 'native.jpg', { type: 'image/jpeg' });
    await app.elements.nativeCameraInput.handlers.change[0]({ target: { files: [file], value: 'photo' } });
    assert.equal(app.elements.captureCanvas.width, 4032);
    assert.equal(app.elements.captureCanvas.height, 3024);
    assert.equal(await app.requests[0].options.body.get('image').text(), 'full-resolution-original');
    assert.equal(app.encodings.length, 0, 'Native JPEG must not be recompressed or resized in the browser');
    assert.equal(app.streams.length, 2);
    assert.equal(app.elements.camera.hidden, false);
    assert.equal(app.elements.pageCount.textContent, '1 trang');
});

test('canceling native HD returns to camera without sending a scan', async () => {
    const app = setup();
    await app.click('openCamera');
    await app.click('nativeCameraButton');
    await app.elements.nativeCameraInput.handlers.cancel[0]();
    assert.equal(app.streams.length, 2);
    assert.equal(app.requests.length, 0);
    assert.equal(app.elements.captureButton.disabled, false);
});

test('an unreadable native photo returns to the library with an error', async () => {
    const app = setup();
    await app.click('openCamera');
    await app.click('nativeCameraButton');
    app.controls.decodeError = true;
    const file = new File(['bad'], 'bad.jpg', { type: 'image/jpeg' });
    await app.elements.nativeCameraInput.handlers.change[0]({ target: { files: [file], value: 'photo' } });
    assert.equal(app.elements.scanner.hidden, true);
    assert.equal(app.elements.openCamera.disabled, false);
    assert.equal(app.requests.length, 0);
});

test('camera resolution fallback only retries unsupported constraints', async () => {
    for (const name of ['OverconstrainedError', 'NotAllowedError']) {
        const app = setup();
        let count = 0;
        app.setMediaRequest(async () => {
            if (++count === 1) { const error = new Error(name); error.name = name; throw error; }
            const track = { stop() {} };
            return { getTracks: () => [track], getVideoTracks: () => [track] };
        });
        await app.click('openCamera');
        assert.equal(app.mediaRequests.length, name === 'OverconstrainedError' ? 2 : 1);
        if (name === 'OverconstrainedError') assert.equal(app.elements.captureButton.disabled, false);
    }
});
