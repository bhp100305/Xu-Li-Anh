'use strict';

const byId = id => document.getElementById(id);
const pages = [];
const maxPages = 15;
const video = byId('camera');
const canvas = byId('captureCanvas');
const surface = byId('imageSurface');
const scanner = byId('scanner');
const pageGrid = byId('pageGrid');
const captureReview = byId('captureReview');
const scanCaptureButton = byId('scanCapture');
const saveCapture = byId('saveCapture');
const pageViewer = byId('pageViewer');

let stream = null;
let busy = false;
let openingCamera = false;
let cameraRequest = 0;
let screen = 'closed';
let capturedFile = null;
let capturedUrl = null;
let liveCorners = null;
let facingMode = 'environment';
let torch = false;
let detectionTimer;
let detectionController;
let previousFocus;
let viewerIndex = -1;
let viewerPreviousFocus;
let viewerZoomed = false;
let nativeCapturePending = false;
let editingCrop = false;
let cropCorners = null;
let cropSourceUrl = null;
let cropSize = null;

function setStatus(message) {
    byId('status').textContent = message;
    byId('cameraHint').textContent = message;
}

function setBusy(value) {
    busy = value;
    byId('imageFiles').disabled = value;
    byId('clearPages').disabled = value;
    byId('exportPdf').disabled = value || pages.length === 0;
    byId('closeScanner').disabled = value;
    byId('chooseImage').disabled = value;
    byId('showPages').disabled = value;
    byId('closeViewer').disabled = value;
    byId('editCrop').disabled = value;
    byId('applyCrop').disabled = value;
    byId('cancelCrop').disabled = value;
    byId('selectWholeImage').disabled = value;
    byId('previousPage').disabled = value || viewerIndex <= 0;
    byId('nextPage').disabled = value || viewerIndex >= pages.length - 1;
    byId('cropHandles').querySelectorAll('button').forEach(button => { button.disabled = value; });
    updateCaptureControls();
    renderPages();
}

function updateCaptureControls() {
    byId('captureButton').disabled = busy || openingCamera || !stream || pages.length >= maxPages;
    byId('openCamera').disabled = busy || openingCamera;
    byId('retakePhoto').disabled = busy || openingCamera;
    byId('switchCamera').disabled = busy || openingCamera || screen !== 'live';
    byId('showPages').disabled = busy || openingCamera || pages.length === 0;
    byId('nativeCameraButton').disabled = busy || openingCamera;
    scanCaptureButton.disabled = busy || openingCamera || screen !== 'review' || !capturedFile;
    byId('scanCapture').firstChild.textContent = busy ? 'Đang xử lý… ' : 'Thử scan lại ';
}

function openScanner(mode) {
    const newlyOpened = scanner.hidden;
    if (newlyOpened) previousFocus = document.activeElement;
    screen = mode;
    scanner.hidden = false;
    scanner.setAttribute('aria-busy', String(mode === 'processing'));
    document.body.classList.add('scanner-open');
    captureReview.hidden = mode !== 'review';
    byId('cameraActions').hidden = mode === 'review';
    const frozen = mode === 'review' || mode === 'processing';
    video.hidden = frozen;
    canvas.hidden = !frozen;
    byId('scannerTitle').textContent = mode === 'processing' ? 'Đang quét tài liệu'
        : mode === 'review' ? 'Ảnh chưa scan thành công' : 'Quét tài liệu';
    fitSurface();
    paintCorners();
    updateCaptureControls();
    if (newlyOpened && !busy) byId('closeScanner').focus();
}

function pauseDetection() {
    cameraRequest++;
    clearTimeout(detectionTimer);
    detectionController?.abort();
    detectionController = null;
}

function stopCamera() {
    pauseDetection();
    openingCamera = false;
    stream?.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    liveCorners = null;
    torch = false;
    byId('torchButton').disabled = true;
    byId('torchButton').setAttribute('aria-pressed', 'false');
    updateCaptureControls();
}

function closeScanner() {
    if (busy) return;
    stopCamera();
    screen = 'closed';
    scanner.hidden = true;
    document.body.classList.remove('scanner-open');
    previousFocus?.focus();
}

function clearCapture() {
    capturedFile = null;
    if (capturedUrl) URL.revokeObjectURL(capturedUrl);
    capturedUrl = null;
    saveCapture.removeAttribute('href');
    saveCapture.hidden = true;
    updateCaptureControls();
}

function fitSurface() {
    const stage = byId('cameraStage');
    const frozen = screen === 'review' || screen === 'processing';
    const width = frozen ? canvas.width : video.videoWidth;
    const height = frozen ? canvas.height : video.videoHeight;
    const ratio = width && height ? width / height : 3 / 4;
    const availableWidth = stage.clientWidth || window.innerWidth;
    const availableHeight = stage.clientHeight || window.innerHeight * 0.6;
    const fittedWidth = Math.min(availableWidth, availableHeight * ratio);
    surface.style.width = `${fittedWidth}px`;
    surface.style.height = `${fittedWidth / ratio}px`;
}

function validCorners(points) {
    if (!Array.isArray(points) || points.length !== 4 || points.some(point =>
        !Array.isArray(point) || point.length !== 2 || point.some(value => !Number.isFinite(value) || value < 0 || value > 1))) return false;
    let area = 0;
    for (let index = 0; index < 4; index++) {
        const a = points[index], b = points[(index + 1) % 4], c = points[(index + 2) % 4];
        if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 0) return false;
        area += a[0] * b[1] - b[0] * a[1];
    }
    return area / 2 >= 0.01;
}

function paintCorners() {
    const points = screen === 'live' ? liveCorners : null;
    const visible = validCorners(points);
    // SVGElement does not share HTMLElement.hidden in every browser.
    byId('cornerOverlay').toggleAttribute('hidden', !visible);
    byId('cameraGuide').hidden = visible || screen !== 'live';
    if (!visible) return;
    const scaled = points.map(point => point.map(value => value * 1000));
    byId('documentPolygon').setAttribute('points', scaled.map(point => point.join(',')).join(' '));
    byId('outsideMask').setAttribute('d', `M0 0H1000V1000H0Z M${scaled.map(point => point.join(' ')).join('L')}Z`);
}

async function detectFile(file, signal) {
    const data = new FormData();
    data.append('image', file, file.name || 'preview.jpg');
    const response = await fetch('/api/multipage/detect', { method: 'POST', body: data, signal });
    if (!response.ok) throw new Error('Chưa thể hiển thị khung hướng dẫn.');
    const result = await response.json();
    return validCorners(result.corners) ? result.corners : null;
}

async function detectLive(request) {
    if (request !== cameraRequest || !stream || screen !== 'live' || !pageViewer.hidden) return;
    try {
        if (video.videoWidth && video.videoHeight) {
            const preview = document.createElement('canvas');
            const ratio = Math.min(1, 480 / Math.max(video.videoWidth, video.videoHeight));
            preview.width = Math.round(video.videoWidth * ratio);
            preview.height = Math.round(video.videoHeight * ratio);
            preview.getContext('2d').drawImage(video, 0, 0, preview.width, preview.height);
            const blob = await new Promise(resolve => preview.toBlob(resolve, 'image/jpeg', 0.65));
            if (request !== cameraRequest || !blob) return;
            detectionController = new AbortController();
            const controller = detectionController;
            const timeout = setTimeout(() => controller.abort(), 3500);
            let detected;
            try { detected = await detectFile(blob, controller.signal); }
            finally { clearTimeout(timeout); }
            if (request !== cameraRequest) return;
            liveCorners = detected;
            paintCorners();
            setStatus(detected ? 'Đã thấy khung tài liệu · Giữ máy ổn định và bấm chụp' : 'Đặt đủ bốn góc tài liệu vào khung hình');
        }
    } catch (error) {
        if (request === cameraRequest) {
            liveCorners = null;
            paintCorners();
            setStatus('Giữ đủ trang giấy trong ảnh rồi bấm chụp để scan.');
        }
    } finally {
        if (request === cameraRequest && stream && screen === 'live' && pageViewer.hidden)
            detectionTimer = setTimeout(() => detectLive(request), 350);
    }
}

async function startCamera() {
    if (stream || busy || openingCamera) return;
    if (pages.length >= maxPages) { setStatus('Đã đủ 15 trang. Xóa bớt trang trước khi chụp thêm.'); return; }
    if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('Camera cần HTTPS hoặc localhost. Bạn có thể dùng nút Chọn ảnh.'); return;
    }
    const request = ++cameraRequest;
    openingCamera = true;
    openScanner('live');
    setStatus('Đang mở camera…');
    updateCaptureControls();
    try {
        let acquired;
        try {
            acquired = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: { ideal: facingMode }, width: { ideal: 2560 }, height: { ideal: 1920 } },
                audio: false,
            });
        } catch (error) {
            if (error.name !== 'OverconstrainedError' || request !== cameraRequest) throw error;
            acquired = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: { ideal: facingMode } }, audio: false,
            });
        }
        if (request !== cameraRequest) { acquired.getTracks().forEach(track => track.stop()); return; }
        stream = acquired;
        video.srcObject = stream;
        await video.play();
        if (request !== cameraRequest) return;
        fitSurface();
        const track = stream.getVideoTracks()[0];
        const capabilities = track.getCapabilities?.() || {};
        byId('torchButton').disabled = !capabilities.torch;
        if (capabilities.focusMode?.includes('continuous') && track.applyConstraints) {
            try { await track.applyConstraints({ advanced: [{ focusMode: 'continuous' }] }); }
            catch { /* Some browsers expose the capability but reject its control. */ }
        }
        if (request !== cameraRequest) return;
        byId('nativeCameraButton').title = `Camera ${video.videoWidth} × ${video.videoHeight} px. Chạm HD để chụp bằng camera điện thoại.`;
        setStatus('Đặt tài liệu vào khung hình · Bấm chụp khi sẵn sàng');
        detectionTimer = setTimeout(() => detectLive(request), 150);
    } catch (error) {
        if (request !== cameraRequest) return;
        stopCamera();
        if (capturedFile) openScanner('review');
        else closeScanner();
        setStatus(`Không thể mở camera: ${error.message}. Bạn có thể chọn ảnh có sẵn.`);
    } finally {
        if (request === cameraRequest) openingCamera = false;
        updateCaptureControls();
    }
}

function storeCapture(file) {
    clearCapture();
    capturedFile = file;
    capturedUrl = URL.createObjectURL(file);
    saveCapture.href = capturedUrl;
    saveCapture.download = file.name;
    saveCapture.hidden = false;
}

async function capturePhoto() {
    if (!stream || busy || openingCamera || !pageViewer.hidden) return;
    if (pages.length >= maxPages) { setStatus('Đã đủ 15 trang. Xuất PDF hoặc xóa bớt trang.'); return; }
    if (!video.videoWidth || !video.videoHeight) { setStatus('Camera chưa sẵn sàng.'); return; }
    setBusy(true);
    pauseDetection();
    liveCorners = null;
    paintCorners();
    let succeeded = false;
    try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        clearCapture();
        openScanner('processing');
        setStatus('Đang tự động scan ảnh vừa chụp…');
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.98));
        if (!blob) throw new Error('Không thể lưu ảnh vừa chụp. Vui lòng chụp lại.');
        storeCapture(new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' }));
        await scanFile(capturedFile);
        clearCapture();
        succeeded = true;
        setStatus(`Đã lưu trang ${pages.length} · Bấm chụp để quét trang tiếp theo.`);
    } catch (error) {
        if (capturedFile) {
            stopCamera();
            openScanner('review');
            setStatus(`${error.message} Thử scan lại hoặc chụp lại cho đủ bốn góc trang giấy.`);
        } else setStatus(error.message);
    }
    finally { setBusy(false); }
    if (stream && screen === 'processing') openScanner('live');
    if (succeeded && pages.length >= maxPages) {
        closeScanner();
        setStatus('Đã quét đủ 15 trang. Bạn có thể xuất PDF.');
    } else if (stream && screen === 'live') {
        detectionTimer = setTimeout(() => detectLive(cameraRequest), 900);
    } else if (capturedFile) scanCaptureButton.focus();
}

async function scanNativePhoto(file) {
    if (busy) return;
    if (pages.length >= maxPages) { setStatus('Đã đủ 15 trang.'); return; }
    setBusy(true);
    stopCamera();
    clearCapture();
    let succeeded = false;
    try {
        if (!file.type.startsWith('image/') || file.size > 12 * 1024 * 1024)
            throw new Error('Vui lòng chụp ảnh không quá 12 MB.');
        const url = URL.createObjectURL(file);
        const image = new Image();
        try {
            image.src = url;
            await image.decode();
            if (image.naturalWidth * image.naturalHeight > 20000000)
                throw new Error('Vui lòng chọn ảnh chụp không quá 20 triệu pixel.');
            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;
            canvas.getContext('2d').drawImage(image, 0, 0);
        } finally { URL.revokeObjectURL(url); }
        // Keep native JPEG/PNG bytes. Convert only formats the server cannot decode.
        if (!['image/jpeg', 'image/png'].includes(file.type)) {
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.98));
            if (!blob) throw new Error('Không thể đọc ảnh chụp HD.');
            file = new File([blob], `camera_hd_${Date.now()}.jpg`, { type: 'image/jpeg' });
        }
        storeCapture(file);
        openScanner('processing');
        setStatus('Đang scan ảnh chụp HD…');
        await scanFile(capturedFile);
        clearCapture();
        succeeded = true;
    } catch (error) {
        if (capturedFile) openScanner('review');
        else closeScannerAfterNativeError();
        setStatus(error.message || 'Không thể đọc ảnh chụp HD.');
    } finally { setBusy(false); }
    if (succeeded) {
        if (pages.length < maxPages) await startCamera();
        else closeScanner();
        setStatus(`Đã lưu trang ${pages.length} từ ảnh chụp HD.`);
    }
}

function closeScannerAfterNativeError() {
    // Decode failure has no usable photo to show; restore the library.
    screen = 'closed';
    scanner.hidden = true;
    document.body.classList.remove('scanner-open');
}

async function scanFiles(files) {
    if (busy) return;
    const available = maxPages - pages.length;
    if (available <= 0) { setStatus('Đã đủ 15 trang.'); return; }
    const selected = Array.from(files).slice(0, available);
    setBusy(true);
    let success = 0;
    const errors = [];
    try {
        for (const [index, file] of selected.entries()) {
            setStatus(`Đang scan ảnh ${index + 1}/${selected.length}…`);
            try {
                await scanFile(file);
                success++;
                renderPages();
            } catch (error) { errors.push(`${file.name}: ${error.message}`); }
        }
    } finally { setBusy(false); }
    setStatus(`Scan thành công ${success} ảnh. ${errors.join('; ')}`);
}

async function scanFile(file) {
    if (!file.type.startsWith('image/')) throw new Error('Tệp không phải ảnh.');
    if (file.size > 12 * 1024 * 1024) throw new Error('Ảnh vượt quá 12 MB.');
    const data = new FormData();
    data.append('image', file, file.name);
    const response = await fetch('/api/multipage/scan', { method: 'POST', body: data });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || 'Scan thất bại.');
    }
    const blob = await response.blob();
    pages.push({ blob, url: URL.createObjectURL(blob), source: file, corners: responseCorners(response) });
}

function responseCorners(response) {
    try {
        const corners = JSON.parse(response.headers?.get('X-Document-Corners') || 'null');
        return validCorners(corners) ? corners : null;
    } catch { return null; }
}

async function scanCapturedPhoto() {
    if (busy || screen !== 'review' || stream || openingCamera || !capturedFile) return;
    if (pages.length >= maxPages) { setStatus('Đã đủ 15 trang. Xóa bớt trang trước khi scan thêm.'); return; }
    setBusy(true);
    setStatus('Đang nắn phối cảnh và xử lý tài liệu…');
    let succeeded = false;
    try {
        await scanFile(capturedFile);
        clearCapture();
        succeeded = true;
        setStatus(`Đã scan ${pages.length} trang. Sẵn sàng lưu PDF.`);
    } catch (error) { setStatus(error.message); }
    finally { setBusy(false); }
    if (succeeded) {
        if (pages.length < maxPages) await startCamera();
        else closeScanner();
    }
}

function renderPages() {
    pageGrid.replaceChildren();
    byId('emptyState').hidden = pages.length > 0;
    byId('pageCount').textContent = `${pages.length} trang`;
    byId('capturedCount').textContent = pages.length;
    const last = byId('lastPage');
    last.hidden = pages.length === 0;
    if (pages.length) last.src = pages[pages.length - 1].url;
    else last.removeAttribute('src');
    byId('exportPdf').disabled = busy || pages.length === 0;
    byId('showPages').disabled = busy || openingCamera || pages.length === 0;
    pages.forEach((page, index) => {
        const card = document.createElement('article'); card.className = 'page-card';
        const image = document.createElement('img'); image.src = page.url; image.alt = `Trang ${index + 1}`;
        const preview = document.createElement('button'); preview.className = 'page-preview';
        preview.type = 'button'; preview.disabled = busy;
        preview.setAttribute('aria-label', `Mở ảnh trang ${index + 1}`);
        preview.addEventListener('click', () => openPageViewer(index));
        const previewLabel = document.createElement('span'); previewLabel.className = 'page-preview-label';
        previewLabel.textContent = 'Xem ảnh ↗';
        preview.append(image, previewLabel);
        const info = document.createElement('div'); info.className = 'page-info';
        const title = document.createElement('strong'); title.textContent = `Trang ${index + 1}`;
        const tools = document.createElement('div'); tools.className = 'page-tools';
        for (const [label, action, disabled] of [
            ['↑', () => movePage(index, -1), index === 0],
            ['↓', () => movePage(index, 1), index === pages.length - 1],
            ['Xóa', () => { URL.revokeObjectURL(pages.splice(index, 1)[0].url); renderPages(); }, false],
        ]) {
            const button = document.createElement('button'); button.textContent = label;
            button.disabled = busy || disabled;
            button.addEventListener('click', () => { if (!busy) action(); });
            tools.appendChild(button);
        }
        info.append(title, tools); card.append(preview, info); pageGrid.appendChild(card);
    });
}

function showViewerPage(index) {
    if (busy || editingCrop || index < 0 || index >= pages.length) return;
    viewerIndex = index;
    const page = pages[index];
    byId('viewerImage').src = page.url;
    byId('viewerImage').alt = `Bản quét trang ${index + 1}`;
    byId('viewerTitle').textContent = `Trang ${index + 1}`;
    byId('viewerCount').textContent = `${index + 1} / ${pages.length}`;
    byId('downloadPage').href = page.url;
    byId('downloadPage').download = `DocuScan-trang-${index + 1}.png`;
    byId('previousPage').disabled = index === 0;
    byId('nextPage').disabled = index === pages.length - 1;
    viewerZoomed = false;
    byId('viewerMessage').hidden = true;
    updateViewerZoom();
}

function openPageViewer(index) {
    if (busy || openingCamera || index < 0 || index >= pages.length) return;
    if (pageViewer.hidden) {
        viewerPreviousFocus = document.activeElement;
        pauseDetection();
    }
    showViewerPage(index);
    pageViewer.hidden = false;
    document.body.classList.add('viewer-open');
    byId('library').inert = true;
    scanner.inert = true;
    byId('closeViewer').focus();
}

function closePageViewer() {
    if (busy || pageViewer.hidden) return;
    const closedIndex = viewerIndex;
    endCropEditor();
    pageViewer.hidden = true;
    viewerIndex = -1;
    byId('viewerImage').removeAttribute('src');
    byId('downloadPage').removeAttribute('href');
    byId('library').inert = false;
    scanner.inert = false;
    document.body.classList.remove('viewer-open');
    const returnFocus = viewerPreviousFocus?.isConnected ? viewerPreviousFocus
        : scanner.hidden ? pageGrid.children[closedIndex]?.children[0] : byId('showPages');
    returnFocus?.focus();
    if (stream && screen === 'live' && !busy)
        detectionTimer = setTimeout(() => detectLive(cameraRequest), 150);
}

function updateViewerZoom() {
    const body = byId('viewerBody');
    body.classList.toggle('is-zoomed', viewerZoomed);
    body.scrollTop = body.scrollLeft = 0;
    byId('zoomViewer').setAttribute('aria-pressed', String(viewerZoomed));
    byId('zoomViewer').setAttribute('aria-label', viewerZoomed ? 'Vừa màn hình' : 'Xem ảnh ở kích thước gốc');
    byId('zoomViewer').textContent = viewerZoomed ? 'Vừa màn hình' : '100%';
}

function fitCropSurface() {
    if (!editingCrop || !cropSize) return;
    const body = byId('viewerBody');
    const availableWidth = Math.max(1, (body.clientWidth || window.innerWidth) - 64);
    const availableHeight = Math.max(1, (body.clientHeight || window.innerHeight * 0.6) - 64);
    const ratio = cropSize[0] / cropSize[1];
    const width = Math.min(availableWidth, availableHeight * ratio);
    byId('cropSurface').style.width = `${width}px`;
    byId('cropSurface').style.height = `${width / ratio}px`;
}

function paintCrop() {
    const scaled = cropCorners.map(point => point.map(value => value * 1000));
    byId('cropPolygon').setAttribute('points', scaled.map(point => point.join(',')).join(' '));
    byId('cropMask').setAttribute('d', `M0 0H1000V1000H0Z M${scaled.map(point => point.join(' ')).join('L')}Z`);
    Array.from(byId('cropHandles').children).forEach((handle, index) => {
        handle.style.left = `${cropCorners[index][0] * 100}%`;
        handle.style.top = `${cropCorners[index][1] * 100}%`;
    });
}

function moveCropCorner(index, point) {
    if (!editingCrop || busy) return;
    const updated = cropCorners.map(corner => corner.slice());
    updated[index] = point.map(value => Math.max(0, Math.min(1, value)));
    if (!validCorners(updated)) return;
    cropCorners = updated;
    paintCrop();
}

async function beginCropEditor() {
    if (busy || editingCrop || viewerIndex < 0) return;
    const page = pages[viewerIndex];
    if (!page.source) return;
    setBusy(true);
    const url = URL.createObjectURL(page.source);
    try {
        const image = new Image();
        image.src = url;
        await image.decode();
        cropSize = [image.naturalWidth, image.naturalHeight];
        cropSourceUrl = url;
        cropCorners = (page.corners || [[0.03, 0.03], [0.97, 0.03], [0.97, 0.97], [0.03, 0.97]])
            .map(point => point.slice());
        editingCrop = true;
        byId('cropImage').src = url;
        byId('cropSurface').hidden = false;
        byId('cropActions').hidden = false;
        byId('viewerNavigation').hidden = true;
        byId('viewerImage').hidden = true;
        for (const id of ['editCrop', 'zoomViewer', 'downloadPage']) byId(id).hidden = true;
        byId('viewerMessage').hidden = true;
        viewerZoomed = false;
        updateViewerZoom();
        byId('viewerBody').classList.add('is-editing');
        const handles = byId('cropHandles');
        handles.replaceChildren();
        ['trên trái', 'trên phải', 'dưới phải', 'dưới trái'].forEach((label, index) => {
            const handle = document.createElement('button');
            handle.type = 'button';
            handle.className = 'corner-handle';
            handle.textContent = index + 1;
            handle.setAttribute('aria-label', `Góc ${label}; dùng phím mũi tên để di chuyển`);
            let pointer = null;
            const move = event => {
                const rect = byId('cropSurface').getBoundingClientRect();
                if (rect.width && rect.height)
                    moveCropCorner(index, [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height]);
            };
            handle.addEventListener('pointerdown', event => {
                if (busy) return;
                event.preventDefault();
                pointer = event.pointerId;
                handle.setPointerCapture(pointer);
                handle.focus();
            });
            handle.addEventListener('pointermove', event => { if (pointer === event.pointerId) move(event); });
            handle.addEventListener('pointerup', event => {
                if (pointer !== event.pointerId) return;
                move(event);
                pointer = null;
            });
            handle.addEventListener('pointercancel', () => { pointer = null; });
            handle.addEventListener('lostpointercapture', () => { pointer = null; });
            handle.addEventListener('keydown', event => {
                const direction = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
                if (!direction) return;
                event.preventDefault();
                event.stopPropagation();
                const step = event.shiftKey ? 10 : 1;
                moveCropCorner(index, cropCorners[index].map((value, axis) => value + direction[axis] * step / cropSize[axis]));
            });
            handles.appendChild(handle);
        });
        fitCropSurface();
        paintCrop();
    } catch {
        URL.revokeObjectURL(url);
        byId('viewerMessage').textContent = 'Không thể mở ảnh gốc để chỉnh góc.';
        byId('viewerMessage').hidden = false;
    } finally { setBusy(false); }
    if (editingCrop) byId('cropHandles').children[0]?.focus();
}

function endCropEditor() {
    if (!editingCrop) return;
    editingCrop = false;
    if (cropSourceUrl) URL.revokeObjectURL(cropSourceUrl);
    cropSourceUrl = cropCorners = cropSize = null;
    byId('cropImage').removeAttribute('src');
    byId('cropSurface').hidden = true;
    byId('cropActions').hidden = true;
    byId('viewerNavigation').hidden = false;
    byId('viewerImage').hidden = false;
    byId('viewerBody').classList.remove('is-editing');
    for (const id of ['editCrop', 'zoomViewer', 'downloadPage']) byId(id).hidden = false;
    byId('viewerMessage').hidden = true;
}

async function applyCrop() {
    if (busy || !editingCrop || !validCorners(cropCorners)) return;
    const page = pages[viewerIndex];
    const selected = cropCorners.map(point => point.slice());
    setBusy(true);
    byId('pageViewer').setAttribute('aria-busy', 'true');
    byId('viewerMessage').textContent = 'Đang nắn lại từ ảnh gốc…';
    byId('viewerMessage').hidden = false;
    try {
        const data = new FormData();
        data.append('image', page.source, page.source.name || 'original.jpg');
        data.append('corners', JSON.stringify(selected));
        const response = await fetch('/api/multipage/rectify', { method: 'POST', body: data });
        if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'Chưa thể nắn lại ảnh.');
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        URL.revokeObjectURL(page.url);
        Object.assign(page, { blob, url, corners: selected });
        endCropEditor();
        setStatus(`Đã nắn lại trang ${viewerIndex + 1}.`);
    } catch (error) {
        byId('viewerMessage').textContent = error.message;
        byId('viewerMessage').hidden = false;
    } finally {
        byId('pageViewer').setAttribute('aria-busy', 'false');
        setBusy(false);
    }
    if (!editingCrop) { showViewerPage(viewerIndex); byId('editCrop').focus(); }
}

function movePage(index, direction) {
    const target = index + direction;
    if (busy || target < 0 || target >= pages.length) return;
    [pages[index], pages[target]] = [pages[target], pages[index]];
    renderPages();
}

async function exportPdf() {
    if (busy || !pages.length) return;
    setBusy(true); setStatus('Đang tạo PDF…');
    try {
        const data = new FormData();
        pages.forEach((page, index) => data.append('pages', page.blob, `page_${index + 1}.png`));
        const response = await fetch('/api/multipage/pdf', { method: 'POST', body: data });
        if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'Không thể tạo PDF.');
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement('a'); link.href = url; link.download = 'DocuScan.pdf';
        document.body.appendChild(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        setStatus(`Đã xuất PDF gồm ${pages.length} trang.`);
    } catch (error) { setStatus(error.message); }
    finally { setBusy(false); }
}

byId('openCamera').addEventListener('click', () => {
    if (capturedFile) { openScanner('review'); setStatus('Thử scan lại ảnh gốc hoặc chụp lại.'); }
    else return startCamera();
});
byId('captureButton').addEventListener('click', capturePhoto);
byId('scanCapture').addEventListener('click', scanCapturedPhoto);
byId('retakePhoto').addEventListener('click', startCamera);
byId('closeScanner').addEventListener('click', closeScanner);
byId('showPages').addEventListener('click', () => openPageViewer(pages.length - 1));
byId('closeViewer').addEventListener('click', closePageViewer);
byId('previousPage').addEventListener('click', () => showViewerPage(viewerIndex - 1));
byId('nextPage').addEventListener('click', () => showViewerPage(viewerIndex + 1));
byId('zoomViewer').addEventListener('click', () => {
    viewerZoomed = !viewerZoomed;
    updateViewerZoom();
});
byId('editCrop').addEventListener('click', beginCropEditor);
byId('cancelCrop').addEventListener('click', () => {
    if (busy) return;
    endCropEditor();
    byId('editCrop').focus();
});
byId('applyCrop').addEventListener('click', applyCrop);
byId('selectWholeImage').addEventListener('click', () => {
    if (busy || !editingCrop) return;
    cropCorners = [[0, 0], [1, 0], [1, 1], [0, 1]];
    paintCrop();
});
byId('chooseImage').addEventListener('click', () => byId('imageFiles').click());
byId('nativeCameraButton').addEventListener('click', () => {
    if (busy || openingCamera || pages.length >= maxPages) return;
    nativeCapturePending = true;
    stopCamera();
    const input = byId('nativeCameraInput');
    input.value = '';
    input.setAttribute('capture', facingMode === 'user' ? 'user' : 'environment');
    input.click();
});
byId('nativeCameraInput').addEventListener('change', async event => {
    nativeCapturePending = false;
    const file = event.target.files[0];
    event.target.value = '';
    if (file) await scanNativePhoto(file);
    else await startCamera();
});
byId('nativeCameraInput').addEventListener('cancel', async () => {
    if (!nativeCapturePending) return;
    nativeCapturePending = false;
    await startCamera();
});
byId('exportPdf').addEventListener('click', exportPdf);
byId('imageFiles').addEventListener('change', async event => {
    if (busy) return;
    const selected = Array.from(event.target.files);
    event.target.value = '';
    if (!selected.length) return;
    const available = maxPages - pages.length;
    if (!available) { setStatus('Đã đủ 15 trang.'); return; }
    stopCamera();
    closeScanner();
    clearCapture();
    await scanFiles(selected);
});
byId('switchCamera').addEventListener('click', async () => {
    if (busy || openingCamera || screen !== 'live') return;
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    stopCamera(); await startCamera();
});
byId('torchButton').addEventListener('click', async () => {
    if (!stream || busy) return;
    try {
        await stream.getVideoTracks()[0].applyConstraints({ advanced: [{ torch: !torch }] });
        torch = !torch;
        byId('torchButton').setAttribute('aria-pressed', String(torch));
        byId('torchButton').setAttribute('aria-label', torch ? 'Tắt đèn camera' : 'Bật đèn camera');
    } catch { setStatus('Camera này không hỗ trợ điều khiển đèn.'); }
});
byId('clearPages').addEventListener('click', () => {
    if (busy) return;
    pages.forEach(page => URL.revokeObjectURL(page.url)); pages.length = 0;
    renderPages(); setStatus('Đã xóa tất cả các trang.');
});
video.addEventListener('loadedmetadata', fitSurface);
window.addEventListener('resize', fitSurface);
window.addEventListener('resize', fitCropSurface);
window.addEventListener('pagehide', stopCamera);
document.addEventListener('keydown', event => {
    const dialog = !pageViewer.hidden ? pageViewer : !scanner.hidden ? scanner : null;
    if (!dialog) return;
    if (event.key === 'Escape' && !busy) {
        event.preventDefault();
        if (dialog === pageViewer && editingCrop) { endCropEditor(); byId('editCrop').focus(); }
        else if (dialog === pageViewer) closePageViewer();
        else closeScanner();
    }
    if (dialog === pageViewer && !editingCrop && ['ArrowLeft', 'ArrowRight'].includes(event.key)) {
        event.preventDefault();
        showViewerPage(viewerIndex + (event.key === 'ArrowLeft' ? -1 : 1));
    }
    if (event.key === 'Tab') {
        const focusable = Array.from(dialog.querySelectorAll('button:not(:disabled), a[href]')).filter(element => element.getClientRects().length);
        const first = focusable[0], last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
});
renderPages(); updateCaptureControls();
