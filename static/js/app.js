(() => {
  'use strict';
  const form = document.getElementById('processForm');
  const input = document.getElementById('imageInput');
  const area = document.getElementById('uploadArea');
  const preview = document.getElementById('previewWrapper');
  const image = document.getElementById('imagePreview');
  const prompt = document.getElementById('uploadPrompt');
  const info = document.getElementById('fileInfo');
  const error = document.getElementById('fileError');
  const button = document.getElementById('processButton');
  const range = document.getElementById('strengthRange');
  let objectUrl;
  let fileVersion = 0;

  function resetPreview() {
    fileVersion += 1;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = undefined;
    image.removeAttribute('src');
    preview.classList.add('hidden');
    info.classList.add('hidden');
    prompt.classList.remove('hidden');
    input.setCustomValidity('');
  }
  function showError(message) {
    input.value = '';
    resetPreview();
    error.textContent = message;
    error.classList.remove('hidden');
    input.setAttribute('aria-invalid', 'true');
  }
  async function selectFile() {
    resetPreview();
    error.classList.add('hidden');
    input.removeAttribute('aria-invalid');
    const file = input.files[0];
    if (!file) return;
    if (!/\.(jpe?g|png|bmp|webp)$/i.test(file.name)) {
      showError('Vui lòng chọn ảnh JPG, PNG, BMP hoặc WEBP.');
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      showError('Ảnh vượt quá 15 MB. Vui lòng chọn ảnh nhỏ hơn.');
      return;
    }
    const version = fileVersion;
    objectUrl = URL.createObjectURL(file);
    image.src = objectUrl;
    input.setCustomValidity('Vui lòng chờ kiểm tra ảnh.');
    try {
      await image.decode();
      if (version !== fileVersion) return;
      input.setCustomValidity('');
      document.getElementById('fileName').textContent = file.name;
      document.getElementById('fileSize').textContent = file.size >= 1024 * 1024
        ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
        : `${Math.max(1, Math.round(file.size / 1024))} KB`;
      prompt.classList.add('hidden');
      preview.classList.remove('hidden');
      info.classList.remove('hidden');
    } catch {
      if (version === fileVersion) showError('Không đọc được ảnh. Vui lòng chọn một ảnh khác.');
    }
  }
  input.addEventListener('change', selectFile);
  document.getElementById('removeFile').addEventListener('click', () => {
    input.value = '';
    resetPreview();
    error.classList.add('hidden');
    input.removeAttribute('aria-invalid');
    input.focus();
  });
  ['dragenter', 'dragover'].forEach(type => area.addEventListener(type, event => {
    event.preventDefault();
    area.classList.add('dragging');
  }));
  area.addEventListener('dragleave', event => {
    if (!area.contains(event.relatedTarget)) area.classList.remove('dragging');
  });
  area.addEventListener('drop', event => {
    event.preventDefault();
    area.classList.remove('dragging');
    const files = event.dataTransfer.files;
    if (!files.length) return;
    if (files.length > 1) {
      showError('Mỗi lần xử lý một ảnh. Vui lòng chọn một tệp.');
      return;
    }
    input.files = files;
    selectFile();
  });
  // Avoid navigating away when a file is dropped outside the upload area.
  window.addEventListener('dragover', event => event.preventDefault());
  window.addEventListener('drop', event => event.preventDefault());

  function updateRange() {
    document.getElementById('rangeMinimum').textContent = `Nhẹ · ${range.min}`;
    document.getElementById('strengthValue').value = Number(range.value).toFixed(1);
    const progress = (range.value - range.min) / (range.max - range.min) * 100;
    range.style.setProperty('--range-progress', `${progress}%`);
  }
  if (range) {
    range.addEventListener('input', updateRange);
    document.querySelectorAll('[name="sharpening_method"]').forEach(radio => {
      radio.addEventListener('change', () => {
        const highboost = radio.value === 'highboost';
        range.min = highboost ? '1.1' : '0.5';
        if (highboost && Number(range.value) <= 1) range.value = '1.5';
        updateRange();
      });
    });
    if (document.querySelector('[name="sharpening_method"]:checked').value === 'highboost') {
      range.min = '1.1';
    }
    updateRange();
  }
  form.addEventListener('submit', () => {
    button.disabled = true;
    button.querySelector('span').textContent = 'Đang xử lý…';
    form.setAttribute('aria-busy', 'true');
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('formFootnote').classList.add('hidden');
  });
  window.addEventListener('pageshow', () => {
    // Browsers may restore a disabled submit button when navigating back.
    button.disabled = false;
    button.querySelector('span').textContent = form.elements.action.value === 'compare'
      ? 'Chạy so sánh' : 'Quét tài liệu';
    form.removeAttribute('aria-busy');
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('formFootnote').classList.remove('hidden');
  });
})();
