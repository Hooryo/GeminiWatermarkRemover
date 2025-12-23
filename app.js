/**
 * Gemini Watermark Remover - Application Logic
 * Handles UI interactions and file processing
 */

(function() {
    'use strict';

    // ==================== State ====================
    let files = [];
    let processedFiles = [];
    let currentMethod = 'alpha-blending';

    // ==================== DOM Elements ====================
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const actions = document.getElementById('actions');
    const clearBtn = document.getElementById('clear-btn');
    const processBtn = document.getElementById('process-btn');
    const progress = document.getElementById('progress');
    const progressText = document.getElementById('progress-text');
    const progressFill = document.getElementById('progress-fill');
    const results = document.getElementById('results');
    const downloadAll = document.getElementById('download-all');
    const downloadAllBtn = document.getElementById('download-all-btn');
    const errorMessage = document.getElementById('error-message');
    const lamaStatus = document.getElementById('lama-status');
    const methodBtns = document.querySelectorAll('.method-btn');

    // ==================== Utility Functions ====================

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.add('show');
        setTimeout(() => {
            errorMessage.classList.remove('show');
        }, 5000);
    }

    function hideError() {
        errorMessage.classList.remove('show');
    }

    // ==================== Method Selection ====================

    methodBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            methodBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMethod = btn.dataset.method;

            // Pre-load LaMa model when selected
            if (currentMethod === 'lama-ai' && !WatermarkEngine.isLamaReady()) {
                WatermarkEngine.initializeLama().catch(err => {
                    console.error('Failed to preload LaMa:', err);
                });
            }
        });
    });

    // Set up LaMa status callback
    WatermarkEngine.setLamaStatusCallback((status) => {
        if (status.ready) {
            lamaStatus.textContent = `Ready (${status.executionProvider?.toUpperCase()})`;
            lamaStatus.style.color = '#22c55e';
        } else if (status.loading) {
            lamaStatus.textContent = status.progress || 'Loading...';
            lamaStatus.style.color = '#eab308';
        } else if (status.error) {
            lamaStatus.textContent = 'Error: ' + status.error;
            lamaStatus.style.color = '#ef4444';
        }
    });

    // ==================== File Handling ====================

    function handleFiles(selectedFiles) {
        const imageFiles = Array.from(selectedFiles).filter(file =>
            file.type === 'image/jpeg' ||
            file.type === 'image/png' ||
            file.type === 'image/webp' ||
            file.name.toLowerCase().match(/\.(jpg|jpeg|png|webp)$/)
        );

        if (imageFiles.length === 0) {
            showError('Please select image files (JPEG, PNG, or WebP)');
            return;
        }

        files = imageFiles;
        processedFiles = [];
        hideError();
        renderFileList();
        results.innerHTML = '';
        downloadAll.style.display = 'none';
    }

    function renderFileList() {
        if (files.length === 0) {
            fileList.innerHTML = '';
            actions.style.display = 'none';
            return;
        }

        fileList.innerHTML = files.map((file, index) => `
            <div class="file-item">
                <div class="file-info">
                    <span>🖼️</span>
                    <div>
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${formatFileSize(file.size)}</div>
                    </div>
                </div>
                <button class="remove-btn" data-index="${index}">✕</button>
            </div>
        `).join('');

        actions.style.display = 'flex';

        // Add remove button handlers
        fileList.querySelectorAll('.remove-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                files = files.filter((_, i) => i !== index);
                renderFileList();
            });
        });
    }

    // ==================== Upload Area Events ====================

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
        fileInput.value = '';
    });

    // ==================== Action Buttons ====================

    clearBtn.addEventListener('click', () => {
        files = [];
        processedFiles.forEach(f => {
            URL.revokeObjectURL(f.originalUrl);
            URL.revokeObjectURL(f.processedUrl);
        });
        processedFiles = [];
        renderFileList();
        results.innerHTML = '';
        downloadAll.style.display = 'none';
    });

    processBtn.addEventListener('click', processFiles);

    downloadAllBtn.addEventListener('click', downloadAllFiles);

    // ==================== Processing ====================

    async function processFiles() {
        if (files.length === 0) return;

        processBtn.disabled = true;
        progress.style.display = 'block';
        results.innerHTML = '';
        downloadAll.style.display = 'none';

        const newProcessedFiles = [];

        try {
            await WatermarkEngine.initialize();

            if (currentMethod === 'lama-ai' && !WatermarkEngine.isLamaReady()) {
                progressText.textContent = 'Loading AI model...';
                await WatermarkEngine.initializeLama();
            }

            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                progressText.textContent = `Processing ${i + 1}/${files.length}: ${file.name}`;
                progressFill.style.width = `${((i + 1) / files.length) * 100}%`;

                try {
                    const result = await WatermarkEngine.processImage(file, currentMethod);
                    const outputType = file.type || 'image/png';
                    const quality = outputType === 'image/jpeg' || outputType === 'image/webp' ? 0.92 : 1.0;
                    const processedBlob = await WatermarkEngine.imageDataToBlob(result.processedImageData, outputType, quality);

                    newProcessedFiles.push({
                        id: `${Date.now()}-${i}`,
                        originalFile: file,
                        originalUrl: URL.createObjectURL(file),
                        processedUrl: URL.createObjectURL(processedBlob),
                        processedBlob,
                        watermarkInfo: result.watermarkInfo,
                        processingTime: result.processingTime,
                        method: result.method,
                    });
                } catch (err) {
                    console.error(`Failed to process ${file.name}:`, err);
                }
            }

            processedFiles = newProcessedFiles;
            renderResults();

        } catch (err) {
            console.error('Processing failed:', err);
            showError('Failed to process images. Please try again.');
        } finally {
            processBtn.disabled = false;
            progress.style.display = 'none';
            progressFill.style.width = '0%';
        }
    }

    // ==================== Results Rendering ====================

    function renderResults() {
        if (processedFiles.length === 0) {
            results.innerHTML = '';
            downloadAll.style.display = 'none';
            return;
        }

        results.innerHTML = processedFiles.map(file => {
            const methodBadgeClass = file.method === 'lama-ai' ? 'badge-purple' : 'badge-blue';
            const methodLabel = file.method === 'lama-ai' ? 'LaMa AI' : 'Alpha Blending';

            return `
                <div class="result-item" data-id="${file.id}">
                    <div class="result-header">
                        <div class="result-info">
                            <div class="result-name">${file.originalFile.name}</div>
                            <div class="result-meta">
                                <span class="badge ${methodBadgeClass}">${methodLabel}</span>
                                <span>${file.processingTime.toFixed(0)}ms</span>
                            </div>
                        </div>
                        <button class="btn btn-primary download-btn" data-id="${file.id}">Download</button>
                    </div>

                    <div class="image-comparison">
                        <div class="comparison-item">
                            <div class="comparison-label">Original</div>
                            <div class="comparison-image">
                                <img src="${file.originalUrl}" alt="Original">
                            </div>
                        </div>
                        <div class="comparison-item">
                            <div class="comparison-label">Processed</div>
                            <div class="comparison-image">
                                <img src="${file.processedUrl}" alt="Processed">
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Add download button handlers
        results.querySelectorAll('.download-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = e.target.dataset.id;
                const file = processedFiles.find(f => f.id === id);
                if (file) downloadFile(file);
            });
        });

        // Show download all button if multiple files
        if (processedFiles.length > 1) {
            downloadAll.style.display = 'block';
        }

        // Clear file list since we're showing results
        fileList.innerHTML = '';
        actions.style.display = 'none';
    }

    // ==================== Download Functions ====================

    function downloadFile(file) {
        const link = document.createElement('a');
        link.href = file.processedUrl;
        const ext = file.originalFile.name.split('.').pop() || 'png';
        const baseName = file.originalFile.name.replace(/\.[^/.]+$/, '');
        link.download = `${baseName}_no_watermark.${ext}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    async function downloadAllFiles() {
        if (processedFiles.length === 0) return;

        // Simple approach: download each file individually
        // For a more sophisticated approach, you'd need to include JSZip
        for (const file of processedFiles) {
            downloadFile(file);
            // Small delay between downloads
            await new Promise(resolve => setTimeout(resolve, 300));
        }
    }

    // ==================== Initialize ====================

    // Pre-initialize engine on page load
    WatermarkEngine.initialize().catch(console.error);

})();

