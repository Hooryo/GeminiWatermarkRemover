/**
 * Gemini Watermark Removal Engine
 * 
 * Supports two methods:
 * 1. Reverse Alpha Blending - Mathematical precision for known watermark patterns
 * 2. LaMa AI Inpainting - AI-powered removal using ONNX model
 * 
 * Zero external dependencies (except ONNX Runtime for LaMa method)
 * 
 * Algorithm source: https://github.com/journey-ad/gemini-watermark-remover (MIT License)
 */

const WatermarkEngine = (function() {
    'use strict';

    // ==================== Constants ====================
    const MAX_ALPHA = 0.99;
    const SEARCH_CONFIG = {
        searchAreaRatio: 0.25,
        minConfidence: 0.3,
    };
    const MASK_CONFIG = {
        dilatePx: 4,
        featherPx: 3,
    };
    const LAMA_MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx?download=1";
    const MODEL_SIZE = 512;
    const MODEL_VERSION = "v2";

    // ==================== State ====================
    let alphaMap48 = null;
    let alphaMap96 = null;
    let initialized = false;
    let lamaWorker = null;
    let lamaReady = false;
    let lamaEP = 'wasm';
    let lamaLoadPromise = null;
    let lamaStatusCallback = null;

    // ==================== Utility Functions ====================

    /**
     * Load an image from a URL
     */
    function loadImageFromUrl(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load image from ${url}`));
            img.src = url;
        });
    }

    /**
     * Load an image from a File object
     */
    function loadImageFromFile(file) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            const url = URL.createObjectURL(file);
            img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(img);
            };
            img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error("Failed to load image"));
            };
            img.src = url;
        });
    }

    /**
     * Calculate alpha map from a watermark template image
     */
    function calculateAlphaMap(imageData) {
        const { width, height, data } = imageData;
        const alphaMap = new Float32Array(width * height);

        for (let i = 0; i < alphaMap.length; i++) {
            const idx = i * 4;
            const r = data[idx];
            const g = data[idx + 1];
            const b = data[idx + 2];
            const maxChannel = Math.max(r, g, b);
            alphaMap[i] = maxChannel / 255.0;
        }

        return alphaMap;
    }

    /**
     * Calculate correlation score between watermark template and image region
     */
    function calculateCorrelation(imageData, alphaMap, startX, startY, size) {
        const { width, data } = imageData;

        let avgBrightness = 0;
        let count = 0;

        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const imgX = startX + col;
                const imgY = startY + row;

                if (imgX < 0 || imgX >= imageData.width || imgY < 0 || imgY >= imageData.height) {
                    continue;
                }

                const imgIdx = (imgY * width + imgX) * 4;
                const brightness = (data[imgIdx] + data[imgIdx + 1] + data[imgIdx + 2]) / 3;
                avgBrightness += brightness;
                count++;
            }
        }

        if (count === 0) return 0;
        avgBrightness /= count;

        let correlation = 0;
        let alphaSum = 0;

        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const imgX = startX + col;
                const imgY = startY + row;

                if (imgX < 0 || imgX >= imageData.width || imgY < 0 || imgY >= imageData.height) {
                    continue;
                }

                const imgIdx = (imgY * width + imgX) * 4;
                const alphaIdx = row * size + col;
                const alpha = alphaMap[alphaIdx];

                if (alpha > 0.05) {
                    const brightness = (data[imgIdx] + data[imgIdx + 1] + data[imgIdx + 2]) / 3;
                    const brightnessDeviation = brightness - avgBrightness;

                    if (brightnessDeviation > 0) {
                        correlation += alpha * (brightnessDeviation / 255);
                    }
                    alphaSum += alpha;
                }
            }
        }

        if (alphaSum === 0) return 0;
        return correlation / alphaSum;
    }

    /**
     * Search for watermark in the image using template matching
     */
    function searchWatermark(imageData, alphaMap, size) {
        const { width, height } = imageData;

        const searchWidth = Math.floor(width * SEARCH_CONFIG.searchAreaRatio);
        const searchHeight = Math.floor(height * SEARCH_CONFIG.searchAreaRatio);

        const startSearchX = width - searchWidth;
        const startSearchY = height - searchHeight;

        let bestX = width - size - 32;
        let bestY = height - size - 32;
        let bestScore = -Infinity;

        const step = Math.max(1, Math.floor(size / 8));

        // Coarse search
        for (let y = startSearchY; y <= height - size; y += step) {
            for (let x = startSearchX; x <= width - size; x += step) {
                const score = calculateCorrelation(imageData, alphaMap, x, y, size);
                if (score > bestScore) {
                    bestScore = score;
                    bestX = x;
                    bestY = y;
                }
            }
        }

        // Fine search around best position
        const refineRange = step * 2;
        for (let y = bestY - refineRange; y <= bestY + refineRange; y++) {
            for (let x = bestX - refineRange; x <= bestX + refineRange; x++) {
                if (x < 0 || x > width - size || y < 0 || y > height - size) continue;
                const score = calculateCorrelation(imageData, alphaMap, x, y, size);
                if (score > bestScore) {
                    bestScore = score;
                    bestX = x;
                    bestY = y;
                }
            }
        }

        return {
            x: bestX,
            y: bestY,
            confidence: Math.max(0, Math.min(1, bestScore * 2)),
        };
    }

    /**
     * Apply reverse alpha blending to remove watermark
     * Formula: original = (watermarked - α × 255) / (1 - α)
     */
    function removeWatermark(imageData, alphaMap, position, watermarkSize) {
        const { width, height, data } = imageData;

        for (let row = 0; row < watermarkSize; row++) {
            for (let col = 0; col < watermarkSize; col++) {
                const imgX = position.x + col;
                const imgY = position.y + row;

                if (imgX < 0 || imgX >= width || imgY < 0 || imgY >= height) {
                    continue;
                }

                const imgIdx = (imgY * width + imgX) * 4;
                const alphaIdx = row * watermarkSize + col;

                const alpha = Math.min(alphaMap[alphaIdx], MAX_ALPHA);

                if (alpha > 0.001) {
                    for (let c = 0; c < 3; c++) {
                        const watermarked = data[imgIdx + c];
                        const original = (watermarked - alpha * 255) / (1.0 - alpha);
                        data[imgIdx + c] = Math.max(0, Math.min(255, Math.round(original)));
                    }
                }
            }
        }
    }

    /**
     * Dilate a binary mask by radius pixels
     */
    function dilateBinaryMask(maskCanvas, radius) {
        const r = Math.max(1, radius | 0);
        const w = maskCanvas.width;
        const h = maskCanvas.height;

        const srcCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
        const src = srcCtx.getImageData(0, 0, w, h).data;

        // Precompute disc offsets
        const offsets = [];
        for (let dy = -r; dy <= r; dy++) {
            for (let dx = -r; dx <= r; dx++) {
                if (dx * dx + dy * dy <= r * r) {
                    offsets.push({ dx, dy });
                }
            }
        }

        const out = new Uint8ClampedArray(w * h * 4);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let hit = false;
                for (let k = 0; k < offsets.length; k++) {
                    const nx = x + offsets[k].dx;
                    const ny = y + offsets[k].dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    const a = src[((ny * w + nx) << 2) + 3];
                    if (a > 127) {
                        hit = true;
                        break;
                    }
                }
                const o = (y * w + x) << 2;
                out[o] = 255;
                out[o + 1] = 255;
                out[o + 2] = 255;
                out[o + 3] = hit ? 255 : 0;
            }
        }

        const outCanvas = document.createElement('canvas');
        outCanvas.width = w;
        outCanvas.height = h;
        outCanvas.getContext('2d').putImageData(new ImageData(out, w, h), 0, 0);
        return outCanvas;
    }

    /**
     * Build mask from watermark alpha map for LaMa inpainting
     */
    function buildMaskFromAlphaMap(alphaMap, watermarkSize, position, imgW, imgH) {
        const mask = document.createElement('canvas');
        mask.width = imgW;
        mask.height = imgH;
        const mctx = mask.getContext('2d');
        const dst = mctx.createImageData(imgW, imgH);
        const d = dst.data;

        // Initialize all pixels as transparent (no inpainting needed)
        for (let i = 0; i < d.length; i += 4) {
            d[i] = 255;
            d[i + 1] = 255;
            d[i + 2] = 255;
            d[i + 3] = 0;
        }

        // Mark watermark area as white (needs inpainting)
        const alphaThreshold = 0.05;
        for (let row = 0; row < watermarkSize; row++) {
            for (let col = 0; col < watermarkSize; col++) {
                const imgX = position.x + col;
                const imgY = position.y + row;

                if (imgX < 0 || imgX >= imgW || imgY < 0 || imgY >= imgH) continue;

                const alphaIdx = row * watermarkSize + col;
                const alpha = alphaMap[alphaIdx];

                if (alpha > alphaThreshold) {
                    const pixelIdx = (imgY * imgW + imgX) * 4;
                    d[pixelIdx + 3] = 255;
                }
            }
        }

        mctx.putImageData(dst, 0, 0);

        // Dilate mask
        const dilated = dilateBinaryMask(mask, MASK_CONFIG.dilatePx);

        // Feather and re-binarize
        const final = document.createElement('canvas');
        final.width = imgW;
        final.height = imgH;
        const fctx = final.getContext('2d');

        if (MASK_CONFIG.featherPx > 0) {
            fctx.filter = `blur(${MASK_CONFIG.featherPx}px)`;
            fctx.drawImage(dilated, 0, 0);
            fctx.filter = 'none';

            const fi = fctx.getImageData(0, 0, imgW, imgH);
            const fd = fi.data;
            for (let i = 0; i < fd.length; i += 4) {
                const a = fd[i + 3] / 255;
                const bin = a > 0.4 ? 255 : 0;
                fd[i] = 255;
                fd[i + 1] = 255;
                fd[i + 2] = 255;
                fd[i + 3] = bin;
            }
            fctx.putImageData(fi, 0, 0);
        } else {
            fctx.drawImage(dilated, 0, 0);
        }

        return final;
    }

    // ==================== LaMa Worker Code ====================
    const LAMA_WORKER_CODE = `
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.js');

let session = null;
let modelEP = null;

function log(message) {
    try {
        self.postMessage({ type: 'log', message });
    } catch (e) {
        console.log(message);
    }
}

const DB_NAME = 'lama-cache';
const STORE = 'models';
const MODEL_VERSION = '${MODEL_VERSION}';

function idbOpen() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(STORE)) {
                db.createObjectStore(STORE);
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function idbGet(key) {
    const db = await idbOpen();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readonly');
        const store = tx.objectStore(STORE);
        const request = store.get(key);
        request.onsuccess = () => resolve(request.result || null);
        request.onerror = () => reject(request.error);
    });
}

async function idbSet(key, value) {
    const db = await idbOpen();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite');
        const store = tx.objectStore(STORE);
        const request = store.put(value, key);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

async function fetchModelWithCache(url) {
    const key = MODEL_VERSION + '::' + url;
    
    try {
        const cached = await idbGet(key);
        if (cached) {
            const buf = await cached.arrayBuffer();
            log('Using cached ONNX model');
            return buf;
        }
    } catch (e) {
        log('Cache read failed: ' + e);
    }
    
    log('Downloading LaMa model (~200MB)...');
    const resp = await fetch(url, { mode: 'cors', credentials: 'omit', cache: 'force-cache' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    
    const buf = await resp.arrayBuffer();
    
    try {
        await idbSet(key, new Blob([buf], { type: 'application/octet-stream' }));
        log('Model cached into IndexedDB');
    } catch (e) {
        log('Cache write failed: ' + e);
    }
    
    return buf;
}

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.wasm.numThreads = 1;

async function tryInitWebGPU() {
    if (!('gpu' in self.navigator)) return false;
    try {
        const adapter = await self.navigator.gpu.requestAdapter();
        if (!adapter) return false;
        const device = await adapter.requestDevice();
        ort.env.webgpu.adapter = adapter;
        ort.env.webgpu.device = device;
        return true;
    } catch (e) {
        return false;
    }
}

function canvasToImage01_CHW(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const { width, height } = canvas;
    const { data } = ctx.getImageData(0, 0, width, height);
    const area = width * height;
    const arr = new Float32Array(3 * area);
    
    for (let i = 0; i < area; i++) {
        const p = i * 4;
        arr[i] = data[p] / 255;
        arr[i + area] = data[p + 1] / 255;
        arr[i + 2 * area] = data[p + 2] / 255;
    }
    
    return { data: arr, shape: [1, 3, height, width] };
}

function canvasToMask01_CHW(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const { width, height } = canvas;
    const { data } = ctx.getImageData(0, 0, width, height);
    const area = width * height;
    const arr = new Float32Array(area);
    
    for (let i = 0; i < area; i++) {
        const p = i * 4;
        const lum = (0.2126 * data[p] + 0.7152 * data[p + 1] + 0.0722 * data[p + 2]) / 255;
        arr[i] = lum > 0.5 ? 1 : 0;
    }
    
    return { data: arr, shape: [1, 1, height, width] };
}

function tensorCHW_toRGBA_255(chw, W, H) {
    const area = W * H;
    const rgba = new Uint8ClampedArray(area * 4);
    
    for (let i = 0; i < area; i++) {
        let r = chw[i];
        let g = chw[i + area];
        let b = chw[i + 2 * area];
        
        r = Math.max(0, Math.min(255, r));
        g = Math.max(0, Math.min(255, g));
        b = Math.max(0, Math.min(255, b));
        
        const o = i * 4;
        rgba[o] = r;
        rgba[o + 1] = g;
        rgba[o + 2] = b;
        rgba[o + 3] = 255;
    }
    
    return rgba;
}

self.onmessage = async (e) => {
    const { type } = e.data || {};
    
    if (type === 'load') {
        try {
            log('Starting model load...');
            const modelBuffer = await fetchModelWithCache(e.data.modelUrl);
            
            const providers = [];
            if (await tryInitWebGPU()) {
                providers.push('webgpu');
                log('WebGPU available');
            }
            providers.push('wasm');
            
            let lastError = null;
            for (const ep of providers) {
                try {
                    log('Trying execution provider: ' + ep);
                    session = await ort.InferenceSession.create(modelBuffer, { executionProviders: [ep] });
                    modelEP = ep;
                    break;
                } catch (err) {
                    lastError = err;
                    log('EP ' + ep + ' failed: ' + err.message);
                }
            }
            
            if (!session) {
                throw new Error('Failed to initialize any execution provider: ' + (lastError?.message || lastError));
            }
            
            log('Model loaded with EP: ' + modelEP);
            self.postMessage({ type: 'loaded', ep: modelEP });
        } catch (err) {
            self.postMessage({ type: 'error', error: 'Model load failed: ' + err.message });
        }
    }
    
    if (type === 'run') {
        if (!session) {
            self.postMessage({ type: 'error', error: 'Session not ready' });
            return;
        }
        
        try {
            const { imgRGBA, maskRGBA, modelSize, outW, outH } = e.data;
            
            const img512 = new OffscreenCanvas(modelSize, modelSize);
            img512.getContext('2d').putImageData(
                new ImageData(new Uint8ClampedArray(imgRGBA), modelSize, modelSize), 
                0, 0
            );
            
            const mask512 = new OffscreenCanvas(modelSize, modelSize);
            mask512.getContext('2d').putImageData(
                new ImageData(new Uint8ClampedArray(maskRGBA), modelSize, modelSize), 
                0, 0
            );
            
            const tImg = canvasToImage01_CHW(img512);
            const tMask = canvasToMask01_CHW(mask512);
            
            const imgT = new ort.Tensor('float32', tImg.data, tImg.shape);
            const maskT = new ort.Tensor('float32', tMask.data, tMask.shape);
            
            const feeds = {};
            if (session.inputNames?.includes('image') && session.inputNames?.includes('mask')) {
                feeds['image'] = imgT;
                feeds['mask'] = maskT;
            } else {
                const names = session.inputNames || ['image', 'mask'];
                feeds[names[0]] = imgT;
                feeds[names[1] || 'mask'] = maskT;
            }
            
            log('Running inference...');
            
            const results = await session.run(feeds);
            const outName = session.outputNames?.[0] || Object.keys(results)[0];
            const outT = results[outName];
            
            const W = outT.dims[3];
            const H = outT.dims[2];
            const outData = outT.data instanceof Float32Array ? outT.data : await outT.getData();
            
            const rgba512 = tensorCHW_toRGBA_255(outData, W, H);
            
            const out512 = new OffscreenCanvas(W, H);
            out512.getContext('2d').putImageData(new ImageData(rgba512, W, H), 0, 0);
            
            const outFull = new OffscreenCanvas(outW, outH);
            const octx = outFull.getContext('2d');
            octx.imageSmoothingEnabled = true;
            octx.imageSmoothingQuality = 'high';
            octx.drawImage(out512, 0, 0, W, H, 0, 0, outW, outH);
            
            const outImg = octx.getImageData(0, 0, outW, outH);
            self.postMessage(
                { type: 'result', rgba: outImg.data.buffer, w: outW, h: outH, ep: modelEP },
                [outImg.data.buffer]
            );
        } catch (err) {
            self.postMessage({ type: 'error', error: 'Inference failed: ' + err.message });
        }
    }
};
`;

    // ==================== Public API ====================

    function updateLamaStatus(partial) {
        if (lamaStatusCallback) {
            lamaStatusCallback({
                loading: false,
                ready: lamaReady,
                error: null,
                executionProvider: lamaEP,
                progress: '',
                ...partial,
            });
        }
    }

    return {
        /**
         * Initialize the engine by loading alpha maps
         */
        async initialize() {
            if (initialized) return;

            try {
                // Load 48x48 alpha map
                const img48 = await loadImageFromUrl("bg_48.png");
                const canvas48 = document.createElement("canvas");
                canvas48.width = img48.width;
                canvas48.height = img48.height;
                const ctx48 = canvas48.getContext("2d");
                ctx48.drawImage(img48, 0, 0);
                const imageData48 = ctx48.getImageData(0, 0, img48.width, img48.height);
                alphaMap48 = calculateAlphaMap(imageData48);

                // Load 96x96 alpha map
                const img96 = await loadImageFromUrl("bg_96.png");
                const canvas96 = document.createElement("canvas");
                canvas96.width = img96.width;
                canvas96.height = img96.height;
                const ctx96 = canvas96.getContext("2d");
                ctx96.drawImage(img96, 0, 0);
                const imageData96 = ctx96.getImageData(0, 0, img96.width, img96.height);
                alphaMap96 = calculateAlphaMap(imageData96);

                initialized = true;
            } catch (error) {
                console.error("Failed to initialize WatermarkEngine:", error);
                throw error;
            }
        },

        /**
         * Set callback for LaMa status updates
         */
        setLamaStatusCallback(callback) {
            lamaStatusCallback = callback;
        },

        /**
         * Initialize LaMa worker and load model
         */
        async initializeLama() {
            if (lamaReady) return;
            if (lamaLoadPromise) return lamaLoadPromise;

            lamaLoadPromise = new Promise((resolve, reject) => {
                updateLamaStatus({ loading: true, progress: 'Initializing worker...' });

                const blob = new Blob([LAMA_WORKER_CODE], { type: "application/javascript" });
                const url = URL.createObjectURL(blob);
                lamaWorker = new Worker(url);

                lamaWorker.onmessage = (msg) => {
                    const { type } = msg.data || {};

                    if (type === 'log') {
                        console.log('🧠 LaMa Worker:', msg.data.message);
                        updateLamaStatus({ progress: msg.data.message });
                        return;
                    }

                    if (type === 'loaded') {
                        lamaReady = true;
                        lamaEP = msg.data.ep || 'wasm';
                        updateLamaStatus({
                            loading: false,
                            ready: true,
                            executionProvider: lamaEP,
                            progress: `Model ready (${lamaEP})`,
                        });
                        resolve();
                    } else if (type === 'error') {
                        const error = msg.data.error;
                        updateLamaStatus({ loading: false, error, progress: '' });
                        reject(new Error(error));
                    }
                };

                lamaWorker.postMessage({ type: 'load', modelUrl: LAMA_MODEL_URL });
            });

            return lamaLoadPromise;
        },

        /**
         * Check if LaMa is ready
         */
        isLamaReady() {
            return lamaReady;
        },

        /**
         * Detect watermark in image
         */
        detectWatermark(imageData) {
            if (!initialized) {
                throw new Error("WatermarkEngine not initialized");
            }

            const result48 = searchWatermark(imageData, alphaMap48, 48);
            const result96 = searchWatermark(imageData, alphaMap96, 96);

            if (result96.confidence > result48.confidence && result96.confidence >= SEARCH_CONFIG.minConfidence) {
                return {
                    size: 96,
                    position: { x: result96.x, y: result96.y },
                    confidence: result96.confidence,
                    detected: result96.confidence >= SEARCH_CONFIG.minConfidence,
                };
            } else {
                return {
                    size: 48,
                    position: { x: result48.x, y: result48.y },
                    confidence: result48.confidence,
                    detected: result48.confidence >= SEARCH_CONFIG.minConfidence,
                };
            }
        },

        /**
         * Process image using Alpha Blending method
         */
        async processWithAlphaBlending(img, startTime) {
            const canvas = document.createElement("canvas");
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);

            const originalImageData = ctx.getImageData(0, 0, img.width, img.height);
            const processedImageData = new ImageData(
                new Uint8ClampedArray(originalImageData.data),
                originalImageData.width,
                originalImageData.height
            );

            const watermarkInfo = this.detectWatermark(originalImageData);

            if (watermarkInfo.detected) {
                const alphaMap = watermarkInfo.size === 48 ? alphaMap48 : alphaMap96;
                removeWatermark(processedImageData, alphaMap, watermarkInfo.position, watermarkInfo.size);
            }

            return {
                success: true,
                originalImageData,
                processedImageData,
                watermarkInfo,
                processingTime: performance.now() - startTime,
                method: 'alpha-blending',
            };
        },

        /**
         * Process image using LaMa AI inpainting
         */
        async processWithLama(img, startTime) {
            if (!lamaWorker || !lamaReady) {
                throw new Error("LaMa worker not ready. Call initializeLama() first.");
            }

            const canvas = document.createElement("canvas");
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);

            const originalImageData = ctx.getImageData(0, 0, img.width, img.height);
            const watermarkInfo = this.detectWatermark(originalImageData);
            const alphaMap = watermarkInfo.size === 48 ? alphaMap48 : alphaMap96;

            // Build mask
            const maskCanvas = buildMaskFromAlphaMap(
                alphaMap,
                watermarkInfo.size,
                watermarkInfo.position,
                img.width,
                img.height
            );

            // Resize to MODEL_SIZE
            const img512 = document.createElement("canvas");
            img512.width = MODEL_SIZE;
            img512.height = MODEL_SIZE;
            img512.getContext("2d").drawImage(canvas, 0, 0, MODEL_SIZE, MODEL_SIZE);

            const mask512 = document.createElement("canvas");
            mask512.width = MODEL_SIZE;
            mask512.height = MODEL_SIZE;
            mask512.getContext("2d").drawImage(maskCanvas, 0, 0, MODEL_SIZE, MODEL_SIZE);

            const imgData512 = img512.getContext("2d").getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
            const maskData512 = mask512.getContext("2d").getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);

            return new Promise((resolve, reject) => {
                const handler = (msg) => {
                    const { type } = msg.data || {};

                    if (type === 'log') {
                        console.log('🧠 LaMa:', msg.data.message);
                        return;
                    }

                    if (type === 'result') {
                        lamaWorker.removeEventListener('message', handler);
                        const { rgba, w, h } = msg.data;
                        const out = new Uint8ClampedArray(rgba);
                        const processedImageData = new ImageData(out, w, h);

                        resolve({
                            success: true,
                            originalImageData,
                            processedImageData,
                            watermarkInfo,
                            processingTime: performance.now() - startTime,
                            method: 'lama-ai',
                        });
                    } else if (type === 'error') {
                        lamaWorker.removeEventListener('message', handler);
                        reject(new Error(msg.data.error));
                    }
                };

                lamaWorker.addEventListener('message', handler);

                lamaWorker.postMessage(
                    {
                        type: 'run',
                        imgRGBA: imgData512.data.buffer,
                        maskRGBA: maskData512.data.buffer,
                        modelSize: MODEL_SIZE,
                        outW: img.width,
                        outH: img.height,
                    },
                    [imgData512.data.buffer, maskData512.data.buffer]
                );
            });
        },

        /**
         * Process a single image file
         */
        async processImage(file, method = 'alpha-blending') {
            const startTime = performance.now();

            await this.initialize();

            const img = await loadImageFromFile(file);

            if (method === 'lama-ai') {
                await this.initializeLama();
                return this.processWithLama(img, startTime);
            } else {
                return this.processWithAlphaBlending(img, startTime);
            }
        },

        /**
         * Convert ImageData to Blob
         */
        imageDataToBlob(imageData, type = "image/png", quality = 1.0) {
            return new Promise((resolve, reject) => {
                const canvas = document.createElement("canvas");
                canvas.width = imageData.width;
                canvas.height = imageData.height;
                const ctx = canvas.getContext("2d");
                ctx.putImageData(imageData, 0, 0);

                canvas.toBlob(
                    (blob) => {
                        if (blob) {
                            resolve(blob);
                        } else {
                            reject(new Error("Failed to create blob"));
                        }
                    },
                    type,
                    quality
                );
            });
        },

        /**
         * Dispose resources
         */
        dispose() {
            if (lamaWorker) {
                lamaWorker.terminate();
                lamaWorker = null;
                lamaReady = false;
                lamaLoadPromise = null;
            }
        }
    };
})();

