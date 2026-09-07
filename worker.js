// UVR Web Worker – MDX-Net vocal separation (UVR_MDXNET_9482)
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js");

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

let session = null;

const SR = 44100;
const N_FFT = 4096;
const HOP = 1024;
const WIN = 4096;
const DIM_F = 2048;
const DIM_T = 256;
const DIM_C = 4;

const DB_NAME = "UVR_Model_Cache_v5";
const STORE_NAME = "models";

function openDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) db.createObjectStore(STORE_NAME);
        };
        req.onsuccess = (e) => resolve(e.target.result);
        req.onerror = (e) => reject(e.target.error);
    });
}
function getCache(key) {
    return openDB().then((db) => new Promise((resolve, reject) => {
        const r = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(key);
        r.onsuccess = () => resolve(r.result);
        r.onerror = () => reject(r.error);
    }));
}
function setCache(key, val) {
    return openDB().then((db) => new Promise((resolve, reject) => {
        const r = db.transaction(STORE_NAME, "readwrite").objectStore(STORE_NAME).put(val, key);
        r.onsuccess = () => resolve();
        r.onerror = () => reject(r.error);
    }));
}
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function fetchAsArrayBuffer(url, onProgress, options = {}) {
    const { maxRetries = 3, timeoutMs = 300000, fallbackUrls = [] } = options;
    const urls = [url, ...fallbackUrls];
    let lastError = null;
    for (const currentUrl of urls) {
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                if (attempt > 1) {
                    await sleep(Math.min(1000 * 2 ** (attempt - 2), 8000));
                    self.postMessage({ status: "status", data: `Retry ${attempt}/${maxRetries}...` });
                }
                const controller = new AbortController();
                const tid = setTimeout(() => controller.abort(), timeoutMs);
                const response = await fetch(currentUrl, { signal: controller.signal, cache: "no-cache" });
                clearTimeout(tid);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const contentLength = +response.headers.get("Content-Length") || 0;
                const reader = response.body.getReader();
                let received = 0;
                const chunks = [];
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    received += value.length;
                    if (onProgress) {
                        onProgress(contentLength > 0
                            ? Math.min(99, Math.round((received / contentLength) * 100))
                            : Math.min(99, Math.round(received / 1e6)));
                    }
                }
                if (received < 1e6) throw new Error(`Model too small (${received} bytes)`);
                const out = new Uint8Array(received);
                let p = 0;
                for (const c of chunks) { out.set(c, p); p += c.length; }
                if (onProgress) onProgress(100);
                return out.buffer;
            } catch (err) {
                lastError = err;
                self.postMessage({ status: "status", data: `Download failed: ${err.message || err}` });
            }
        }
    }
    throw new Error(`Failed to download model: ${lastError && lastError.message}`);
}

function bitReverse(i, n) {
    let rev = 0, t = n >> 1;
    while (t > 0) { rev = (rev << 1) | (i & 1); i >>= 1; t >>= 1; }
    return rev;
}

function fft(re, im) {
    const n = re.length;
    for (let i = 0; i < n; i++) {
        const j = bitReverse(i, n);
        if (i < j) {
            let t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const ang = -2 * Math.PI / len;
        const wr0 = Math.cos(ang), wi0 = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let wr = 1, wi = 0;
            for (let j = 0; j < len / 2; j++) {
                const u_re = re[i + j], u_im = im[i + j];
                const i2 = i + j + len / 2;
                const v_re = re[i2] * wr - im[i2] * wi;
                const v_im = re[i2] * wi + im[i2] * wr;
                re[i + j] = u_re + v_re; im[i + j] = u_im + v_im;
                re[i2] = u_re - v_re; im[i2] = u_im - v_im;
                const nwr = wr * wr0 - wi * wi0;
                wi = wr * wi0 + wi * wr0;
                wr = nwr;
            }
        }
    }
}

function hannWindow(n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
    return w;
}

function resample(input, fromRate, toRate) {
    if (fromRate === toRate) return input;
    const ratio = toRate / fromRate;
    const outLen = Math.max(1, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
        const src = i / ratio;
        const i0 = Math.floor(src);
        const i1 = Math.min(i0 + 1, input.length - 1);
        const f = src - i0;
        out[i] = input[i0] * (1 - f) + input[i1] * f;
    }
    return out;
}

function stft(signal, nfft, hop, window) {
    const pad = nfft;
    const padded = new Float32Array(signal.length + 2 * pad);
    padded.set(signal, pad);
    const numFrames = Math.floor((padded.length - nfft) / hop) + 1;
    const nBins = nfft / 2 + 1;
    const real = new Float32Array(numFrames * nBins);
    const imag = new Float32Array(numFrames * nBins);
    const re = new Float32Array(nfft);
    const im = new Float32Array(nfft);

    for (let t = 0; t < numFrames; t++) {
        const off = t * hop;
        for (let i = 0; i < nfft; i++) {
            re[i] = padded[off + i] * window[i];
            im[i] = 0;
        }
        fft(re, im);
        for (let b = 0; b < nBins; b++) {
            real[t * nBins + b] = re[b];
            imag[t * nBins + b] = im[b];
        }
    }
    return { real, imag, numFrames, nBins };
}

function istft(real, imag, nfft, hop, window, outLen) {
    const numFrames = real.length / (nfft / 2 + 1);
    const nBins = nfft / 2 + 1;
    const pad = nfft;
    const paddedLen = (numFrames - 1) * hop + nfft;
    const accum = new Float32Array(paddedLen);
    const wsum = new Float32Array(paddedLen);
    const re = new Float32Array(nfft);
    const im = new Float32Array(nfft);

    for (let t = 0; t < numFrames; t++) {
        for (let b = 0; b < nBins; b++) {
            re[b] = real[t * nBins + b];
            im[b] = imag[t * nBins + b];
        }
        for (let b = 1; b < nfft / 2; b++) {
            re[nfft - b] = re[b];
            im[nfft - b] = -im[b];
        }
        im[0] = 0;
        im[nfft / 2] = 0;
        fft(re, im);
        for (let i = 0; i < nfft; i++) re[i] /= nfft;
        const off = t * hop;
        for (let i = 0; i < nfft; i++) {
            accum[off + i] += re[i] * window[i];
            wsum[off + i] += window[i] * window[i];
        }
    }
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
        const idx = pad + i;
        if (idx < paddedLen) {
            const d = wsum[idx];
            out[i] = d > 1e-8 ? accum[idx] / d : 0;
        }
    }
    return out;
}

function packChunk(realL, imagL, realR, imagR, f0, numFrames, nBins) {
    const data = new Float32Array(DIM_C * DIM_F * DIM_T);
    for (let c = 0; c < 4; c++) {
        const srcReal = c === 0 ? realL : c === 2 ? realR : null;
        const srcImag = c === 1 ? imagL : c === 3 ? imagR : null;
        const src = srcReal || srcImag;
        for (let b = 0; b < DIM_F; b++) {
            for (let t = 0; t < DIM_T; t++) {
                const sf = f0 + t;
                let v = 0;
                if (sf < numFrames && b < nBins) v = src[sf * nBins + b] || 0;
                data[c * DIM_F * DIM_T + b * DIM_T + t] = v;
            }
        }
    }
    return data;
}

function unpackChunk(outData, realL, imagL, realR, imagR, f0, numFrames, nBins, chunkFrames) {
    for (let c = 0; c < 4; c++) {
        const dst = c === 0 ? realL : c === 1 ? imagL : c === 2 ? realR : imagR;
        for (let b = 0; b < DIM_F; b++) {
            for (let t = 0; t < chunkFrames; t++) {
                const sf = f0 + t;
                if (sf >= numFrames || b >= nBins) continue;
                dst[sf * nBins + b] = outData[c * DIM_F * DIM_T + b * DIM_T + t];
            }
        }
    }
}

async function separate(left, right) {
    const window = hannWindow(WIN);
    const stftL = stft(left, N_FFT, HOP, window);
    const stftR = stft(right, N_FFT, HOP, window);
    const { numFrames, nBins } = stftL;

    const vRealL = new Float32Array(numFrames * nBins);
    const vImagL = new Float32Array(numFrames * nBins);
    const vRealR = new Float32Array(numFrames * nBins);
    const vImagR = new Float32Array(numFrames * nBins);

    const numChunks = Math.ceil(numFrames / DIM_T);
    self.postMessage({
        status: "status",
        data: `MDX inference: ${numFrames} frames in ${numChunks} chunks...`
    });

    for (let c = 0; c < numChunks; c++) {
        const f0 = c * DIM_T;
        const chunkFrames = Math.min(DIM_T, numFrames - f0);
        const inputData = packChunk(stftL.real, stftL.imag, stftR.real, stftR.imag, f0, numFrames, nBins);
        const tensor = new ort.Tensor("float32", inputData, [1, DIM_C, DIM_F, DIM_T]);
        const feeds = {};
        feeds[session.inputNames[0]] = tensor;
        const results = await session.run(feeds);
        const out = results[session.outputNames[0]];
        unpackChunk(out.data, vRealL, vImagL, vRealR, vImagR, f0, numFrames, nBins, chunkFrames);
        try { if (tensor.dispose) tensor.dispose(); } catch (_) {}
        try { if (out.dispose) out.dispose(); } catch (_) {}
        self.postMessage({ status: "processing-progress", data: Math.round(((c + 1) / numChunks) * 85) });
        self.postMessage({ status: "status", data: `Chunk ${c + 1}/${numChunks}` });
    }

    self.postMessage({ status: "status", data: "Inverse STFT (vocals)..." });
    const vocalLeft = istft(vRealL, vImagL, N_FFT, HOP, window, left.length);
    const vocalRight = istft(vRealR, vImagR, N_FFT, HOP, window, right.length);

    const instLeft = new Float32Array(left.length);
    const instRight = new Float32Array(right.length);
    for (let i = 0; i < left.length; i++) {
        instLeft[i] = left[i] - vocalLeft[i];
        instRight[i] = right[i] - vocalRight[i];
    }
    return { vocalLeft, vocalRight, instLeft, instRight };
}

self.onmessage = async function (e) {
    const { action, data } = e.data;
    try {
        if (action === "load-model") {
            const { name, url } = data;
            self.postMessage({ status: "status", data: `checking cache for ${name}...` });
            let bytes = await getCache(name);
            const fallbacks = [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_9482.onnx",
                "https://huggingface.co/csukuangfj/sherpa-onnx-src-sep-models/resolve/main/UVR_MDXNET_9482.onnx"
            ];
            if (!bytes || (bytes.byteLength || bytes.length || 0) < 1e6) {
                self.postMessage({ status: "status", data: `Downloading ${name} (~29 MB vocal separation model)...` });
                bytes = await fetchAsArrayBuffer(url || fallbacks[0], (p) => {
                    self.postMessage({ status: "download-progress", data: { name, percent: p } });
                }, { fallbackUrls: fallbacks, timeoutMs: 300000, maxRetries: 4 });
                await setCache(name, bytes);
                self.postMessage({ status: "status", data: `${name} downloaded & cached.` });
            } else {
                self.postMessage({ status: "status", data: `${name} loaded from cache.` });
            }
            self.postMessage({ status: "status", data: "Initializing ONNX Session..." });
            if (bytes instanceof Uint8Array) bytes = bytes.buffer;
            try {
                session = await ort.InferenceSession.create(bytes, {
                    executionProviders: ["wasm"],
                    graphOptimizationLevel: "all"
                });
            } catch (err) {
                self.postMessage({ status: "status", data: "WASM failed, trying webgpu..." });
                session = await ort.InferenceSession.create(bytes, {
                    executionProviders: ["webgpu", "wasm"],
                    graphOptimizationLevel: "all"
                });
            }
            self.postMessage({ status: "status", data: `Model ${name} loaded successfully!` });
            self.postMessage({ status: "model-loaded" });

        } else if (action === "process-audio") {
            if (!session) throw new Error("Model not loaded");
            let { leftChannel, rightChannel, sampleRate } = data;

            self.postMessage({ status: "status", data: `Resampling ${sampleRate}Hz → ${SR}Hz...` });
            leftChannel = resample(leftChannel, sampleRate, SR);
            rightChannel = resample(rightChannel, sampleRate, SR);

            self.postMessage({ status: "status", data: "Running MDX vocal separation..." });
            const result = await separate(leftChannel, rightChannel);

            if (sampleRate !== SR) {
                self.postMessage({ status: "status", data: `Resampling output → ${sampleRate}Hz...` });
                result.vocalLeft = resample(result.vocalLeft, SR, sampleRate);
                result.vocalRight = resample(result.vocalRight, SR, sampleRate);
                result.instLeft = resample(result.instLeft, SR, sampleRate);
                result.instRight = resample(result.instRight, SR, sampleRate);
            }

            self.postMessage({ status: "processing-progress", data: 100 });
            self.postMessage({ status: "result", data: result });
        }
    } catch (err) {
        self.postMessage({
            status: "error",
            data: (err && (err.stack || err.message)) || String(err)
        });
    }
};
