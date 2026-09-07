// ONNX Runtime Web Worker – segment-based processing to avoid OOM
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js");

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

let session = null;

const DB_NAME = "UVR_Model_Cache_v4";
const STORE_NAME = "models";

function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };
        request.onsuccess = (e) => resolve(e.target.result);
        request.onerror = (e) => reject(e.target.error);
    });
}

function getCache(key) {
    return openDB().then((db) => new Promise((resolve, reject) => {
        const request = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(key);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    }));
}

function setCache(key, val) {
    return openDB().then((db) => new Promise((resolve, reject) => {
        const request = db.transaction(STORE_NAME, "readwrite").objectStore(STORE_NAME).put(val, key);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    }));
}

function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
}

async function fetchAsArrayBuffer(url, onProgress, options = {}) {
    const { maxRetries = 3, timeoutMs = 180000, fallbackUrls = [] } = options;
    const urlsToTry = [url, ...fallbackUrls];
    let lastError = null;

    for (const currentUrl of urlsToTry) {
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                if (attempt > 1) {
                    const delay = Math.min(1000 * Math.pow(2, attempt - 2), 8000);
                    self.postMessage({
                        status: "status",
                        data: `Retry ${attempt}/${maxRetries} in ${Math.round(delay / 1000)}s...`
                    });
                    await sleep(delay);
                }

                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
                const response = await fetch(currentUrl, { signal: controller.signal, cache: "no-cache" });
                clearTimeout(timeoutId);

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const contentLength = +response.headers.get("Content-Length") || 0;
                const reader = response.body.getReader();
                let receivedLength = 0;
                const chunks = [];

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    receivedLength += value.length;
                    if (onProgress) {
                        if (contentLength > 0) {
                            onProgress(Math.min(99, Math.round((receivedLength / contentLength) * 100)));
                        } else {
                            onProgress(Math.min(99, Math.round(receivedLength / (1024 * 1024))));
                        }
                    }
                }

                if (contentLength > 0 && receivedLength < contentLength * 0.9) {
                    throw new Error(`Incomplete download: got ${receivedLength} of ${contentLength}`);
                }
                if (receivedLength === 0) throw new Error("Downloaded empty file");
                if (receivedLength < 1000000) {
                    throw new Error(`Model too small (${receivedLength} bytes). Need ~18 MB UVR-DeNoise-Lite-single.onnx`);
                }

                const result = new Uint8Array(receivedLength);
                let pos = 0;
                for (const c of chunks) { result.set(c, pos); pos += c.length; }
                if (onProgress) onProgress(100);
                return result.buffer;
            } catch (err) {
                lastError = err;
                const msg = err.name === "AbortError" ? "Timeout" : (err.message || String(err));
                self.postMessage({ status: "status", data: `Download attempt failed: ${msg}` });
                if (msg.includes("HTTP 404") || msg.includes("HTTP 403")) break;
            }
        }
    }
    throw new Error(`Failed to download model: ${lastError && lastError.message}`);
}

function bitReverse(i, n) {
    let rev = 0, temp = n >> 1;
    while (temp > 0) {
        rev = (rev << 1) | (i & 1);
        i >>= 1;
        temp >>= 1;
    }
    return rev;
}

function fft(re, im) {
    const n = re.length;
    for (let i = 0; i < n; i++) {
        let j = bitReverse(i, n);
        if (i < j) {
            let t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const angle = -2 * Math.PI / len;
        const wlen_re = Math.cos(angle), wlen_im = Math.sin(angle);
        for (let i = 0; i < n; i += len) {
            let w_re = 1.0, w_im = 0.0;
            for (let j = 0; j < len / 2; j++) {
                const u_re = re[i + j], u_im = im[i + j];
                const idx2 = i + j + len / 2;
                const v_re = re[idx2] * w_re - im[idx2] * w_im;
                const v_im = re[idx2] * w_im + im[idx2] * w_re;
                re[i + j] = u_re + v_re;
                im[i + j] = u_im + v_im;
                re[idx2] = u_re - v_re;
                im[idx2] = u_im - v_im;
                const nw_re = w_re * wlen_re - w_im * wlen_im;
                const nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re; w_im = nw_im;
            }
        }
    }
}

function padReflect(x, pad) {
    const padded = new Float32Array(x.length + 2 * pad);
    padded.set(x, pad);
    for (let i = 0; i < pad; i++) padded[pad - 1 - i] = x[Math.min(i + 1, x.length - 1)];
    for (let i = 0; i < pad; i++) padded[padded.length - pad + i] = x[Math.max(0, x.length - 2 - i)];
    return padded;
}

function runSTFT(signal, nfft, hopLength) {
    const pad = nfft / 2;
    const padded = padReflect(signal, pad);
    const numFrames = Math.floor((padded.length - nfft) / hopLength) + 1;
    const numBins = nfft / 2 + 1;
    const window = new Float32Array(nfft);
    for (let i = 0; i < nfft; i++) window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nfft - 1)));

    const magnitudes = new Float32Array(numBins * numFrames);
    const phases = new Float32Array(numBins * numFrames);
    const re = new Float32Array(nfft);
    const im = new Float32Array(nfft);

    for (let t = 0; t < numFrames; t++) {
        const offset = t * hopLength;
        for (let i = 0; i < nfft; i++) {
            re[i] = padded[offset + i] * window[i];
            im[i] = 0;
        }
        fft(re, im);
        for (let b = 0; b < numBins; b++) {
            const r = re[b], m = im[b];
            magnitudes[b * numFrames + t] = Math.sqrt(r * r + m * m);
            phases[b * numFrames + t] = Math.atan2(m, r);
        }
    }
    return { magnitudes, phases, numFrames };
}

function runISTFT(magnitudes, phases, nfft, hopLength, originalLength) {
    const numBins = nfft / 2 + 1;
    const numFrames = magnitudes.length / numBins;
    const pad = nfft / 2;
    const paddedLength = (numFrames - 1) * hopLength + nfft;
    const accum = new Float32Array(paddedLength);
    const windowSquaredSum = new Float32Array(paddedLength);
    const window = new Float32Array(nfft);
    for (let i = 0; i < nfft; i++) window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nfft - 1)));
    const re = new Float32Array(nfft);
    const im = new Float32Array(nfft);

    for (let t = 0; t < numFrames; t++) {
        for (let b = 0; b < numBins; b++) {
            const mag = magnitudes[b * numFrames + t];
            const phase = phases[b * numFrames + t];
            re[b] = mag * Math.cos(phase);
            im[b] = mag * Math.sin(phase);
        }
        for (let b = 1; b < nfft / 2; b++) {
            re[nfft - b] = re[b];
            im[nfft - b] = -im[b];
        }
        im[0] = 0;
        im[nfft / 2] = 0;
        fft(re, im);
        for (let i = 0; i < nfft; i++) re[i] /= nfft;
        const offset = t * hopLength;
        for (let i = 0; i < nfft; i++) {
            accum[offset + i] += re[i] * window[i];
            windowSquaredSum[offset + i] += window[i] * window[i];
        }
    }

    const output = new Float32Array(originalLength);
    for (let i = 0; i < originalLength; i++) {
        const val = accum[pad + i] || 0;
        const denom = windowSquaredSum[pad + i] || 0;
        output[i] = denom > 1e-4 ? val / denom : val;
    }
    return output;
}

async function inferChunk(stftLeft, stftRight, f0, chunkLen, numFrames) {
    const paddedLen = Math.max(16, Math.ceil(chunkLen / 16) * 16);
    const inputData = new Float32Array(2 * 1024 * paddedLen);

    for (let b = 0; b < 1024; b++) {
        const dst = b * paddedLen;
        const src = b * numFrames + f0;
        for (let f = 0; f < paddedLen; f++) {
            inputData[dst + f] = stftLeft.magnitudes[src + Math.min(f, chunkLen - 1)] || 0;
        }
    }
    for (let b = 0; b < 1024; b++) {
        const dst = 1024 * paddedLen + b * paddedLen;
        const src = b * numFrames + f0;
        for (let f = 0; f < paddedLen; f++) {
            inputData[dst + f] = stftRight.magnitudes[src + Math.min(f, chunkLen - 1)] || 0;
        }
    }

    const inputTensor = new ort.Tensor("float32", inputData, [1, 2, 1024, paddedLen]);
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    const results = await session.run(feeds);
    const out = results[session.outputNames[0]];
    const mask = out.data;
    const outBins = out.dims[2];
    const outFrames = out.dims[3];

    const maskL = new Float32Array(1025 * chunkLen);
    const maskR = new Float32Array(1025 * chunkLen);
    maskL.fill(1);
    maskR.fill(1);

    for (let b = 0; b < Math.min(1025, outBins); b++) {
        for (let f = 0; f < chunkLen && f < outFrames; f++) {
            maskL[b * chunkLen + f] = mask[0 * (outBins * outFrames) + b * outFrames + f];
            maskR[b * chunkLen + f] = mask[1 * (outBins * outFrames) + b * outFrames + f];
        }
    }

    try { if (inputTensor.dispose) inputTensor.dispose(); } catch (_) {}
    try { if (out.dispose) out.dispose(); } catch (_) {}

    return { maskL, maskR };
}

async function processSegment(leftSeg, rightSeg) {
    const nfft = 2048;
    const hopLength = 1024;
    const stftLeft = runSTFT(leftSeg, nfft, hopLength);
    const stftRight = runSTFT(rightSeg, nfft, hopLength);
    const numFrames = stftLeft.numFrames;
    const CHUNK = 128;

    const vocalMagL = new Float32Array(1025 * numFrames);
    const vocalMagR = new Float32Array(1025 * numFrames);
    const instMagL = new Float32Array(1025 * numFrames);
    const instMagR = new Float32Array(1025 * numFrames);

    for (let f0 = 0; f0 < numFrames; f0 += CHUNK) {
        const chunkLen = Math.min(CHUNK, numFrames - f0);
        const { maskL, maskR } = await inferChunk(stftLeft, stftRight, f0, chunkLen, numFrames);

        for (let b = 0; b < 1025; b++) {
            for (let f = 0; f < chunkLen; f++) {
                const gi = b * numFrames + f0 + f;
                const li = b * chunkLen + f;
                const oL = stftLeft.magnitudes[gi] || 0;
                const oR = stftRight.magnitudes[gi] || 0;
                const mL = maskL[li];
                const mR = maskR[li];
                vocalMagL[gi] = mL * oL;
                vocalMagR[gi] = mR * oR;
                instMagL[gi] = (1 - mL) * oL;
                instMagR[gi] = (1 - mR) * oR;
            }
        }
    }

    const len = leftSeg.length;
    return {
        vocalLeft: runISTFT(vocalMagL, stftLeft.phases, nfft, hopLength, len),
        vocalRight: runISTFT(vocalMagR, stftRight.phases, nfft, hopLength, len),
        instLeft: runISTFT(instMagL, stftLeft.phases, nfft, hopLength, len),
        instRight: runISTFT(instMagR, stftRight.phases, nfft, hopLength, len)
    };
}

self.onmessage = async function (e) {
    const { action, data } = e.data;

    try {
        if (action === "load-model") {
            const { name, url, dataUrl } = data;
            self.postMessage({ status: "status", data: `checking cache for ${name}...` });

            let onnxBytes = await getCache(name);
            const onnxFallbacks = [
                "https://baobabprince.github.io/ultimatevocalremovergui/converted_models/UVR-DeNoise-Lite-single.onnx",
                "https://raw.githubusercontent.com/baobabprince/ultimatevocalremovergui/master/converted_models/UVR-DeNoise-Lite-single.onnx"
            ];

            if (!onnxBytes || (onnxBytes.byteLength || onnxBytes.length || 0) < 1000000) {
                self.postMessage({ status: "status", data: `Downloading ${name} (~18 MB)...` });
                onnxBytes = await fetchAsArrayBuffer(url, (percent) => {
                    self.postMessage({ status: "download-progress", data: { name, percent } });
                }, { fallbackUrls: onnxFallbacks, timeoutMs: 180000, maxRetries: 4 });
                await setCache(name, onnxBytes);
                self.postMessage({ status: "status", data: `${name} downloaded & cached.` });
            } else {
                self.postMessage({ status: "status", data: `${name} loaded from cache.` });
            }

            self.postMessage({ status: "status", data: "Initializing ONNX Session (WASM)..." });
            if (onnxBytes instanceof Uint8Array) onnxBytes = onnxBytes.buffer;

            const sessionOptions = {
                executionProviders: ["wasm"],
                graphOptimizationLevel: "all"
            };

            try {
                session = await ort.InferenceSession.create(onnxBytes, sessionOptions);
            } catch (firstErr) {
                self.postMessage({ status: "status", data: `WASM failed (${firstErr.message}), trying webgpu...` });
                sessionOptions.executionProviders = ["webgpu", "wasm"];
                session = await ort.InferenceSession.create(onnxBytes, sessionOptions);
            }

            self.postMessage({ status: "status", data: `Model ${name} loaded successfully!` });
            self.postMessage({ status: "model-loaded" });

        } else if (action === "process-audio") {
            if (!session) throw new Error("Model is not loaded yet");

            const { leftChannel, rightChannel } = data;
            const totalLen = leftChannel.length;

            const SEG_SAMPLES = 6 * 48000;
            const OVERLAP = 2048;

            const vocalLeft = new Float32Array(totalLen);
            const vocalRight = new Float32Array(totalLen);
            const instLeft = new Float32Array(totalLen);
            const instRight = new Float32Array(totalLen);
            const weight = new Float32Array(totalLen);

            const numSegs = Math.ceil(totalLen / SEG_SAMPLES);
            self.postMessage({
                status: "status",
                data: `Processing ${numSegs} time segments (~6s each) to save memory...`
            });

            for (let s = 0; s < numSegs; s++) {
                const start = s * SEG_SAMPLES;
                const end = Math.min(totalLen, start + SEG_SAMPLES + OVERLAP);
                const leftSeg = leftChannel.subarray(start, end);
                const rightSeg = rightChannel.subarray(start, end);

                self.postMessage({
                    status: "status",
                    data: `Segment ${s + 1}/${numSegs} (${((end - start) / 48000).toFixed(1)}s)...`
                });

                const result = await processSegment(leftSeg, rightSeg);
                const segLen = result.vocalLeft.length;

                for (let i = 0; i < segLen; i++) {
                    const gi = start + i;
                    if (gi >= totalLen) break;
                    let w = 1.0;
                    if (s > 0 && i < OVERLAP) w = i / OVERLAP;
                    if (s < numSegs - 1 && i >= segLen - OVERLAP) w = (segLen - 1 - i) / OVERLAP;
                    if (w < 0) w = 0;

                    vocalLeft[gi] += result.vocalLeft[i] * w;
                    vocalRight[gi] += result.vocalRight[i] * w;
                    instLeft[gi] += result.instLeft[i] * w;
                    instRight[gi] += result.instRight[i] * w;
                    weight[gi] += w;
                }

                const pct = Math.round(((s + 1) / numSegs) * 90);
                self.postMessage({ status: "processing-progress", data: pct });
            }

            for (let i = 0; i < totalLen; i++) {
                const w = weight[i] || 1;
                vocalLeft[i] /= w;
                vocalRight[i] /= w;
                instLeft[i] /= w;
                instRight[i] /= w;
            }

            self.postMessage({ status: "processing-progress", data: 100 });
            self.postMessage({
                status: "result",
                data: { vocalLeft, vocalRight, instLeft, instRight }
            });
        }
    } catch (err) {
        self.postMessage({
            status: "error",
            data: (err && (err.stack || err.message)) || String(err)
        });
    }
};
