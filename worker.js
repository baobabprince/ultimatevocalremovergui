// ONNX Runtime Web Worker for UVR5 Browser Edition
// cache v4 + chunked inference to avoid OOM on long files

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
    return openDB().then((db) => {
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(STORE_NAME, "readonly");
            const store = transaction.objectStore(STORE_NAME);
            const request = store.get(key);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    });
}

function setCache(key, val) {
    return openDB().then((db) => {
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(STORE_NAME, "readwrite");
            const store = transaction.objectStore(STORE_NAME);
            const request = store.put(val, key);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    });
}

function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
}

async function fetchAsArrayBuffer(url, onProgress, options = {}) {
    const {
        maxRetries = 3,
        timeoutMs = 180000,
        fallbackUrls = []
    } = options;

    const urlsToTry = [url, ...fallbackUrls];
    let lastError = null;

    for (let urlIndex = 0; urlIndex < urlsToTry.length; urlIndex++) {
        const currentUrl = urlsToTry[urlIndex];

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                if (attempt > 1) {
                    const delay = Math.min(1000 * Math.pow(2, attempt - 2), 8000);
                    self.postMessage({
                        status: "status",
                        data: `Retry ${attempt}/${maxRetries} in ${Math.round(delay / 1000)}s... (${currentUrl.split("/").pop()})`
                    });
                    await sleep(delay);
                }

                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

                const response = await fetch(currentUrl, {
                    signal: controller.signal,
                    cache: "no-cache"
                });
                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status} for ${currentUrl}`);
                }

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
                    throw new Error(
                        `Incomplete download: got ${receivedLength} of ${contentLength} bytes`
                    );
                }

                if (receivedLength === 0) {
                    throw new Error("Downloaded empty file");
                }

                if (contentLength > 0 && receivedLength !== contentLength) {
                    self.postMessage({
                        status: "status",
                        data: `Note: size mismatch (got ${receivedLength}, header ${contentLength}) – accepting completed download`
                    });
                }

                const result = new Uint8Array(receivedLength);
                let position = 0;
                for (const chunk of chunks) {
                    result.set(chunk, position);
                    position += chunk.length;
                }
                if (onProgress) onProgress(100);

                if (receivedLength < 1000000) {
                    throw new Error(
                        `Model file is too small (${receivedLength} bytes). ` +
                        `Expected UVR-DeNoise-Lite-single.onnx (~18 MB) in converted_models/.`
                    );
                }
                return result.buffer;
            } catch (err) {
                lastError = err;
                const msg = err.name === "AbortError"
                    ? `Timeout after ${timeoutMs / 1000}s`
                    : (err.message || String(err));

                self.postMessage({
                    status: "status",
                    data: `Download attempt failed: ${msg}`
                });

                if (msg.includes("HTTP 404") || msg.includes("HTTP 403")) {
                    break;
                }
            }
        }
    }

    throw new Error(
        `Failed to download model after retries. Last error: ${lastError && (lastError.message || lastError)}. ` +
        `Check your network connection and try again.`
    );
}

function bitReverse(i, n) {
    let rev = 0;
    let temp = n >> 1;
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
            let temp = re[i]; re[i] = re[j]; re[j] = temp;
            temp = im[i]; im[i] = im[j]; im[j] = temp;
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        let angle = -2 * Math.PI / len;
        let wlen_re = Math.cos(angle);
        let wlen_im = Math.sin(angle);
        for (let i = 0; i < n; i += len) {
            let w_re = 1.0;
            let w_im = 0.0;
            for (let j = 0; j < len / 2; j++) {
                let u_re = re[i + j];
                let u_im = im[i + j];
                let idx2 = i + j + len / 2;
                let v_re = re[idx2] * w_re - im[idx2] * w_im;
                let v_im = re[idx2] * w_im + im[idx2] * w_re;
                re[i + j] = u_re + v_re;
                im[i + j] = u_im + v_im;
                re[idx2] = u_re - v_re;
                im[idx2] = u_im - v_im;
                let next_w_re = w_re * wlen_re - w_im * wlen_im;
                let next_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
        }
    }
}

function padReflect(x, pad) {
    const padded = new Float32Array(x.length + 2 * pad);
    padded.set(x, pad);
    for (let i = 0; i < pad; i++) {
        padded[pad - 1 - i] = x[i + 1];
    }
    for (let i = 0; i < pad; i++) {
        padded[padded.length - pad + i] = x[x.length - 2 - i];
    }
    return padded;
}

function runSTFT(signal, nfft, hopLength) {
    const pad = nfft / 2;
    const padded = padReflect(signal, pad);
    const numFrames = Math.floor((padded.length - nfft) / hopLength) + 1;
    const numBins = nfft / 2 + 1;

    const window = new Float32Array(nfft);
    for (let i = 0; i < nfft; i++) {
        window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nfft - 1)));
    }

    const magnitudes = new Float32Array(numBins * numFrames);
    const phases = new Float32Array(numBins * numFrames);
    const re = new Float32Array(nfft);
    const im = new Float32Array(nfft);

    for (let t = 0; t < numFrames; t++) {
        const offset = t * hopLength;
        for (let i = 0; i < nfft; i++) {
            re[i] = padded[offset + i] * window[i];
            im[i] = 0.0;
        }
        fft(re, im);
        for (let b = 0; b < numBins; b++) {
            const r = re[b];
            const m = im[b];
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
    for (let i = 0; i < nfft; i++) {
        window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nfft - 1)));
    }

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
        im[0] = 0.0;
        im[nfft / 2] = 0.0;

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
        const val = accum[pad + i];
        const denom = windowSquaredSum[pad + i];
        output[i] = denom > 1e-4 ? val / denom : val;
    }
    return output;
}

self.onmessage = async function (e) {
    const { action, data } = e.data;

    try {
        if (action === "load-model") {
            const { name, url, dataUrl } = data;
            self.postMessage({ status: "status", data: `checking cache for ${name}...` });

            let onnxBytes = await getCache(name);
            let onnxDataBytes = dataUrl ? await getCache(name + ".data") : null;

            const onnxFallbacks = [
                `https://baobabprince.github.io/ultimatevocalremovergui/converted_models/UVR-DeNoise-Lite-single.onnx`,
                `https://raw.githubusercontent.com/baobabprince/ultimatevocalremovergui/master/converted_models/UVR-DeNoise-Lite-single.onnx`
            ];
            const dataFallbacks = dataUrl ? [
                `https://baobabprince.github.io/ultimatevocalremovergui/converted_models/UVR-DeNoise-Lite.onnx.data`,
                `https://raw.githubusercontent.com/baobabprince/ultimatevocalremovergui/master/converted_models/UVR-DeNoise-Lite.onnx.data`
            ] : [];

            if (!onnxBytes) {
                self.postMessage({ status: "status", data: `Downloading ${name} (~18 MB)...` });
                onnxBytes = await fetchAsArrayBuffer(
                    url,
                    (percent) => {
                        self.postMessage({
                            status: "download-progress",
                            data: { name, percent }
                        });
                    },
                    { fallbackUrls: onnxFallbacks, timeoutMs: 180000, maxRetries: 4 }
                );
                await setCache(name, onnxBytes);
                self.postMessage({ status: "status", data: `${name} downloaded & cached.` });
            } else {
                const cachedSize = onnxBytes.byteLength || (onnxBytes.length || 0);
                if (cachedSize < 1000000) {
                    self.postMessage({ status: "status", data: `Cached model too small (${cachedSize} bytes) – re-downloading...` });
                    onnxBytes = await fetchAsArrayBuffer(
                        url,
                        (percent) => {
                            self.postMessage({
                                status: "download-progress",
                                data: { name, percent }
                            });
                        },
                        { fallbackUrls: onnxFallbacks, timeoutMs: 180000, maxRetries: 4 }
                    );
                    await setCache(name, onnxBytes);
                } else {
                    self.postMessage({ status: "status", data: `${name} loaded from cache.` });
                }
            }

            if (dataUrl && !onnxDataBytes) {
                self.postMessage({ status: "status", data: `Downloading weights data for ${name}...` });
                onnxDataBytes = await fetchAsArrayBuffer(
                    dataUrl,
                    (percent) => {
                        self.postMessage({
                            status: "download-progress",
                            data: { name: name + " data", percent }
                        });
                    },
                    { fallbackUrls: dataFallbacks, timeoutMs: 180000, maxRetries: 4 }
                );
                await setCache(name + ".data", onnxDataBytes);
            }

            self.postMessage({ status: "status", data: `Initializing ONNX Session (WASM)...` });

            if (onnxBytes instanceof Uint8Array) onnxBytes = onnxBytes.buffer;
            if (onnxDataBytes instanceof Uint8Array) onnxDataBytes = onnxDataBytes.buffer;

            const sessionOptions = {
                executionProviders: ["wasm"],
                graphOptimizationLevel: "all"
            };

            if (onnxDataBytes) {
                sessionOptions.externalData = [
                    {
                        data: onnxDataBytes,
                        path: "UVR-DeNoise-Lite.onnx.data"
                    }
                ];
            }

            try {
                session = await ort.InferenceSession.create(onnxBytes, sessionOptions);
            } catch (firstErr) {
                self.postMessage({
                    status: "status",
                    data: `WASM failed (${firstErr.message}), trying webgpu...`
                });
                sessionOptions.executionProviders = ["webgpu", "wasm"];
                session = await ort.InferenceSession.create(onnxBytes, sessionOptions);
            }

            self.postMessage({ status: "status", data: `Model ${name} loaded successfully!` });
            self.postMessage({ status: "model-loaded" });

        } else if (action === "process-audio") {
            if (!session) throw new Error("Model is not loaded yet");

            const { leftChannel, rightChannel } = data;
            const originalLength = leftChannel.length;

            self.postMessage({ status: "status", data: "Running STFT analysis on stereo audio..." });
            const nfft = 2048;
            const hopLength = 1024;

            const stftLeft = runSTFT(leftChannel, nfft, hopLength);
            const stftRight = runSTFT(rightChannel, nfft, hopLength);
            const numFrames = stftLeft.numFrames;

            // Chunked inference – avoids WASM OOM on long tracks
            const CHUNK = 256;
            const maskLAll = new Float32Array(1025 * numFrames);
            const maskRAll = new Float32Array(1025 * numFrames);
            maskLAll.fill(1);
            maskRAll.fill(1);

            const numChunks = Math.ceil(numFrames / CHUNK);
            self.postMessage({
                status: "status",
                data: `Running Model Inference: ${numFrames} frames in ${numChunks} chunks of ${CHUNK}...`
            });

            for (let c = 0; c < numChunks; c++) {
                const f0 = c * CHUNK;
                const f1 = Math.min(numFrames, f0 + CHUNK);
                const chunkLen = f1 - f0;
                const paddedLen = Math.ceil(chunkLen / 16) * 16;

                const inputData = new Float32Array(1 * 2 * 1024 * paddedLen);
                for (let b = 0; b < 1024; b++) {
                    const offsetDst = b * paddedLen;
                    const offsetSrc = b * numFrames + f0;
                    for (let f = 0; f < paddedLen; f++) {
                        const srcF = Math.min(f, chunkLen - 1);
                        inputData[offsetDst + f] = stftLeft.magnitudes[offsetSrc + srcF];
                    }
                }
                for (let b = 0; b < 1024; b++) {
                    const offsetDst = 1024 * paddedLen + b * paddedLen;
                    const offsetSrc = b * numFrames + f0;
                    for (let f = 0; f < paddedLen; f++) {
                        const srcF = Math.min(f, chunkLen - 1);
                        inputData[offsetDst + f] = stftRight.magnitudes[offsetSrc + srcF];
                    }
                }

                const inputTensor = new ort.Tensor("float32", inputData, [1, 2, 1024, paddedLen]);
                const feeds = {};
                feeds[session.inputNames[0]] = inputTensor;

                const results = await session.run(feeds);
                const outputTensor = results[session.outputNames[0]];
                const outputMask = outputTensor.data;
                const outBins = outputTensor.dims[2];
                const outFrames = outputTensor.dims[3];

                for (let b = 0; b < Math.min(1025, outBins); b++) {
                    for (let f = 0; f < chunkLen; f++) {
                        if (f < outFrames) {
                            maskLAll[b * numFrames + f0 + f] =
                                outputMask[0 * (outBins * outFrames) + b * outFrames + f];
                            maskRAll[b * numFrames + f0 + f] =
                                outputMask[1 * (outBins * outFrames) + b * outFrames + f];
                        }
                    }
                }

                const pct = Math.round(10 + (50 * (c + 1)) / numChunks);
                self.postMessage({ status: "processing-progress", data: pct });
                self.postMessage({
                    status: "status",
                    data: `Chunk ${c + 1}/${numChunks} done`
                });
            }

            self.postMessage({ status: "status", data: "Reconstructing separated audio channels..." });

            const vocalMagLeft = new Float32Array(1025 * numFrames);
            const vocalMagRight = new Float32Array(1025 * numFrames);
            const instMagLeft = new Float32Array(1025 * numFrames);
            const instMagRight = new Float32Array(1025 * numFrames);

            for (let b = 0; b < 1025; b++) {
                const offsetSrc = b * numFrames;
                for (let f = 0; f < numFrames; f++) {
                    const mL = maskLAll[offsetSrc + f];
                    const mR = maskRAll[offsetSrc + f];
                    const originalL = stftLeft.magnitudes[offsetSrc + f] || 0;
                    const originalR = stftRight.magnitudes[offsetSrc + f] || 0;
                    vocalMagLeft[offsetSrc + f] = mL * originalL;
                    vocalMagRight[offsetSrc + f] = mR * originalR;
                    instMagLeft[offsetSrc + f] = (1.0 - mL) * originalL;
                    instMagRight[offsetSrc + f] = (1.0 - mR) * originalR;
                }
            }

            self.postMessage({ status: "processing-progress", data: 80 });

            const vocalLeft = runISTFT(vocalMagLeft, stftLeft.phases, nfft, hopLength, originalLength);
            const vocalRight = runISTFT(vocalMagRight, stftRight.phases, nfft, hopLength, originalLength);
            const instLeft = runISTFT(instMagLeft, stftLeft.phases, nfft, hopLength, originalLength);
            const instRight = runISTFT(instMagRight, stftRight.phases, nfft, hopLength, originalLength);

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
