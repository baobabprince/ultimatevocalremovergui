// ONNX Runtime Web Worker for UVR5 Browser Edition

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js");

// Set wasm paths
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

let session = null;

// IndexedDB Caching for Model files
const DB_NAME = "UVR_Model_Cache";
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
            request.onerror = () => reject(request.onerror);
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
            request.onerror = () => reject(request.onerror);
        });
    });
}

// FFT helper functions
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

// Reflection padding helper
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

// STFT and iSTFT engine
function runSTFT(signal, nfft, hopLength) {
    const pad = nfft / 2;
    const padded = padReflect(signal, pad);
    const numFrames = Math.floor((padded.length - nfft) / hopLength) + 1;
    const numBins = nfft / 2 + 1;

    // Hanning Window
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
            const mag = Math.sqrt(r * r + m * m);
            const phase = Math.atan2(m, r);
            magnitudes[b * numFrames + t] = mag;
            phases[b * numFrames + t] = phase;
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

        // Conjugate symmetry
        for (let b = 1; b < nfft / 2; b++) {
            re[nfft - b] = re[b];
            im[nfft - b] = -im[b];
        }
        im[0] = 0.0;
        im[nfft / 2] = 0.0;

        fft(re, im); // Using FFT instead of ifft to avoid extra inverse imaginary flip
        for (let i = 0; i < nfft; i++) {
            re[i] /= nfft;
        }

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

// Listen for messages from main thread
self.onmessage = async function (e) {
    const { action, data } = e.data;

    try {
        if (action === "load-model") {
            const { name, url, dataUrl } = data;
            self.postMessage({ status: "status", data: `checking cache for ${name}...` });

            let onnxBytes = await getCache(name);
            let onnxDataBytes = null;
            if (dataUrl) {
                onnxDataBytes = await getCache(name + ".data");
            }

            if (!onnxBytes) {
                self.postMessage({ status: "status", data: `Downloading ${name} (may take a few minutes)...` });

                // Fetch with progress
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`Failed to download model file (HTTP ${response.status}) - make sure the model is built and placed in the converted_models/ directory`);
                }
                const reader = response.body.getReader();
                const contentLength = +response.headers.get('Content-Length') || 10000000;
                let receivedLength = 0;
                let chunks = [];
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    receivedLength += value.length;
                    self.postMessage({
                        status: "download-progress",
                        data: { name, percent: Math.round((receivedLength / contentLength) * 100) }
                    });
                }
                const positionBytes = new Uint8Array(receivedLength);
                let position = 0;
                for (let chunk of chunks) {
                    positionBytes.set(chunk, position);
                    position += chunk.length;
                }
                onnxBytes = positionBytes.buffer;
                await setCache(name, onnxBytes);
            }

            if (dataUrl && !onnxDataBytes) {
                self.postMessage({ status: "status", data: `Downloading weights data for ${name}...` });
                const response = await fetch(dataUrl);
                if (!response.ok) {
                    throw new Error(`Failed to download model weights data (HTTP ${response.status}) - make sure the .data file exists under converted_models/`);
                }
                const reader = response.body.getReader();
                const contentLength = +response.headers.get('Content-Length') || 18000000;
                let receivedLength = 0;
                let chunks = [];
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    receivedLength += value.length;
                    self.postMessage({
                        status: "download-progress",
                        data: { name: name + " data", percent: Math.round((receivedLength / contentLength) * 100) }
                    });
                }
                const positionBytes = new Uint8Array(receivedLength);
                let position = 0;
                for (let chunk of chunks) {
                    positionBytes.set(chunk, position);
                    position += chunk.length;
                }
                onnxDataBytes = positionBytes.buffer;
                await setCache(name + ".data", onnxDataBytes);
            }

            self.postMessage({ status: "status", data: `Initializing ONNX Session (WebGPU / WASM)...` });

            // Initialize ONNX Session
            const options = {
                executionProviders: ["webgpu", "wasm"],
            };
            if (onnxDataBytes) {
                options.externalData = [
                    {
                        data: new Uint8Array(onnxDataBytes),
                        path: "UVR-DeNoise-Lite.onnx.data"
                    }
                ];
            }

            session = await ort.InferenceSession.create(new Uint8Array(onnxBytes), options);
            self.postMessage({ status: "status", data: `Model ${name} loaded successfully!` });
            self.postMessage({ status: "model-loaded" });

        } else if (action === "process-audio") {
            if (!session) {
                throw new Error("Model is not loaded yet");
            }
            const { leftChannel, rightChannel, sampleRate } = data;
            const originalLength = leftChannel.length;

            self.postMessage({ status: "status", data: "Running STFT analysis on stereo audio..." });
            const nfft = 2048;
            const hopLength = 1024;

            const stftLeft = runSTFT(leftChannel, nfft, hopLength);
            const stftRight = runSTFT(rightChannel, nfft, hopLength);
            const numFrames = stftLeft.numFrames;

            // Pad the frame count to a multiple of 16 to prevent dimension mismatch in the U-Net skip connections
            const paddedNumFrames = Math.ceil(numFrames / 16) * 16;

            self.postMessage({ status: "status", data: `Running Model Inference over ${numFrames} frames (padded to ${paddedNumFrames})...` });

            // Prepare Input Tensor: shape [1, 2, 1024, paddedNumFrames]
            const inputTensorSize = 1 * 2 * 1024 * paddedNumFrames;
            const inputData = new Float32Array(inputTensorSize);

            // Left Channel magnitudes for bins 0..1023
            for (let b = 0; b < 1024; b++) {
                const offsetDst = 0 * (1024 * paddedNumFrames) + b * paddedNumFrames;
                const offsetSrc = b * numFrames;
                for (let f = 0; f < paddedNumFrames; f++) {
                    const srcFrame = Math.min(f, numFrames - 1);
                    inputData[offsetDst + f] = stftLeft.magnitudes[offsetSrc + srcFrame];
                }
            }

            // Right Channel magnitudes for bins 0..1023
            for (let b = 0; b < 1024; b++) {
                const offsetDst = 1 * (1024 * paddedNumFrames) + b * paddedNumFrames;
                const offsetSrc = b * numFrames;
                for (let f = 0; f < paddedNumFrames; f++) {
                    const srcFrame = Math.min(f, numFrames - 1);
                    inputData[offsetDst + f] = stftRight.magnitudes[offsetSrc + srcFrame];
                }
            }

            const inputTensor = new ort.Tensor("float32", inputData, [1, 2, 1024, paddedNumFrames]);

            const feeds = {};
            feeds[session.inputNames[0]] = inputTensor;

            self.postMessage({ status: "processing-progress", data: 10 });
            const results = await session.run(feeds);
            self.postMessage({ status: "processing-progress", data: 60 });

            const outputTensor = results[session.outputNames[0]];
            const outputMask = outputTensor.data; // Float32Array

            self.postMessage({ status: "status", data: "Reconstructing separated audio channels..." });

            // Dynamic dimensions of output mask to avoid indexing mismatch bugs
            const outputDims = outputTensor.dims; // [1, 2, height, width]
            const outBins = outputDims[2]; // e.g. 1025 or 1024
            const outFrames = outputDims[3]; // paddedNumFrames

            // We need 1025 bins for iSTFT
            const vocalMagLeft = new Float32Array(1025 * numFrames);
            const vocalMagRight = new Float32Array(1025 * numFrames);
            const instMagLeft = new Float32Array(1025 * numFrames);
            const instMagRight = new Float32Array(1025 * numFrames);

            for (let b = 0; b < 1025; b++) {
                const offsetSrc = b * numFrames;

                for (let f = 0; f < numFrames; f++) {
                    // Fetch mask using precise dynamic dimensions with secure fallback
                    let maskL = 1.0;
                    let maskR = 1.0;

                    if (b < outBins && f < outFrames) {
                        const offsetLeft = 0 * (outBins * outFrames) + b * outFrames + f;
                        const offsetRight = 1 * (outBins * outFrames) + b * outFrames + f;
                        maskL = outputMask[offsetLeft];
                        maskR = outputMask[offsetRight];
                    }

                    const originalL = stftLeft.magnitudes[offsetSrc + f];
                    const originalR = stftRight.magnitudes[offsetSrc + f];

                    vocalMagLeft[offsetSrc + f] = maskL * originalL;
                    vocalMagRight[offsetSrc + f] = maskR * originalR;

                    instMagLeft[offsetSrc + f] = (1.0 - maskL) * originalL;
                    instMagRight[offsetSrc + f] = (1.0 - maskR) * originalR;
                }
            }

            self.postMessage({ status: "processing-progress", data: 80 });

            // Run iSTFT
            const vocalLeft = runISTFT(vocalMagLeft, stftLeft.phases, nfft, hopLength, originalLength);
            const vocalRight = runISTFT(vocalMagRight, stftRight.phases, nfft, hopLength, originalLength);

            const instLeft = runISTFT(instMagLeft, stftLeft.phases, nfft, hopLength, originalLength);
            const instRight = runISTFT(instMagRight, stftRight.phases, nfft, hopLength, originalLength);

            self.postMessage({ status: "processing-progress", data: 100 });
            self.postMessage({
                status: "result",
                data: {
                    vocalLeft,
                    vocalRight,
                    instLeft,
                    instRight
                }
            });
        }
    } catch (err) {
        self.postMessage({ status: "error", data: err.stack || err.message });
    }
};
