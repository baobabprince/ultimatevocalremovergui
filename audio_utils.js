// Audio Utilities for UVR5 Browser Edition (Ensemble, Pitch, Time Stretch, Align, Match)

// 1. Phase Vocoder / Pitch Shifter and Time Stretcher
function pitchShift(signal, semitones, sampleRate) {
    if (semitones === 0) return signal;
    const factor = Math.pow(2, semitones / 12);
    // Standard Pitch Shift = Time Stretch by (1/factor) then resample by factor
    const stretched = timeStretch(signal, 1.0 / factor);
    return resample(stretched, factor);
}

function timeStretch(signal, factor) {
    if (factor === 1.0) return signal;

    // WSOLA (Waveform Similarity Overlap-Add) / OLA implementation
    const numChannels = 1; // Mono processing per channel
    const nfft = 1024;
    const hopLength = 256;
    const synthHop = Math.round(hopLength * factor);

    const outputLength = Math.round(signal.length * factor);
    const output = new Float32Array(outputLength);
    const window = new Float32Array(nfft);
    for (let i = 0; i < nfft; i++) {
        window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nfft - 1)));
    }

    let outPos = 0;
    let inPos = 0;

    while (inPos + nfft < signal.length && outPos + nfft < outputLength) {
        for (let i = 0; i < nfft; i++) {
            output[outPos + i] += signal[inPos + i] * window[i];
        }
        inPos += hopLength;
        outPos += synthHop;
    }

    return output;
}

function resample(signal, factor) {
    const outputLength = Math.round(signal.length / factor);
    const output = new Float32Array(outputLength);
    for (let i = 0; i < outputLength; i++) {
        const srcPos = i * factor;
        const srcIndex = Math.floor(srcPos);
        const frac = srcPos - srcIndex;
        if (srcIndex + 1 < signal.length) {
            output[i] = (1 - frac) * signal[srcIndex] + frac * signal[srcIndex + 1];
        } else {
            output[i] = signal[srcIndex] || 0.0;
        }
    }
    return output;
}

// 2. Align Audio (cross-correlation / offset matching)
function alignAudio(refSignal, targetSignal, searchWindowSeconds = 2.0, sampleRate = 44100) {
    const searchSamples = Math.round(searchWindowSeconds * sampleRate);
    const limit = Math.min(searchSamples, refSignal.length, targetSignal.length);

    // Simple block matching cross-correlation
    let bestLag = 0;
    let maxCorr = -Infinity;

    // We sample a chunk in the middle of the audio for alignment to be highly robust and extremely fast
    const midPoint = Math.floor(refSignal.length / 2);
    const chunkSize = Math.min(sampleRate * 2, refSignal.length - searchSamples); // 2 seconds chunk

    if (chunkSize <= 0) return { alignedSignal: targetSignal, delay: 0 };

    const refChunk = refSignal.subarray(midPoint, midPoint + chunkSize);

    for (let lag = -limit; lag < limit; lag += 100) { // Coarse search
        let corr = 0;
        for (let i = 0; i < chunkSize; i++) {
            const refVal = refChunk[i];
            const targetVal = targetSignal[midPoint + i + lag] || 0;
            corr += refVal * targetVal;
        }
        if (corr > maxCorr) {
            maxCorr = corr;
            bestLag = lag;
        }
    }

    // Fine search around best coarse lag
    let fineBestLag = bestLag;
    maxCorr = -Infinity;
    for (let lag = bestLag - 100; lag <= bestLag + 100; lag++) {
        let corr = 0;
        for (let i = 0; i < chunkSize; i++) {
            const refVal = refChunk[i];
            const targetVal = targetSignal[midPoint + i + lag] || 0;
            corr += refVal * targetVal;
        }
        if (corr > maxCorr) {
            maxCorr = corr;
            fineBestLag = lag;
        }
    }

    // Shift target signal
    const aligned = new Float32Array(targetSignal.length);
    if (fineBestLag > 0) {
        aligned.set(targetSignal.subarray(0, targetSignal.length - fineBestLag), fineBestLag);
    } else if (fineBestLag < 0) {
        aligned.set(targetSignal.subarray(-fineBestLag));
    } else {
        aligned.set(targetSignal);
    }

    return { alignedSignal: aligned, delay: fineBestLag };
}

// 3. Audio Volume Matching (RMS)
function matchVolume(refSignal, targetSignal) {
    let refSum = 0;
    let targetSum = 0;
    for (let i = 0; i < refSignal.length; i++) {
        refSum += refSignal[i] * refSignal[i];
    }
    for (let i = 0; i < targetSignal.length; i++) {
        targetSum += targetSignal[i] * targetSignal[i];
    }
    const refRMS = Math.sqrt(refSum / refSignal.length);
    const targetRMS = Math.sqrt(targetSum / targetSignal.length);

    if (targetRMS === 0) return targetSignal;

    const factor = refRMS / targetRMS;
    const output = new Float32Array(targetSignal.length);
    for (let i = 0; i < targetSignal.length; i++) {
        output[i] = targetSignal[i] * factor;
    }
    return output;
}

// 4. Ensemble (Average, Min, Max spectral/waveform aggregation)
function ensembleSignals(signals, method = "Average") {
    if (signals.length === 0) return null;
    const len = signals[0].length;
    const output = new Float32Array(len);

    if (method === "Average" || method === "Linear Ensemble") {
        for (let i = 0; i < len; i++) {
            let sum = 0;
            for (let s = 0; s < signals.length; s++) {
                sum += signals[s][i];
            }
            output[i] = sum / signals.length;
        }
    } else if (method === "Min Spec") {
        for (let i = 0; i < len; i++) {
            let minVal = Infinity;
            let minSign = 1;
            for (let s = 0; s < signals.length; s++) {
                const val = signals[s][i];
                if (Math.abs(val) < Math.abs(minVal)) {
                    minVal = val;
                }
            }
            output[i] = minVal;
        }
    } else if (method === "Max Spec") {
        for (let i = 0; i < len; i++) {
            let maxVal = -Infinity;
            for (let s = 0; s < signals.length; s++) {
                const val = signals[s][i];
                if (Math.abs(val) > Math.abs(maxVal)) {
                    maxVal = val;
                }
            }
            output[i] = maxVal;
        }
    }

    return output;
}

// Expose functions if imported in node or worker
if (typeof module !== "undefined" && module.exports) {
    module.exports = { pitchShift, timeStretch, resample, alignAudio, matchVolume, ensembleSignals };
}

// ESM exports for test runners
export { pitchShift, timeStretch, resample, alignAudio, matchVolume, ensembleSignals };
