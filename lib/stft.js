/**
 * Pure STFT / iSTFT helpers extracted for testing.
 * These mirror the implementation in worker.js
 */

export function bitReverse(i, n) {
    let rev = 0;
    let temp = n >> 1;
    while (temp > 0) {
        rev = (rev << 1) | (i & 1);
        i >>= 1;
        temp >>= 1;
    }
    return rev;
}

export function fft(re, im) {
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

export function padReflect(x, pad) {
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

export function runSTFT(signal, nfft, hopLength) {
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
            const mag = Math.sqrt(r * r + m * m);
            const phase = Math.atan2(m, r);
            magnitudes[b * numFrames + t] = mag;
            phases[b * numFrames + t] = phase;
        }
    }

    return { magnitudes, phases, numFrames };
}

export function runISTFT(magnitudes, phases, nfft, hopLength, originalLength) {
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
