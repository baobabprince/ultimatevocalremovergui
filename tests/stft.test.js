import { describe, it, expect } from 'vitest';
import { bitReverse, fft, padReflect, runSTFT, runISTFT } from '../lib/stft.js';

describe('bitReverse', () => {
  it('reverses bits correctly for power-of-two sizes', () => {
    expect(bitReverse(0, 8)).toBe(0);
    expect(bitReverse(1, 8)).toBe(4);
    expect(bitReverse(2, 8)).toBe(2);
    expect(bitReverse(3, 8)).toBe(6);
    expect(bitReverse(4, 8)).toBe(1);
  });
});

describe('fft', () => {
  it('computes identity on DC signal', () => {
    const n = 8;
    const re = new Float32Array(n).fill(1);
    const im = new Float32Array(n).fill(0);
    fft(re, im);
    expect(re[0]).toBeCloseTo(n, 5);
    for (let i = 1; i < n; i++) {
      expect(Math.abs(re[i])).toBeLessThan(1e-5);
      expect(Math.abs(im[i])).toBeLessThan(1e-5);
    }
  });

  it('handles a pure sine wave', () => {
    const n = 16;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = Math.sin(2 * Math.PI * 2 * i / n);
    }
    fft(re, im);
    const mag2 = Math.sqrt(re[2] * re[2] + im[2] * im[2]);
    const mag14 = Math.sqrt(re[14] * re[14] + im[14] * im[14]);
    expect(mag2).toBeGreaterThan(n / 4);
    expect(mag14).toBeGreaterThan(n / 4);
  });
});

describe('padReflect', () => {
  it('pads correctly with reflection', () => {
    const x = new Float32Array([1, 2, 3, 4, 5]);
    const padded = padReflect(x, 2);
    expect(Array.from(padded)).toEqual([3, 2, 1, 2, 3, 4, 5, 4, 3]);
  });

  it('handles pad=0', () => {
    const x = new Float32Array([1, 2, 3]);
    const padded = padReflect(x, 0);
    expect(Array.from(padded)).toEqual([1, 2, 3]);
  });
});

describe('STFT / iSTFT round-trip', () => {
  it('reconstructs a simple signal with low error', () => {
    const sampleRate = 44100;
    const duration = 0.1;
    const length = Math.floor(sampleRate * duration);
    const signal = new Float32Array(length);

    for (let i = 0; i < length; i++) {
      signal[i] = 0.5 * Math.sin(2 * Math.PI * 440 * i / sampleRate);
    }

    const nfft = 512;
    const hop = 256;

    const { magnitudes, phases, numFrames } = runSTFT(signal, nfft, hop);
    expect(numFrames).toBeGreaterThan(5);

    const reconstructed = runISTFT(magnitudes, phases, nfft, hop, length);

    const start = nfft;
    const end = length - nfft;
    let errorSum = 0;
    let signalSum = 0;
    for (let i = start; i < end; i++) {
      const err = reconstructed[i] - signal[i];
      errorSum += err * err;
      signalSum += signal[i] * signal[i];
    }
    const relError = Math.sqrt(errorSum / (signalSum + 1e-12));
    expect(relError).toBeLessThan(0.15);
  });

  it('produces correct number of frames', () => {
    const signal = new Float32Array(2048);
    const { numFrames } = runSTFT(signal, 512, 256);
    expect(numFrames).toBe(9);
  });

  it('handles silent signal', () => {
    const signal = new Float32Array(1024);
    const { magnitudes, phases } = runSTFT(signal, 256, 128);
    const reconstructed = runISTFT(magnitudes, phases, 256, 128, 1024);
    const maxAbs = Math.max(...reconstructed.map(Math.abs));
    expect(maxAbs).toBeLessThan(1e-5);
  });
});
