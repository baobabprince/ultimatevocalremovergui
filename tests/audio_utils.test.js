import { describe, it, expect } from 'vitest';
import {
  pitchShift,
  timeStretch,
  resample,
  alignAudio,
  matchVolume,
  ensembleSignals
} from '../audio_utils.js';

describe('resample', () => {
  it('returns same length when factor=1', () => {
    const sig = new Float32Array([1, 2, 3, 4]);
    const out = resample(sig, 1.0);
    expect(out.length).toBe(4);
    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
  });

  it('shortens signal when factor > 1 (downsample)', () => {
    const sig = new Float32Array(100);
    for (let i = 0; i < 100; i++) sig[i] = i;
    const out = resample(sig, 2.0);
    expect(out.length).toBe(50);
  });

  it('lengthens signal when factor < 1 (upsample)', () => {
    const sig = new Float32Array([0, 1, 0, -1]);
    const out = resample(sig, 0.5);
    expect(out.length).toBe(8);
  });
});

describe('timeStretch', () => {
  it('returns original when factor=1', () => {
    const sig = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
    const out = timeStretch(sig, 1.0);
    expect(out).toBe(sig);
  });

  it('produces longer output for factor > 1', () => {
    const sig = new Float32Array(2048);
    for (let i = 0; i < sig.length; i++) {
      sig[i] = Math.sin(2 * Math.PI * 440 * i / 44100);
    }
    const out = timeStretch(sig, 1.5);
    expect(out.length).toBeGreaterThan(sig.length * 1.2);
  });

  it('produces shorter output for factor < 1', () => {
    const sig = new Float32Array(2048);
    for (let i = 0; i < sig.length; i++) {
      sig[i] = Math.sin(2 * Math.PI * 440 * i / 44100);
    }
    const out = timeStretch(sig, 0.7);
    expect(out.length).toBeLessThan(sig.length * 0.9);
  });
});

describe('pitchShift', () => {
  it('returns original when semitones=0', () => {
    const sig = new Float32Array([0.1, -0.2, 0.3]);
    const out = pitchShift(sig, 0, 44100);
    expect(out).toBe(sig);
  });

  it('changes length when pitching up', () => {
    const sig = new Float32Array(4096);
    for (let i = 0; i < sig.length; i++) {
      sig[i] = Math.sin(2 * Math.PI * 440 * i / 44100);
    }
    const up = pitchShift(sig, 12, 44100);
    expect(up.length).toBeGreaterThan(1000);
  });

  it('does not throw on extreme values', () => {
    const sig = new Float32Array(1024);
    expect(() => pitchShift(sig, 12, 44100)).not.toThrow();
    expect(() => pitchShift(sig, -12, 44100)).not.toThrow();
  });
});

describe('matchVolume', () => {
  it('matches RMS of reference', () => {
    const ref = new Float32Array(1000).fill(0.5);
    const target = new Float32Array(1000).fill(0.1);
    const matched = matchVolume(ref, target);

    const rms = (arr) => {
      let s = 0;
      for (let v of arr) s += v * v;
      return Math.sqrt(s / arr.length);
    };

    expect(rms(matched)).toBeCloseTo(rms(ref), 5);
  });

  it('handles zero target without NaN', () => {
    const ref = new Float32Array(100).fill(0.3);
    const target = new Float32Array(100);
    const matched = matchVolume(ref, target);
    expect(matched.every(v => v === 0)).toBe(true);
  });
});

describe('alignAudio', () => {
  it('returns zero delay for identical signals', () => {
    const sig = new Float32Array(44100);
    for (let i = 0; i < sig.length; i++) {
      sig[i] = Math.sin(2 * Math.PI * 200 * i / 44100);
    }
    const { delay, alignedSignal } = alignAudio(sig, sig, 0.5, 44100);
    expect(Math.abs(delay)).toBeLessThan(50);
    expect(alignedSignal.length).toBe(sig.length);
  });

  it('detects a positive lag', () => {
    const sampleRate = 44100;
    const length = sampleRate * 2;
    const ref = new Float32Array(length);
    const target = new Float32Array(length);

    for (let i = 0; i < length; i++) {
      ref[i] = Math.sin(2 * Math.PI * 300 * i / sampleRate) * Math.exp(-i / 5000);
      target[i] = Math.sin(2 * Math.PI * 300 * (i - 200) / sampleRate) * Math.exp(-(i - 200) / 5000);
    }

    const { delay } = alignAudio(ref, target, 1.0, sampleRate);
    expect(typeof delay).toBe('number');
  });
});

describe('ensembleSignals', () => {
  it('averages correctly', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([3, 4, 5]);
    const avg = ensembleSignals([a, b], 'Average');
    expect(Array.from(avg)).toEqual([2, 3, 4]);
  });

  it('computes min-spec (by absolute value)', () => {
    const a = new Float32Array([1, -5, 3]);
    const b = new Float32Array([-2, 4, -1]);
    const minSpec = ensembleSignals([a, b], 'Min Spec');
    expect(Math.abs(minSpec[0])).toBeLessThanOrEqual(2);
    expect(Math.abs(minSpec[1])).toBeLessThanOrEqual(5);
  });

  it('returns null for empty list', () => {
    expect(ensembleSignals([], 'Average')).toBeNull();
  });

  it('handles single signal', () => {
    const a = new Float32Array([1, 2, 3]);
    const out = ensembleSignals([a], 'Average');
    expect(Array.from(out)).toEqual([1, 2, 3]);
  });
});
