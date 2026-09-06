import { describe, it, expect } from 'vitest';

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function writeWav(left, right, sampleRate) {
  const buffer = new ArrayBuffer(44 + left.length * 2 * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + left.length * 2 * 2, true);
  writeString(view, 8, 'WAVE');

  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);

  writeString(view, 36, 'data');
  view.setUint32(40, left.length * 2 * 2, true);

  let index = 44;
  for (let i = 0; i < left.length; i++) {
    let valL = Math.max(-1, Math.min(1, left[i]));
    let valR = Math.max(-1, Math.min(1, right[i]));
    view.setInt16(index, valL < 0 ? valL * 0x8000 : valL * 0x7FFF, true);
    view.setInt16(index + 2, valR < 0 ? valR * 0x8000 : valR * 0x7FFF, true);
    index += 4;
  }

  return new Blob([view], { type: 'audio/wav' });
}

describe('writeWav', () => {
  it('produces a valid RIFF/WAVE header', async () => {
    const left = new Float32Array([0.5, -0.5, 0.25]);
    const right = new Float32Array([-0.5, 0.5, -0.25]);
    const blob = writeWav(left, right, 44100);

    expect(blob.type).toBe('audio/wav');
    expect(blob.size).toBe(44 + 3 * 4);

    const buf = await blob.arrayBuffer();
    const view = new DataView(buf);

    expect(String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3))).toBe('RIFF');
    expect(String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11))).toBe('WAVE');
    expect(String.fromCharCode(view.getUint8(12), view.getUint8(13), view.getUint8(14), view.getUint8(15))).toBe('fmt ');
    expect(view.getUint16(20, true)).toBe(1);
    expect(view.getUint16(22, true)).toBe(2);
    expect(view.getUint32(24, true)).toBe(44100);
    expect(view.getUint16(34, true)).toBe(16);
  });

  it('clips samples to [-1, 1]', async () => {
    const left = new Float32Array([2.0, -2.0]);
    const right = new Float32Array([1.5, -1.5]);
    const blob = writeWav(left, right, 48000);
    const buf = await blob.arrayBuffer();
    const view = new DataView(buf);

    const sample0L = view.getInt16(44, true);
    expect(sample0L).toBe(0x7FFF);

    const sample0R = view.getInt16(46, true);
    expect(sample0R).toBe(0x7FFF);
  });

  it('handles empty channels', async () => {
    const left = new Float32Array(0);
    const right = new Float32Array(0);
    const blob = writeWav(left, right, 44100);
    expect(blob.size).toBe(44);
  });
});
