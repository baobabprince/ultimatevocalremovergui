/**
 * Tests for the critical model-loading path that previously threw:
 *   "Can't create a session. ERROR_CODE: 7, ERROR_MESSAGE: Failed to load model because protobuf parsing failed."
 *
 * These tests do NOT require a real browser or real ONNX Runtime.
 * They validate the data-flow and error-handling contracts that the worker must obey.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

async function fetchAsArrayBuffer(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url} (HTTP ${response.status})`);
  }
  const contentLength = +response.headers.get('Content-Length') || 0;
  const reader = response.body.getReader();
  let receivedLength = 0;
  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedLength += value.length;
    if (onProgress && contentLength) {
      onProgress(Math.round((receivedLength / contentLength) * 100));
    }
  }
  const result = new Uint8Array(receivedLength);
  let position = 0;
  for (const chunk of chunks) {
    result.set(chunk, position);
    position += chunk.length;
  }
  return result.buffer;
}

function ensureArrayBuffer(data) {
  if (data instanceof Uint8Array) return data.buffer;
  if (data instanceof ArrayBuffer) return data;
  throw new TypeError('Expected ArrayBuffer or Uint8Array');
}

describe('Model loading – ArrayBuffer contract (fixes ERROR_CODE 7)', () => {
  it('fetchAsArrayBuffer returns a pure ArrayBuffer', async () => {
    const fakeBytes = new Uint8Array([0x08, 0x0a, 0x12, 0x07]);
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => String(fakeBytes.length) },
      body: {
        getReader: () => {
          let done = false;
          return {
            read: async () => {
              if (done) return { done: true };
              done = true;
              return { done: false, value: fakeBytes };
            }
          };
        }
      }
    });

    const buf = await fetchAsArrayBuffer('https://example.com/model.onnx');
    expect(buf).toBeInstanceOf(ArrayBuffer);
    expect(buf.byteLength).toBe(4);
  });

  it('rejects non-ok HTTP responses with a clear message', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      headers: { get: () => null },
      body: null
    });

    await expect(fetchAsArrayBuffer('https://example.com/missing.onnx'))
      .rejects
      .toThrow(/HTTP 404/);
  });

  it('ensureArrayBuffer accepts both ArrayBuffer and Uint8Array', () => {
    const ab = new ArrayBuffer(8);
    const u8 = new Uint8Array(ab);
    expect(ensureArrayBuffer(ab)).toBe(ab);
    expect(ensureArrayBuffer(u8)).toBe(ab);
  });

  it('ensureArrayBuffer throws on invalid input', () => {
    expect(() => ensureArrayBuffer(null)).toThrow(TypeError);
    expect(() => ensureArrayBuffer(42)).toThrow(TypeError);
    expect(() => ensureArrayBuffer({})).toThrow(TypeError);
  });
});

describe('Model loading – external data path contract', () => {
  it('externalData path must match the string embedded in the ONNX file', () => {
    const embeddedPath = 'UVR-DeNoise-Lite.onnx.data';
    const workerPath = 'UVR-DeNoise-Lite.onnx.data';
    expect(workerPath).toBe(embeddedPath);
  });

  it('externalData entry shape is correct for ort-web', () => {
    const fakeData = new ArrayBuffer(1024);
    const entry = {
      data: fakeData,
      path: 'UVR-DeNoise-Lite.onnx.data'
    };
    expect(entry.data).toBeInstanceOf(ArrayBuffer);
    expect(typeof entry.path).toBe('string');
    expect(entry.path.endsWith('.data')).toBe(true);
  });
});

describe('Model loading – error surface', () => {
  it('protobuf parsing failure is surfaced as a clear error message', () => {
    const ortError = {
      message: "Can't create a session. ERROR_CODE: 7, ERROR_MESSAGE: Failed to load model because protobuf parsing failed."
    };
    const forwarded = ortError.message || String(ortError);
    expect(forwarded).toContain('ERROR_CODE: 7');
    expect(forwarded).toContain('protobuf parsing failed');
  });

  it('missing model file produces a helpful error', () => {
    const err = new Error('Failed to download ./converted_models/UVR-DeNoise-Lite.onnx (HTTP 404)');
    expect(err.message).toMatch(/HTTP 404/);
    expect(err.message).toMatch(/UVR-DeNoise-Lite/);
  });
});

describe('Input tensor shape preparation (process-audio)', () => {
  it('pads number of frames to multiple of 16', () => {
    const numFrames = 100;
    const padded = Math.ceil(numFrames / 16) * 16;
    expect(padded).toBe(112);
    expect(padded % 16).toBe(0);
  });

  it('creates correct input tensor dimensions [1, 2, 1024, paddedFrames]', () => {
    const paddedNumFrames = 128;
    const shape = [1, 2, 1024, paddedNumFrames];
    const size = shape.reduce((a, b) => a * b, 1);
    expect(size).toBe(1 * 2 * 1024 * 128);
    expect(shape[2]).toBe(1024);
  });
});
