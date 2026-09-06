# Tests – Ultimate Vocal Remover Web Edition

## What is covered

| Suite | File | What it tests |
|-------|------|---------------|
| STFT / iSTFT | `stft.test.js` | FFT correctness, reflection padding, round-trip reconstruction error, silent signal |
| Audio tools | `audio_utils.test.js` | pitchShift, timeStretch, resample, alignAudio, matchVolume, ensemble (Average / Min Spec) |
| Model loading | `model_loading.test.js` | **The exact ERROR_CODE 7 / protobuf parsing failure**, ArrayBuffer contract, external-data path, tensor shape padding |
| WAV export | `wav.test.js` | RIFF/WAVE header, clipping, empty buffers |

## Running the tests

```bash
npm install
npm test
```

Watch mode:

```bash
npm run test:watch
```

## Notes on the critical bug (ERROR_CODE 7)

The model-loading tests assert the contracts that were broken when the original
`onnxruntime-web@1.20.1` + incorrect `Uint8Array` / `ArrayBuffer` handling
produced:

```
Can't create a session. ERROR_CODE: 7,
ERROR_MESSAGE: Failed to load model because protobuf parsing failed.
```

After the fix (ort 1.22.0 + pure ArrayBuffer + correct externalData path)
these contracts must stay true.
