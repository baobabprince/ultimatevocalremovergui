import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: false,
    include: ['tests/**/*.test.js'],
    testTimeout: 15000,
    coverage: {
      reporter: ['text', 'html'],
      include: ['lib/**', 'audio_utils.js']
    }
  }
});
