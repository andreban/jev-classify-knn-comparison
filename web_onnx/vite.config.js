import { defineConfig } from 'vite'

export default defineConfig({
  base: '',
  build: {
    outDir: 'dist/toxicity-detection-onnx/',
    emptyOutDir: true,
  }
})
