import { defineConfig } from 'vite'
import path from 'node:path'

export default defineConfig({
  server: {
    fs: {
      // Allow Vite to serve files from the project root and its public/ folder.
      // Slidev's slide-import-guard plugin treats `<img src="/figures/...">` as an
      // import and validates it against this list — defaults are too strict.
      strict: false,
      allow: [
        path.resolve(__dirname),
        path.resolve(__dirname, 'public'),
        path.resolve(__dirname, 'public', 'figures'),
      ],
    },
  },
})
