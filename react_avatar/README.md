# HoloPi React Avatar

This Vite/React Three Fiber application renders the HoloPi avatar and retail scene. It polls the central orchestrator for audio, visemes, carousel state, focused products, and gesture commands. The rendered canvas and synthesized-audio track are published to MediaMTX as the `avatar` stream by WHIP.

## Configuration

Copy the example file before local development:

```bash
cp .env.example .env.local
```

Configure:

- `VITE_MEDIAMTX_URL`: WHIP publication endpoint for the rendered avatar stream.
- `VITE_ORCHESTRATOR_URL`: HTTP base URL for scene-state polling and control.

The browser-side code can read only Vite's `VITE_` variables. The separate Node/Puppeteer launcher reads `HOLOPI_REACT_URL`, `HOLOPI_PUPPETEER_DATA_DIR`, and `HOLOPI_PUPPETEER_CACHE_DIR` from its process environment; see the repository-level `.env.example`.

## Required Assets

The source repository does not include the complete project-specific `public/` assets. A full scene requires the original models, products, animations, audio, and related JSON files in their expected subdirectories. See `../docs/reproducibility/artifact-manifest.csv`.

## Development

```bash
npm install
npm run dev -- --host
```

Build verification:

```bash
npm run build
```

The central orchestrator launches both the Vite server and `launch-hologram.js` on the retained Windows deployment. The launcher uses Puppeteer to render the scene headlessly with GPU-oriented Chromium options. It is not the edge display receiver: `public/dome.html` is the separate WHEP page used to consume the returned `avatar` stream.
