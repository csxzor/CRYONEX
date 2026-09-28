# Submission deck generator

`build.js` writes the 7-slide SIH idea-submission deck (`CRYONEX_SIH26153_CRYOBLUE.pptx`).
Every number in it comes from `results/` (see `docs/BENCHMARKS.md`); `heartbleed.json` is the
per-minute risk timeline of the Heartbleed demo sample exported from the release bundle.

```bash
cd docs/deck
npm install pptxgenjs react-icons react react-dom sharp   # once, outside the repo is fine
node build.js ../../CRYONEX_SIH26153_CRYOBLUE.pptx
```

Preview/QA: LibreOffice (e.g. `flatpak run org.libreoffice.LibreOffice --headless --convert-to pdf <deck>`).
