# Game-data thumbnail stress test

This opt-in test uses a local Warcraft III installation and a built sibling
`casc-ts` checkout. Game files and generated screenshots stay on your machine.
It requires Node 22 or newer and Chromium:

```sh
npm ci
npx playwright install chromium
```

Build `casc-ts` with `npm ci` and `npm run build` in that checkout, then run:

```sh
npm run test:game-data -- --game "C:/Program Files (x86)/Warcraft III"
```

`WC3_GAME_PATH` can supply the installation path. `--casc <directory>` overrides
the default `../casc-ts`. The runner expands every available `.w3mod` namespace,
including SD, HD, locale and censored variants, then extracts and attempts a real
WebGL2 thumbnail for every MDX/MDL file. It resolves texture extensions and parent
archive namespaces, supporting BLP, compressed DDS and TGA textures.

Each model tries Stand first, then other animations at three offsets, with 500ms
of simulation for particles/ribbons. A shared `ThumbnailSession` keeps textures
warm within a 32 MiB GPU budget; compressed sources have a separate 64 MiB budget.
Only one model renders at a time. Missing textures are reported and use renderer
fallbacks; this is a visibility assertion, not a texture-fidelity assertion.

A pixel counts as visible when any RGBA channel exceeds 12/255, including additive
effects. By default a capture must cover at least 2% of the pixels **and** span
at least 50% of the image along one axis. A majority-of-pixels requirement would
reject useful thin models and hollow effects. Use `--min-coverage 0.5` for that
stricter policy, or adjust `--min-span` independently. Empty camera-only assets
and invisible archive variants fail too; none are silently excluded.

The default output is `_build/game-data-thumbnails`: PNGs, a streaming
`results.jsonl` journal, and a final `results.json` containing coverage, span,
selected pose, timing, missing textures and model structure. A failed model does
not interrupt the census; any failure makes the command exit with status 1.
No parsed model, game texture or screenshot is committed to the repository.

Useful focused runs:

```sh
npm run test:game-data -- --game "C:/Program Files (x86)/Warcraft III" --filter "war3.w3mod:units/" --limit 20
npm run test:game-data -- --game "C:/Program Files (x86)/Warcraft III" --filter "banditmage_portrait" --size 256 --output _build/portrait-test
```

Other options: `--size` (16–2048, default 128), `--limit` (default all models),
`--output`, `--filter` (case-insensitive substring). Hardware and browser drivers
can change coverage at antialiased edges, so thresholds deliberately have margin.
The suite is separate from `npm test` because it needs proprietary installed data.
`npm run test:browser` runs synthetic renderer regressions without game files.
