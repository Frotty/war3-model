# AGENTS Guide for `war3-model`

WC3 model toolkit: MDX/MDL parser, generator, converter, and WebGL/WebGPU previewer.

## Public entry points (`index.ts`)
- `parseMDX(buffer)` / `parseMDL(text)` — parse a model into the `model` object graph.
- `generateMDX` / `generateMDL` — serialise back out.
- `decodeBLP` / `getBLPImageData` — BLP texture decoding.
- `ModelRenderer` (from [renderer/modelRenderer.ts](renderer/modelRenderer.ts)) — the renderer.

## Consumed by `../wurst4vscode` (model preview feature)
The sibling VS Code extension depends on this package via `"war3-model": "file:../war3-model"`
and imports **exactly one renderer**: `ModelRenderer`.

- Import site: [../wurst4vscode/src/webview/mdxViewer.ts:1](../wurst4vscode/src/webview/mdxViewer.ts#L1)
  — `import { parseMDL, parseMDX, ModelRenderer, decodeBLP, getBLPImageData } from 'war3-model'`.
- The extension drives it through the **WebGL2 path only**: it creates the context with
  `canvas.getContext('webgl2', …)` ([mdxViewer.ts:472](../wurst4vscode/src/webview/mdxViewer.ts#L472))
  and calls `renderer.initGL(gl)` — *not* `initGPUDevice`. The WebGPU branch of `ModelRenderer`
  is never exercised by the extension.
- Per-frame loop: `setCamera` → `update(delta)` → `render(mv, proj, { wireframe })` inside a
  `requestAnimationFrame` loop ([mdxViewer.ts:270](../wurst4vscode/src/webview/mdxViewer.ts#L270)).
- Models are categorised at construction as **SD** or **HD** via
  `isHD = model.Geosets?.some(g => g.SkinWeights?.length > 0)` ([modelRenderer.ts:384](renderer/modelRenderer.ts#L384)).
  Classic units take the SD path; Reforged units take the HD (PBR + env-map) path.

> When optimising for the extension, the **WebGL2 SD/HD render path** and the **MDX binary parser**
> are what matter. The WebGPU path, the MDL text parser, and the generators are not on the hot path.

---

## Performance: making parse + render fast enough for inline rendering

Everything below has been implemented. The reference model for anything Reforged is
`arthas.mdx` at the repository root: v1200, 7.91 MB, 464,007 keyframes, 36 geosets across four
LOD levels, 24 DDS textures plus `ReplaceableTextures/EnvironmentMap.blp`.

Verified by `npm run typecheck`, `npm run lint` and `npm test`. The test suite covers the
AnimVector round trip, the BLP decoder against an independent per-pixel reference, and the
animation output of every model in the repo (`ModelRenderer.update()` runs without a GL context,
so the whole interpolator is exercised headlessly). The GL-only changes still want an in-browser
smoke test against an SD model, an HD model, and a wireframe toggle.

### What the numbers were

Measured on Node 22.17.1 / V8. GPU sample counts are counted from the shader sources, not timed.

| | before | after |
| --- | --- | --- |
| `parseMDX(arthas.mdx)` | 32.2 ms | 6.6 ms |
| `update()` × 600 frames, arthas | 20.2 ms | 16.5 ms |
| BLP1 Direct 1024², all mips | 8.50 ms | 2.90 ms |
| BLP1 Direct 1024², level 0 | 5.94 ms | 1.86 ms |
| alpha histogram per 1024² texture | 1.70 ms | 0 (debug only) |
| env cubemap allocation | 256 MB | 3.1 MB |
| IBL precompute | ~230M fetches per model | once per context |

### Where the work went

**A. Parsing.** Bulk geometry arrays read through `ArrayBuffer.slice()` + a typed-array view
rather than N `DataView` calls. `AnimVector` now stores keyframes in flat arrays — `Frames`,
`Values`, `InTans`, `OutTans`, `VectorSize` — instead of one object plus one small typed array
per keyframe; `AnimVector.Keys` is a lazy accessor materialising views over that storage, so the
generators and `docs/optframes` keep working and the renderer never pays for the objects.
Assigning `Keys` rebuilds the flat storage. The accessor lives on a prototype, not per instance:
`Object.defineProperty` per object drops each one into dictionary mode and costs more per frame
than the lazy materialisation saves.

**B. Render loop.** Node matrices upload in a single `uniformMatrix4fv`. WebGL2 VAOs record
per-geoset attribute layout once. Static layer texture ids resolve at load; only keyframed ones
re-interpolate per frame. Per-material reflection-layer and replaceable-mask lookups resolve at
load too (`initMaterialLookups`), except for materials with keyframed texture ids, which still
resolve live. Light/camera/shadow uniforms and the BRDF LUT bind hoisted out of the geoset loop.
Blend, depth and cull state is shadowed and only re-issued on change (`applyLayerState`), reset
once per frame. Debug logging resolves its flag once in the constructor and every hot-path call
site guards on it, so keys and payload objects are never built.

**C. Textures.** The alpha histogram in `setTextureImageData` is behind the debug flag. The
palettized BLP decoder writes one 32-bit store per pixel through a palette LUT. Mip levels are
allocated with `texStorage2D` and filled with `texSubImage2D` from the raw view; a caller passing
only level 0 gets the rest from `generateMipmap`. Anisotropy defaults to 4 (`setMaxAnisotropy`)
and the driver maximum is queried once per context, not once per texture. Geoset buffers and VAOs
upload lazily on first draw, so LOD levels that never render never upload.

**D. Environment maps.** `ENV_MAP_SIZE` is 256, not 2048 — the intermediate cubemap only feeds a
32² irradiance map and a 128² prefiltered map, and nothing else samples it except the optional
skybox. Irradiance and prefiltered cubemaps are cached per context by texture path
(`sharedEnvMaps`), as is the BRDF LUT (`sharedBrdfLUT`), because none of them depend on the
model; `destroy()` leaves them alone and `ModelRenderer.releaseSharedResources(gl)` frees them.
The diffuse convolution samples at `sampleDelta = 0.05`; the prefilter takes 256 importance
samples with a PDF-derived `textureLod` mip instead of 1024 at level 0. The prefilter target is
no longer hand-zeroed after `texStorage2D` already allocated it.

**E. Missing textures.** `initFallbackTextures` binds 1×1 white / flat-normal / default-ORM
stand-ins wherever a model texture has not loaded. Without them the sampler is left unbound,
WebGL returns `(0, 0, 0, 1)`, and HD models render as a black silhouette until every texture
arrives.

**F. Shader compilation.** `getShader` no longer queries `COMPILE_STATUS` straight after
`compileShader`, which forced a synchronous driver stall per shader; `checkProgram` validates
after linking and reports whichever stage failed. Particle and ribbon controllers skip compiling
and drawing entirely when the model has no emitters.

### Still open

- **Decode in a worker.** `blp/decode.ts` and `third_party/decoder.js` are pure
  `ArrayBuffer` → `ArrayBuffer` and have no DOM dependency, so they can move off the UI thread.
- **One context, one program set.** The extension still builds a fresh WebGL2 context and
  recompiles the full shader set per model. The env-map and BRDF caches above are already keyed
  by context and pay off the moment a context is shared; programs vary only by
  SD / HD / software-skinning, so a three-entry cache covers everything.
- **Do not micro-optimise `third_party/decoder.js`.** Measured: replacing the 8×8 de-block copy
  with `set()`/`subarray()` is three times *slower* (4.56 ms → 13.97 ms), and hoisting the scale
  multiply out of the resample loop changes nothing (3.30 ms → 3.29 ms). Win by calling it fewer
  times, not by rewriting it.
