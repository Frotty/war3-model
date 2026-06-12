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

CASC extraction and BLP decode are already fast and cached, so the remaining cost is in (A) MDX
parsing and (B) the WebGL2 render/update loop. Findings below are ordered by expected payoff.
File/line references point at the current code so the work is unambiguous.

> **Status:** items A1, B3, B4 and B5 below are **implemented** (see the ✅ notes). They are
> validated by typecheck, lint, and a 12-model parse + round-trip test (geometry byte-identical
> after `generateMDX` → `parseMDX`). The render-loop changes (B3–B5) are WebGL2-gated and need an
> in-browser smoke test against an SD and an HD model before release. A2 and C remain open.

### A. MDX parsing ([mdx/parse.ts](mdx/parse.ts))

1. **Bulk geometry arrays are read element-by-element through `DataView`.**
   `float32Array`, `int32Array`, `uint16Array`, `uint8Array` ([parse.ts:100](mdx/parse.ts#L100)) loop
   N times calling `getFloat32`/`getInt32` per element. For large geosets (vertices, normals,
   tangents, UVs, faces) this is the dominant parse cost.
   - **Fix:** for these bulk reads, build a typed-array view over a copied slice:
     `new Float32Array(this.ab.slice(pos, pos + len * 4))` (one `memcpy` + zero-cost view) instead
     of N `DataView` calls. MDX is little-endian and JS engines are little-endian, so the bytes map
     directly. Use `.slice()` (not `new Float32Array(ab, pos, len)`) because chunk offsets are **not
     guaranteed 4-byte aligned** — the direct view constructor throws on a misaligned `pos`.
   - Keep `DataView` for scalar header fields; only the large arrays benefit.
   - ✅ **Done:** `State.float32Array` / `uint8Array` / new `uint16Array` now slice-and-view
     ([parse.ts:100](mdx/parse.ts#L100)); geoset Vertices/Normals/Faces/VertexGroup/TVertices and the
     HD Tangents/SkinWeights reads go through them.

2. **Parse is fully synchronous and blocks the webview thread.** For inline rendering of many models,
   move `parseMDX` into a Web Worker (it only needs the `ArrayBuffer`, which is transferable) so the
   UI thread stays responsive and several models can parse in parallel.

### B. WebGL2 render/update loop ([renderer/modelRenderer.ts](renderer/modelRenderer.ts))

3. **Bone matrices are uploaded with up to 254 individual `uniformMatrix4fv` calls per frame.**
   The SD/HD hardware-skinning path loops `j = 0 … MAX_NODES (254)` and issues one
   `gl.uniformMatrix4fv` per node every frame ([modelRenderer.ts:1384](renderer/modelRenderer.ts#L1384)),
   using the `uNodesMatrices[i]` uniform-array locations resolved at init
   ([modelRenderer.ts:2627](renderer/modelRenderer.ts#L2627)).
   - **Fix:** a UBO is the textbook answer, but the SD shaders are GLSL ES 1.00 (WebGL1-compatible)
     and UBOs would force `#version 300 es`, breaking the WebGL1 fallback. Instead, pack all node
     matrices into one contiguous `Float32Array` (indexed by ObjectId, matching the shader's bone
     indices) and upload the whole `uNodesMatrices[]` array with a **single `uniformMatrix4fv`** call
     against the `[0]` location — WebGL fills consecutive array elements from one buffer. Same
     ~254→1 collapse, no shader change, both WebGL1 and WebGL2 keep working.
   - ✅ **Done:** packed buffer `nodesMatricesBuffer` + single upload
     ([modelRenderer.ts:1391](renderer/modelRenderer.ts#L1391)); init now resolves only the `[0]`
     uniform location instead of 254.

4. **Vertex attributes are re-bound and re-pointed on every draw call, every frame.**
   Both the HD branch ([modelRenderer.ts:1452](renderer/modelRenderer.ts#L1452)) and the SD per-layer
   loop ([modelRenderer.ts:1512](renderer/modelRenderer.ts#L1512)) call `bindBuffer` +
   `vertexAttribPointer` for position/normal/uv/group(/skin/weight/tangent) on each draw. In the SD
   path this repeats for *every material layer* of the geoset even though the geometry is identical.
   - **Fix:** use **WebGL2 Vertex Array Objects (VAOs)**. Record one VAO per geoset at load time
     (`initBuffers`), then `bindVertexArray` once per geoset in `render`. This removes dozens of
     redundant GL calls per geoset per frame and the per-layer re-binding entirely.
   - ✅ **Done (WebGL2 only):** `createGeosetVAO` records attribute layout in `initBuffers`; `render`
     binds the VAO per geoset and the old per-draw `bindBuffer`/`vertexAttribPointer` is gated behind
     `!this.useVAO`. The element buffer is still bound explicitly per draw so wireframe toggling
     works; VAOs are deleted in `destroy`. WebGL1 keeps the original path.

5. **`update()` re-evaluates every material layer's texture id every frame.**
   The loop at [modelRenderer.ts:948](renderer/modelRenderer.ts#L948) walks all materials × layers
   each frame; for the common case where `TextureID` is a plain `number` (static), it just re-copies
   the constant.
   - **Fix:** at load, precompute which layers actually have animated (`AnimVector`) texture/normal/
     ORM/reflection ids and only re-interp those per frame; copy static ids once.
   - ✅ **Done:** `initLayerTextureAnimations` resolves static ids once and records only keyframed
     entries in `animatedLayerTextures`; `update` now iterates just that list
     ([modelRenderer.ts](renderer/modelRenderer.ts)).

### C. Startup cost for inline / thumbnail use

6. **A fresh `ModelRenderer` + `initGL` recompiles the full shader set per model.**
   `initGL` ([modelRenderer.ts:659](renderer/modelRenderer.ts#L659)) compiles the SD/HD programs and,
   for HD models, also the env-cubemap, convolution, prefilter and BRDF-LUT programs plus an eager
   BRDF-LUT render ([modelRenderer.ts:676](renderer/modelRenderer.ts#L676)). The extension currently
   creates a brand-new renderer and context per loaded model
   ([mdxViewer.ts:476](../wurst4vscode/src/webview/mdxViewer.ts#L476)).
   - **Fix for inline rendering:** share **one** WebGL2 context and a renderer/program pool across
     thumbnails rather than one context per model (browsers also cap the number of live WebGL
     contexts). Shader programs are model-independent and can be compiled once and reused.
   - **Lazy-compile** the env/prefilter/BRDF programs: they are only needed when `render` is called
     with `env`/`useEnvironmentMap`. Inline thumbnails that don't use IBL should not pay for them.
   - For static thumbnails, render a **single frame on demand** instead of running a continuous
     `requestAnimationFrame` loop per preview.

### Suggested order of work
1 → 3 → 4 give the largest wins (parse bulk-read, bone UBO, VAOs). 6 matters specifically once many
models are rendered inline simultaneously. 2 and 5 are incremental. Profile with a heavy HD Reforged
model (many bones + multi-layer materials), which stresses every item above.
