## How to render model in browser

First of all, you need to get the model somehow. Load as external file, allow to user to provide it or maybe some other way.
Then you need all textures, so the model would render properly.

Place `<canvas>` in page somewhere, get it's rendering context and do some initial setup:

```ts
const canvas = document.querySelector('canvas');
try {
    gl = canvas.getContext('webgl');

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
} catch (err) {
    alert(err);
}
```

Create renderer instance:

```ts
const modelRenderer = new ModelRenderer(model);
```

Set textures:
```ts
modelRenderer.setTextureImage('somename.blp', img, textureFlags);
```

Init camera and model-view / projection martices:
```ts
...
mat4.perspective(pMatrix, Math.PI / 4 , canvas.width / canvas.height, 0.1, 10000.0);
mat4.lookAt(mvMatrix, cameraPos, cameraTarget, cameraUp);
```

And then call `update` + `setCamera` + `render` methods on `requestAnimationFrame`:

```ts
requestAnimationFrame(tick);

function tick(timestamp: number) {
    const delta = timstamp - start;
    modelRenderer.update(delta);
    modelRenderer.setCamera(cameraPos, cameraQuat);
    modelRenderer.render(mvMatrix, pMatrix);
}
```

This doc missing a lot of nuances, but the basic plan should be more clear now.

For detailed info please see [example source](https://github.com/4eb0da/war3-model/blob/master/docs/preview/preview.ts)

## Awaiting a complete frame

Initialize the renderer before loading textures. `loadTextures` calls your resolver once per unique nonempty model texture path and accepts decoded `imageData`, an HTML `image`, a `blp` buffer, or compressed `dds` data. The resolver handles archive paths and replaceable texture lookup in your application. Failed or missing textures reject by default.

```ts
renderer.initGL(gl);
await renderer.loadTextures(async texture => ({
    type: 'blp',
    buffer: await resolveModelTexture(texture.Image),
}), {timeoutMs: 15000});
renderer.setPose(0, 1); // evaluate 1ms after sequence start and reset effect/global clocks
await renderer.renderAsync(view, projection, {timeoutMs: 15000});
```

`setPose` and `renderAsync` do not schedule animation. Continue calling `update(delta)` to play, or stop calling it to keep the pose frozen. `whenReady()` also works with manual `setTextureImage*` uploads. `whenRendered()` resolves after the first texture-complete draw finishes on the GPU; the consumer must keep its render loop running until then. Both support `signal` and `timeoutMs`, and pending waits reject when the renderer is destroyed. `renderAsync` waits for the particular draw it submits.

## Cached thumbnails

```ts
import {renderModelThumbnail} from 'war3-model';

const thumbnail = await renderModelThumbnail(model, {
    width: 256,
    height: 256,
    loadTexture: async texture => ({
        type: 'blp',
        buffer: await resolveModelTexture(texture.Image),
    }),
    timeoutMs: 15000,
});
// Cache thumbnail.blob (PNG); display with URL.createObjectURL and revoke it later.
```

The helper defaults to Stand, evaluates 1ms after its start, waits for textures and GPU completion, fits posed visible mesh vertices and live effects, and downsamples a 2× render into the exact requested dimensions. Authored extents, unused vertices, hidden geosets and other LODs do not control framing. If the initial pose is empty, it searches the chosen sequence. Particles and ribbons automatically get a short warmup derived from 25% of the sequence duration, capped at 500ms of simulated time (about 30 small updates). Squirt emission keys select an earlier time for short bursts; one empty-bounds retry adds at most 250ms. This is computation, not a real-time wait. Use `frameOffsetMs` and `warmupMs` to override the pose; `warmupMs: 0` and `findVisibleFrame: false` capture an exact initial frame. Long delayed effects may still require an explicit offset.

On the default transparent background the helper measures visible pixels in a viewport no larger than 512 × 512, then redraws at the requested resolution, so transparent texture margins do not leave a tiny thumbnail. CPU readback stays at most 1 MiB and two scans; returned `bounds` remain geometric. Custom backgrounds retain geometric framing. Transparent WebGL captures also accumulate coverage alpha for additive meshes, particles and ribbons so visible RGB survives PNG encoding. Coverage follows final color brightness, making black texels transparent; fragment RGB is compensated to preserve the additive framebuffer color. Alpha approximates compositing the saved image. Regular viewer rendering retains model blend behavior. `ModelRenderer.setCaptureAlphaEnabled(true)` enables this explicitly for other WebGL capture consumers.

Thumbnail framing defaults to `zoom: 1.2`, trading slight edge clipping for more visible detail. Set `zoom: 1` for the full fit, or choose a multiplier between 0 and 4. The visible-pixel refinement also fills the frame more closely instead of accepting a subject at roughly two-thirds of its target span.

Optional settings include `sequence`, `cameraDirection` (Z up), `padding`, `background`, `supersampling`, `teamColor`, and `levelOfDetail`. Texture loading is strict by default; `allowMissingTextures: true` permits fallbacks and returns the unavailable paths in `missingTextures`. `useEnvironmentMap` defaults to false for fast editor captures. Model lights use bright SD defaults; `setLightingOptions({ambient, diffuse})` tunes an existing renderer, and Unshaded layers bypass lighting.

For a thumbnail batch, pass a reusable thumbnail-only WebGL2 `canvas` created with `preserveDrawingBuffer: true`, and await each capture before reusing it. Concurrent captures on separate canvases are supported. Captures release their own textures and model buffers. Call `ModelRenderer.releaseSharedResources(gl)` when disposing a reused context. Thumbnail generation requires browser canvas/WebGL2 support and supports cancellation via `signal`.

### Keeping thumbnail textures warm

`ThumbnailSession` owns a reusable context, queues captures, and retains a bounded LRU cache of uploaded GPU textures. Mesh and effect shader programs are shared per context too. It skips file loading, BLP decoding and GPU upload on a cache hit.

```ts
import {ThumbnailSession} from 'war3-model';

const session = new ThumbnailSession({maxCachedTextureBytes: 64 * 1024 * 1024});
const result = await session.render(model, {
    width: 128, height: 128,
    loadTexture: resolveTexture,
    textureNamespace: archiveId,
    maxTextureSize: 256,
});
// Repeated captures retain shared textures until LRU eviction or explicit cleanup.
console.log(session.getCacheStats());
await session.clearTextures(); // invalidate changed assets
await session.destroy();
```

Use the same `textureNamespace` for models sharing the same asset archive/root, and different namespaces when identical paths can resolve to different files. Cache identity includes path, wrap flags and texture cap. Changing texture contents under the same identity requires `clearTextures`. The default cache budget is 64 MiB, conservatively accounting as RGBA with mipmaps. A texture exceeding the budget is rendered but not retained. Each call's resolver is used only on misses; simultaneous calls to a session are queued.

The namespace and texture cap also isolate derived environment maps when they are enabled. `clearTextures` invalidates those derived images as well, while keeping programs warm. The LRU budget counts source image textures; optional derived environment maps are separate shared resources. Other renderers sharing a context can call `setEnvironmentMapNamespace(archiveId)` before uploading environment textures, and `ModelRenderer.releaseSharedEnvironmentMaps(gl)` to invalidate derived images.

Thumbnail texture resolution defaults to a maximum edge of 512 pixels; set `maxTextureSize: 0` for full resolution. BLP/DDS sources select an existing smaller mip before upload (BLP also avoids decoding the large mip). Image sources are resized when needed. Compressed DDS without a sufficiently small authored mip rejects rather than exceeding the cap. `renderer.setTextureSizeLimit(256)` applies the same policy to subsequent manual or async uploads for inline previews; ordinary renderers default to original resolution.

For quick previews, use low texture caps, skip environment processing before initialization with `setEnvironmentMapProcessingEnabled(false)`, and render with `useEnvironmentMap: false`. Full viewers can keep original textures and enable environment maps. Existing geometry, UVs, animation, material flags and sorting are retained in both configurations.

Run `npm run build-lib && npm test`, `npm run typecheck`, and `npm run lint` for regressions. For actual GPU/canvas checks, run `npm run dev` and open `/test/browser-renderer.html`. It checks PNG pixels, framing, shading, warm-cache reuse/eviction, texture resizing, readiness, WGSL compilation and a WebGPU SD draw when an adapter is available. Optional `?mdl=/_build/Azerite01.mdl&blp=/_build/Azerite01.blp` checks the supplied Azerite fixture after copying those assets into the ignored `_build` directory.
