import {mat4, quat, vec3} from 'gl-matrix';
import {Model} from '../model';
import {ModelRenderer} from './modelRenderer';
import {boundsCorners, emptyBounds, expandBounds, VisibleBounds} from './geometry';
import {TextureLoader} from './textureLoader';
import {abortError, cancellable, WaitOptions} from './readiness';

const activeCanvases = new WeakSet<HTMLCanvasElement | OffscreenCanvas>();

interface CachedTexture {texture: WebGLTexture; bytes: number}
interface ThumbnailCache {
    textures: Map<string, CachedTexture>;
    maximumBytes: number;
    bytes: number;
}

export interface ThumbnailOptions extends WaitOptions {
    width: number;
    height: number;
    loadTexture: TextureLoader;
    /** Reuse a thumbnail-only canvas/context across a batch. It must preserve its drawing buffer. */
    canvas?: HTMLCanvasElement | OffscreenCanvas;
    sequence?: number;
    /** Milliseconds after the sequence start; defaults to one evaluated frame (1ms). */
    frameOffsetMs?: number;
    /** Search the chosen sequence if its initial pose contains no visible geometry. */
    findVisibleFrame?: boolean;
    /** Simulate particles/trails before capturing; zero by default. */
    warmupMs?: number;
    levelOfDetail?: number;
    padding?: number;
    /** Direction from the model toward the camera in WC3's Z-up coordinate system. */
    cameraDirection?: vec3;
    background?: [number, number, number, number];
    supersampling?: number;
    allowMissingTextures?: boolean;
    teamColor?: vec3;
    useEnvironmentMap?: boolean;
    /** Longest uploaded texture edge; defaults to 512. Zero keeps original resolution. */
    maxTextureSize?: number;
    /** Asset/archive identity for warm sessions; paths alone may collide between models. */
    textureNamespace?: string;
}

export interface ThumbnailSessionOptions {
    canvas?: HTMLCanvasElement | OffscreenCanvas;
    /** Conservative RGBA+mips accounting; defaults to 64 MiB. */
    maxCachedTextureBytes?: number;
}

/** Sequential captures share a context and bounded LRU GPU texture cache. */
export class ThumbnailSession {
    private readonly canvas: HTMLCanvasElement | OffscreenCanvas;
    private readonly cache: ThumbnailCache;
    private disposed = false;
    private readonly ownsCanvas: boolean;
    private pending = Promise.resolve();

    constructor ({canvas, maxCachedTextureBytes = 64 * 1024 * 1024}: ThumbnailSessionOptions = {}) {
        if (!Number.isFinite(maxCachedTextureBytes) || maxCachedTextureBytes < 0) throw new Error('Invalid texture cache budget');
        this.canvas = canvas || newCanvas(1, 1);
        this.ownsCanvas = !canvas;
        this.cache = {textures: new Map(), bytes: 0, maximumBytes: maxCachedTextureBytes};
    }

    public render (model: Model, options: Omit<ThumbnailOptions, 'canvas'>): Promise<ThumbnailResult> {
        const result = this.pending.then(() => {
            if (this.disposed) throw new Error('Thumbnail session was destroyed');
            return renderThumbnail(model, {...options, canvas: this.canvas}, this.cache);
        });
        this.pending = result.then(() => {}, () => {});
        return result;
    }

    public getCacheStats (): {textures: number; estimatedBytes: number} {
        return {textures: this.cache.textures.size, estimatedBytes: this.cache.bytes};
    }

    /** Queue cleanup after current captures; the next capture reloads textures. */
    public clearTextures (): Promise<void> {
        const clear = this.pending.then(() => this.clearCache());
        this.pending = clear;
        return clear;
    }

    public async destroy (): Promise<void> {
        this.disposed = true;
        await this.pending;
        this.clearCache();
        const gl = this.canvas.getContext('webgl2', {preserveDrawingBuffer: true}) as WebGL2RenderingContext;
        if (gl) ModelRenderer.releaseSharedResources(gl);
        if (gl && this.ownsCanvas) gl.getExtension('WEBGL_lose_context')?.loseContext();
    }

    private clearCache (): void {
        const gl = this.canvas.getContext('webgl2', {preserveDrawingBuffer: true}) as WebGL2RenderingContext;
        this.cache.textures.forEach(entry => gl?.deleteTexture(entry.texture));
        this.cache.textures.clear();
        this.cache.bytes = 0;
        if (gl) ModelRenderer.releaseSharedEnvironmentMaps(gl);
    }
}

export interface ThumbnailResult {
    blob: Blob;
    width: number;
    height: number;
    sequence: number;
    frame: number;
    bounds: VisibleBounds;
    missingTextures: string[];
}

export interface ThumbnailCamera {
    view: mat4;
    projection: mat4;
    position: vec3;
    rotation: quat;
}

/** Orthographic framing is stable across model scales and avoids oversized authored extents. */
export function fitThumbnailCamera (bounds: VisibleBounds, width: number, height: number,
    direction: vec3 = vec3.fromValues(1, -1, 0.65), padding = 0.08): ThumbnailCamera {
    if (!Number.isFinite(width) || !Number.isFinite(height) || !(width > 0 && height > 0) || !Number.isFinite(padding) || padding < 0 || padding >= 0.5 ||
        [0, 1, 2].some(axis => !Number.isFinite(bounds.minimum[axis]) || !Number.isFinite(bounds.maximum[axis]) || bounds.minimum[axis] > bounds.maximum[axis])) {
        throw new Error('Invalid thumbnail framing options');
    }
    const center = vec3.scale(vec3.create(), vec3.add(vec3.create(), bounds.minimum, bounds.maximum), 0.5);
    const diagonal = Math.max(vec3.distance(bounds.minimum, bounds.maximum), 0.01);
    const ray = vec3.normalize(vec3.create(), direction);
    if (!vec3.length(ray) || ![ray[0], ray[1], ray[2]].every(Number.isFinite)) throw new Error('Camera direction must be finite and non-zero');
    const position = vec3.scaleAndAdd(vec3.create(), center, ray, diagonal * 2);
    const up = Math.abs(ray[2]) > 0.99 ? vec3.fromValues(0, 1, 0) : vec3.fromValues(0, 0, 1);
    const view = mat4.lookAt(mat4.create(), position, center, up);
    const cameraBounds = emptyBounds();
    for (const corner of boundsCorners(bounds)) expandBounds(cameraBounds, vec3.transformMat4(corner, corner, view));
    const sizeX = Math.max(cameraBounds.maximum[0] - cameraBounds.minimum[0], diagonal * 0.001);
    const sizeY = Math.max(cameraBounds.maximum[1] - cameraBounds.minimum[1], diagonal * 0.001);
    const halfHeight = Math.max(sizeY, sizeX * height / width) / (2 * (1 - 2 * padding));
    const halfWidth = halfHeight * width / height;
    const projection = mat4.ortho(mat4.create(), -halfWidth, halfWidth, -halfHeight, halfHeight,
        Math.max(0.001, -cameraBounds.maximum[2] - diagonal * 0.1), -cameraBounds.minimum[2] + diagonal * 0.1);
    // WC3's billboard reference plane faces +X, while a camera view faces -Z.
    const inverse = mat4.invert(mat4.create(), view);
    const cameraRotation = mat4.getRotation(quat.create(), inverse);
    const basis = quat.rotationTo(quat.create(), vec3.fromValues(1, 0, 0), vec3.fromValues(0, 0, 1));
    const rotation = quat.mul(quat.create(), cameraRotation, basis);
    return {view, projection, position, rotation};
}

function newCanvas (width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
    if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
    if (typeof document === 'undefined') throw new Error('Thumbnail rendering requires a browser canvas');
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
}

/** Loads all resources, freezes a useful pose, fits the visible mesh, and returns an encoded PNG. */
export async function renderModelThumbnail (model: Model, options: ThumbnailOptions): Promise<ThumbnailResult> {
    return renderThumbnail(model, options);
}

async function renderThumbnail (model: Model, options: ThumbnailOptions, cache?: ThumbnailCache): Promise<ThumbnailResult> {
    if (options.signal?.aborted) throw abortError();
    if (options.canvas && activeCanvases.has(options.canvas)) throw new Error('Thumbnail canvas is already rendering; await its previous capture');
    if (options.canvas) activeCanvases.add(options.canvas);
    try {
        return await captureThumbnail(model, options, cache);
    } finally {
        if (options.canvas) activeCanvases.delete(options.canvas);
    }
}

async function captureThumbnail (model: Model, options: ThumbnailOptions, cache?: ThumbnailCache): Promise<ThumbnailResult> {
    const {width, height, supersampling = 2, levelOfDetail = 0, warmupMs = 0} = options;
    if (![width, height, supersampling].every(value => Number.isInteger(value) && value > 0) ||
        width * supersampling > 8192 || height * supersampling > 8192 ||
        !Number.isFinite(warmupMs) || warmupMs < 0 || warmupMs > 10000) {
        throw new Error('Invalid thumbnail size, supersampling, or warmup');
    }
    // Acquire limits on a tiny backing buffer before allocating the requested capture size.
    const canvas = options.canvas || newCanvas(1, 1);
    const gl = canvas.getContext('webgl2', {alpha: true, antialias: true, preserveDrawingBuffer: true}) as WebGL2RenderingContext;
    if (!gl) throw new Error('WebGL2 is required for thumbnails');
    if (!gl.getContextAttributes()?.preserveDrawingBuffer) throw new Error('Thumbnail canvas must preserve its drawing buffer');
    const renderer = new ModelRenderer(model);
    const maxTextureSize = options.maxTextureSize ?? 512;
    const cacheKey = (path: string): string => JSON.stringify([options.textureNamespace || '', path,
        model.Textures.find(texture => texture.Image === path)?.Flags || 0, maxTextureSize]);
    let captured: ThumbnailResult;
    try {
        const viewport = gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array;
        const renderbufferLimit = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number;
        const backingWidth = width * supersampling;
        const backingHeight = height * supersampling;
        if (backingWidth > Math.min(renderbufferLimit, viewport[0]) || backingHeight > Math.min(renderbufferLimit, viewport[1])) {
            throw new Error('Thumbnail backing size exceeds WebGL context limits');
        }
        canvas.width = backingWidth;
        canvas.height = backingHeight;
        if (gl.isContextLost() || gl.drawingBufferWidth !== backingWidth || gl.drawingBufferHeight !== backingHeight) {
            throw new Error('Unable to allocate the requested thumbnail drawing buffer');
        }
        renderer.setEnvironmentMapProcessingEnabled(options.useEnvironmentMap ?? false);
        renderer.setEnvironmentMapNamespace(options.textureNamespace || '');
        renderer.setTextureSizeLimit(maxTextureSize);
        renderer.initGL(gl);
        model.Textures.forEach(texture => {
            const key = cacheKey(texture.Image);
            const entry = cache?.textures.get(key);
            if (entry && gl.isTexture(entry.texture)) {
                renderer.adoptTexture(texture.Image, entry.texture);
                cache.textures.delete(key);
                cache.textures.set(key, entry);
            } else if (entry) {
                cache.textures.delete(key);
                cache.bytes -= entry.bytes;
            }
        });
        if (options.teamColor) renderer.setTeamColor(options.teamColor);
        const missingTextures = await renderer.loadTextures(options.loadTexture, options);
        const stand = model.Sequences.findIndex(sequence => /^stand(?:\s|$)/i.test(sequence.Name));
        const sequence = options.sequence ?? (stand >= 0 ? stand : 0);
        let offset = options.frameOffsetMs ?? 1;
        renderer.setPose(sequence, offset);
        const interval = model.Sequences[sequence]?.Interval || [0, 0];
        let bounds = renderer.getVisibleBounds({levelOfDetail});
        if (!bounds && !warmupMs && options.findVisibleFrame !== false) {
            for (let step = 1; step <= 16 && !bounds; ++step) {
                offset = (interval[1] - interval[0]) * step / 16;
                renderer.setPose(sequence, offset);
                bounds = renderer.getVisibleBounds({levelOfDetail});
            }
        }
        for (let remaining = warmupMs; remaining > 0;) {
            const delta = Math.min(remaining, 1000 / 60);
            renderer.update(delta);
            remaining -= delta;
        }
        bounds = renderer.getVisibleBounds({levelOfDetail});
        if (!bounds) throw new Error('The selected pose has no visible geometry; try warmupMs for particle-only models');
        // Billboards change bounds when the camera changes. Refit their evaluated pose as well.
        let camera = fitThumbnailCamera(bounds, width, height, options.cameraDirection, options.padding);
        for (let step = 0; step < 3; ++step) {
            renderer.setCamera(camera.position, camera.rotation);
            renderer.update(0);
            bounds = renderer.getVisibleBounds({levelOfDetail}) || bounds;
            camera = fitThumbnailCamera(bounds, width, height, options.cameraDirection, options.padding);
        }
        renderer.setCamera(camera.position, camera.rotation);
        const background = options.background || [0, 0, 0, 0];
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(background[0], background[1], background[2], background[3]);
        gl.depthMask(true);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        await renderer.renderAsync(camera.view, camera.projection, {...options, levelOfDetail});
        // Transparent texture regions can make geometric bounds much larger than the visible
        // subject (notably portraits). Reframe on the GPU rather than enlarging a blurry crop.
        if (background.every(channel => channel === 0)) {
            // Downsample before readback: both CPU memory and synchronous scanning stay
            // bounded even when the requested supersampled backing buffer is very large.
            const scale = Math.min(1, 512 / Math.max(canvas.width, canvas.height));
            const measureWidth = Math.max(1, Math.round(canvas.width * scale));
            const measureHeight = Math.max(1, Math.round(canvas.height * scale));
            const measurement = newCanvas(measureWidth, measureHeight);
            const measurementContext = measurement.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
            if (!measurementContext) throw new Error('Unable to create thumbnail measurement canvas');
            for (let attempt = 0; attempt < 2; ++attempt) {
                measurementContext.clearRect(0, 0, measureWidth, measureHeight);
                measurementContext.drawImage(canvas, 0, 0, measureWidth, measureHeight);
                const pixels = measurementContext.getImageData(0, 0, measureWidth, measureHeight).data;
                let minX = measureWidth, minY = measureHeight, maxX = -1, maxY = -1, count = 0;
                for (let offset = 0; offset < pixels.length; offset += 4) {
                    if (Math.max(pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]) <= 12) continue;
                    const pixel = offset / 4, x = pixel % measureWidth, y = Math.floor(pixel / measureWidth);
                    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
                    count++;
                }
                const span = Math.max((maxX - minX + 1) / measureWidth, (maxY - minY + 1) / measureHeight);
                const target = 1 - 2 * (options.padding ?? 0.08);
                if (count < 4 || span >= target * 0.75) break;
                const zoom = Math.min(16, target / span);
                const centerX = (minX + maxX + 1) / measureWidth - 1;
                const centerY = 1 - (minY + maxY + 1) / measureHeight;
                for (const index of [0, 4, 8]) camera.projection[index] *= zoom;
                for (const index of [1, 5, 9]) camera.projection[index] *= zoom;
                camera.projection[12] = (camera.projection[12] - centerX) * zoom;
                camera.projection[13] = (camera.projection[13] - centerY) * zoom;
                gl.depthMask(true);
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                await renderer.renderAsync(camera.view, camera.projection, {...options, levelOfDetail});
            }
        }
        // Copy to a separate canvas before encoding; a reusable GL canvas may render another model later.
        const output = newCanvas(width, height);
        const context = output.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
        if (!context) throw new Error('Unable to create thumbnail output canvas');
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = 'high';
        context.drawImage(canvas, 0, 0, width, height);
        const encoding = 'convertToBlob' in output ? output.convertToBlob({type: 'image/png'}) :
            new Promise<Blob>((resolve, reject) => output.toBlob(blob => blob ? resolve(blob) : reject(new Error('PNG encoding failed')), 'image/png'));
        const blob = await cancellable(encoding, options);
        captured = {blob, width, height, sequence, frame: renderer.getFrame(), bounds, missingTextures};
    } finally {
        // Loader-created textures belong to this capture, even on a reused context.
        new Set(model.Textures.map(texture => texture.Image).filter(Boolean)).forEach(path => {
            const texture = renderer.getTexture(path);
            if (!texture) return;
            const key = cacheKey(path);
            if (cache?.textures.get(key)?.texture === texture) return;
            const dimensions = renderer.getTextureDimensions(path);
            const bytes = dimensions ? Math.ceil(dimensions[0] * dimensions[1] * 4 * 4 / 3) : Infinity;
            if (cache && bytes <= cache.maximumBytes) {
                cache.textures.set(key, {texture, bytes});
                cache.bytes += bytes;
            } else gl.deleteTexture(texture);
        });
        while (cache && cache.bytes > cache.maximumBytes && cache.textures.size) {
            const key = cache.textures.keys().next().value as string;
            const entry = cache.textures.get(key);
            gl.deleteTexture(entry.texture);
            cache.textures.delete(key);
            cache.bytes -= entry.bytes;
        }
        renderer.destroy();
        if (!options.canvas) {
            ModelRenderer.releaseSharedResources(gl);
            gl.getExtension('WEBGL_lose_context')?.loseContext();
        }
    }
    return captured;
}
