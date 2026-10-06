/** Choose an authored mip before decoding/uploading; avoid touching the full-size image. */
export function textureMipLevel (width: number, height: number, levels: number, limit: number): number {
    if (!limit) return 0;
    let level = 0;
    while (level + 1 < levels && Math.max(width, height) > limit) {
        width = Math.max(1, Math.floor(width / 2));
        height = Math.max(1, Math.floor(height / 2));
        ++level;
    }
    return level;
}

function canvas (width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
    if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
    const result = document.createElement('canvas');
    result.width = width;
    result.height = height;
    return result;
}

export function resizeTexture (source: ImageData | HTMLImageElement, limit: number): ImageData {
    const input = canvas(source.width, source.height);
    const context = input.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    if (!context) throw new Error('Unable to resize texture');
    if ('data' in source) {
        const pixels = context.createImageData(source.width, source.height);
        pixels.data.set(source.data);
        context.putImageData(pixels, 0, 0);
    } else context.drawImage(source, 0, 0);
    const scale = Math.min(1, limit / Math.max(source.width, source.height));
    const output = canvas(Math.max(1, Math.floor(source.width * scale)), Math.max(1, Math.floor(source.height * scale)));
    const resized = output.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    if (!resized) throw new Error('Unable to resize texture');
    resized.imageSmoothingEnabled = true;
    resized.imageSmoothingQuality = 'high';
    resized.drawImage(input, 0, 0, output.width, output.height);
    return resized.getImageData(0, 0, output.width, output.height);
}
