import decodeJPEG from '../third_party/decoder';
import {BLPImage, BLPContent, BLPType} from './blpimage';

function keyword (view: DataView, offset: number): string {
    return String.fromCharCode(
        view.getUint8(offset),
        view.getUint8(offset + 1),
        view.getUint8(offset + 2),
        view.getUint8(offset + 3)
    );
}

function uint32 (view: DataView, offset: number): number {
    return view.getUint32(offset * 4, true);
}

function bitVal (data: Uint8Array, bitCount: number, index: number): number {
    // only 1, 4 or 8 bits
    const byte = data[Math.floor(index * bitCount / 8)],
        valsPerByte = 8 / bitCount;

    return (byte >> (valsPerByte - index % valsPerByte - 1)) & ((1 << bitCount) - 1);
}

// BLP palettes are BGRA. Packing them into a Uint32Array once lets the per-pixel loop below be a
// single 32-bit store per pixel instead of a DataView read plus three byte writes.
function buildPaletteLUT (blp: BLPImage): Uint32Array {
    const palette = new Uint8Array(blp.data, 39 * 4, 256 * 4);
    const lut = new Uint32Array(256);
    const bytes = new Uint8Array(lut.buffer);

    for (let i = 0; i < 256; ++i) {
        bytes[i * 4]     = palette[i * 4 + 2];
        bytes[i * 4 + 1] = palette[i * 4 + 1];
        bytes[i * 4 + 2] = palette[i * 4];
        bytes[i * 4 + 3] = 255;
    }

    return lut;
}

interface ImageDataLike {
    width: number;
    height: number;
    data: ImageDataArray;
    colorSpace: 'srgb' | 'display-p3' | undefined;
}

// node.js have no native ImageData
function createImageData (width: number, height: number): ImageDataLike {
    if (typeof ImageData !== 'undefined') {
        return new ImageData(width, height);
    } else {
        return {
            width,
            height,
            data: new Uint8ClampedArray(width * height * 4),
            colorSpace: 'srgb'
        };
    }
}

export function decode (arrayBuffer: ArrayBuffer): BLPImage {
    const view = new DataView(arrayBuffer);

    const image: BLPImage = {
        type: BLPType.BLP1,
        width: 0,
        height: 0,
        content: BLPContent.JPEG,
        alphaBits: 0,
        mipmaps: [],
        data: arrayBuffer,
    };

    const type = keyword(view, 0);

    if (type === 'BLP0' || type === 'BLP2') {
        throw new Error('BLP0/BLP2 not supported');
    }
    if (type !== 'BLP1') {
        throw new Error('Not a blp image');
    }

    image.content = uint32(view, 1);

    if (image.content !== BLPContent.JPEG && image.content !== BLPContent.Direct) {
        throw new Error('Unknown BLP content');
    }

    image.alphaBits = uint32(view, 2);
    image.width = uint32(view, 3);
    image.height = uint32(view, 4);

    for (let i = 0; i < 16; ++i) {
        const mipmap = {
            offset: uint32(view, 7 + i),
            size: uint32(view, 7 + 16 + i)
        };

        if (mipmap.size > 0) {
            image.mipmaps.push(mipmap);
        } else {
            break;
        }
    }

    return image;
}

export function getImageData (blp: BLPImage, mipmapLevel: number): ImageDataLike {
    const view = new DataView(blp.data),
        uint8Data = new Uint8Array(blp.data),
        mipmap = blp.mipmaps[mipmapLevel];

    if (blp.content === BLPContent.JPEG) {
        const headerSize = uint32(view, 39),
            data = new Uint8Array(headerSize + mipmap.size);

        data.set(uint8Data.subarray(40 * 4, 40 * 4 + headerSize));
        data.set(uint8Data.subarray(mipmap.offset, mipmap.offset + mipmap.size), headerSize);

        return decodeJPEG(data);
    } else {
        // Clamped shift, not a divide: a BLP mip chain stops each axis at 1, so a 32x16 image
        // ends 4x2, 2x1, 1x1. Dividing gave a fractional height for the deepest levels of any
        // non-square texture, and a fractional pixel count with it.
        const width = Math.max(1, blp.width >> mipmapLevel),
            height = Math.max(1, blp.height >> mipmapLevel),
            size = width * height,
            indices = new Uint8Array(blp.data, mipmap.offset, size),
            imageData = createImageData(width, height),
            out = imageData.data;

        // One 32-bit store per pixel through the palette LUT, rather than a DataView.getUint8 for
        // the index plus three separate byte writes for BGR. The LUT already carries alpha 255,
        // which covers the no-alpha case with no extra work.
        const lut = buildPaletteLUT(blp);
        const out32 = new Uint32Array(out.buffer, out.byteOffset, size);

        for (let i = 0; i < size; ++i) {
            out32[i] = lut[indices[i]];
        }

        if (blp.alphaBits > 0) {
            const alphaData = new Uint8Array(blp.data, mipmap.offset + size, Math.ceil(size * blp.alphaBits / 8));

            if (blp.alphaBits === 8) {
                // The overwhelmingly common case: alpha is already one byte per pixel.
                for (let i = 0; i < size; ++i) {
                    out[i * 4 + 3] = alphaData[i];
                }
            } else {
                const valPerAlphaBit = 255 / ((1 << blp.alphaBits) - 1);

                for (let i = 0; i < size; ++i) {
                    out[i * 4 + 3] = bitVal(alphaData, blp.alphaBits, i) * valPerAlphaBit;
                }
            }
        }

        return imageData;
    }
}
