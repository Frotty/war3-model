import assert from 'node:assert/strict';

import { decodeBLP, getBLPImageData } from '../dist/es/war3-model.mjs';

function run(name, fn) {
    fn();
    console.log(`ok - ${name}`);
}

const HEADER_SIZE = 39 * 4;
const PALETTE_SIZE = 256 * 4;
const DATA_START = HEADER_SIZE + PALETTE_SIZE;

/** A BLP1 with Direct (palettized) content and a full mip chain, filled with reproducible noise. */
function buildDirectBLP(width, height, alphaBits) {
    const mipmaps = [];
    let offset = DATA_START;
    let w = width;
    let h = height;

    while (mipmaps.length < 16) {
        const size = w * h + Math.ceil(w * h * alphaBits / 8);
        mipmaps.push({ offset, size });
        offset += size;
        if (w === 1 && h === 1) {
            break;
        }
        w = Math.max(1, w >> 1);
        h = Math.max(1, h >> 1);
    }

    const buffer = new ArrayBuffer(offset);
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);

    bytes.set([0x42, 0x4C, 0x50, 0x31]); // 'BLP1'
    view.setUint32(4, 1, true); // Direct content
    view.setUint32(8, alphaBits, true);
    view.setUint32(12, width, true);
    view.setUint32(16, height, true);
    mipmaps.forEach((mipmap, i) => {
        view.setUint32((7 + i) * 4, mipmap.offset, true);
        view.setUint32((7 + 16 + i) * 4, mipmap.size, true);
    });

    for (let i = 0; i < PALETTE_SIZE; ++i) {
        bytes[HEADER_SIZE + i] = (i * 97 + 11) & 255;
    }
    for (let i = DATA_START; i < offset; ++i) {
        bytes[i] = (i * 181 + 7) & 255;
    }

    return buffer;
}

function bitVal(data, bitCount, index) {
    const byte = data[Math.floor(index * bitCount / 8)];
    const valsPerByte = 8 / bitCount;

    return (byte >> (valsPerByte - index % valsPerByte - 1)) & ((1 << bitCount) - 1);
}

/** Straightforward per-pixel reference decode, independent of the shipped implementation. */
function referenceDecode(blp, level) {
    const view = new DataView(blp.data);
    const mipmap = blp.mipmaps[level];
    const palette = new Uint8Array(blp.data, HEADER_SIZE, PALETTE_SIZE);
    const width = Math.max(1, blp.width >> level);
    const height = Math.max(1, blp.height >> level);
    const size = width * height;
    const alphaData = new Uint8Array(blp.data, mipmap.offset + size, Math.ceil(size * blp.alphaBits / 8));
    const data = new Uint8ClampedArray(size * 4);
    const valPerAlphaBit = 255 / ((1 << blp.alphaBits) - 1);

    for (let i = 0; i < size; ++i) {
        const paletteIndex = view.getUint8(mipmap.offset + i) * 4;

        data[i * 4] = palette[paletteIndex + 2];
        data[i * 4 + 1] = palette[paletteIndex + 1];
        data[i * 4 + 2] = palette[paletteIndex];
        data[i * 4 + 3] = blp.alphaBits > 0 ? bitVal(alphaData, blp.alphaBits, i) * valPerAlphaBit : 255;
    }

    return { width, height, data };
}

run('palettized BLP decode matches a per-pixel reference at every alpha depth', () => {
    let levelsChecked = 0;

    for (const alphaBits of [0, 1, 4, 8]) {
        for (const [width, height] of [[64, 64], [32, 16], [8, 32], [8, 8]]) {
            const blp = decodeBLP(buildDirectBLP(width, height, alphaBits));

            for (let level = 0; level < blp.mipmaps.length; ++level) {
                const expected = referenceDecode(blp, level);
                const actual = getBLPImageData(blp, level);
                const label = `alphaBits=${alphaBits} ${width}x${height} level=${level}`;

                assert.equal(actual.width, expected.width, `${label}: width`);
                assert.equal(actual.height, expected.height, `${label}: height`);
                assert.deepEqual(actual.data, expected.data, `${label}: pixels`);

                levelsChecked++;
            }
        }
    }

    assert.ok(levelsChecked > 0, 'no mip levels examined');
});

run('mip level dimensions clamp at 1 for non-square textures', () => {
    const blp = decodeBLP(buildDirectBLP(32, 16, 8));
    const deepest = getBLPImageData(blp, blp.mipmaps.length - 1);

    assert.equal(deepest.width, 1);
    assert.equal(deepest.height, 1);
    assert.equal(deepest.data.length, 4);

    // A 32x16 image ends 4x2, 2x1, 1x1 — never a fractional height.
    for (let level = 0; level < blp.mipmaps.length; ++level) {
        const image = getBLPImageData(blp, level);

        assert.ok(Number.isInteger(image.width), `level ${level} width ${image.width}`);
        assert.ok(Number.isInteger(image.height), `level ${level} height ${image.height}`);
        assert.ok(image.width >= 1 && image.height >= 1, `level ${level} degenerate`);
    }
});

console.log('all tests passed');
