import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseMDX, generateMDX } from '../dist/es/war3-model.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

function readModel(fileName) {
    const buffer = fs.readFileSync(path.join(repoRoot, fileName));
    return parseMDX(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
}

function run(name, fn) {
    fn();
    console.log(`ok - ${name}`);
}

// .mdx files are gitignored, so a fresh clone has no fixtures to work with. Skip rather than
// fail: drop any .mdx model into the repository root to exercise these.
function runWithModels(name, fn) {
    if (models.length === 0) {
        console.log(`skip - ${name} (no .mdx fixtures in ${repoRoot})`);
        return;
    }
    run(name, fn);
}

function runWith(fileName, name, fn) {
    if (!fs.existsSync(path.join(repoRoot, fileName))) {
        console.log(`skip - ${name} (${fileName} not present)`);
        return;
    }
    run(name, fn);
}

/** Every AnimVector reachable from a model, in a stable order. */
function collectAnimVectors(model) {
    const found = [];
    const seen = new Set();

    (function walk(value, depth) {
        if (!value || typeof value !== 'object' || depth > 8 || seen.has(value)) {
            return;
        }
        seen.add(value);

        if (Array.isArray(value)) {
            for (const item of value) {
                walk(item, depth + 1);
            }
            return;
        }
        if (value.Frames instanceof Int32Array && 'LineType' in value && 'VectorSize' in value) {
            found.push(value);
            return;
        }
        for (const key in value) {
            walk(value[key], depth + 1);
        }
    })(model, 0);

    return found;
}

const models = fs.readdirSync(repoRoot).filter(name => /\.mdx$/i.test(name));

runWithModels('AnimVector flat storage survives a generateMDX round trip', () => {
    let totalKeyframes = 0;

    for (const name of models) {
        const model = readModel(name);
        const reparsed = parseMDX(generateMDX(model));

        const original = collectAnimVectors(model);
        const actual = collectAnimVectors(reparsed);

        assert.equal(actual.length, original.length, `${name}: AnimVector count`);

        for (let i = 0; i < original.length; ++i) {
            const a = original[i];
            const b = actual[i];

            assert.equal(b.LineType, a.LineType, `${name}: vector ${i} LineType`);
            assert.equal(b.VectorSize, a.VectorSize, `${name}: vector ${i} VectorSize`);
            assert.equal(b.GlobalSeqId, a.GlobalSeqId, `${name}: vector ${i} GlobalSeqId`);
            assert.deepEqual(b.Frames, a.Frames, `${name}: vector ${i} Frames`);
            assert.deepEqual(b.Values, a.Values, `${name}: vector ${i} Values`);
            assert.deepEqual(b.InTans, a.InTans, `${name}: vector ${i} InTans`);
            assert.deepEqual(b.OutTans, a.OutTans, `${name}: vector ${i} OutTans`);

            totalKeyframes += a.Frames.length;
        }
    }

    assert.ok(totalKeyframes > 0, 'no keyframes examined');
});

runWith('BHolyWings.mdx', 'AnimVector.Keys is a faithful view of the flat arrays', () => {
    const model = readModel('BHolyWings.mdx');
    const vector = collectAnimVectors(model).find(it => it.Frames.length > 3);

    assert.ok(vector, 'no AnimVector with more than three keyframes');
    assert.equal(vector.Keys.length, vector.Frames.length);

    for (let i = 0; i < vector.Keys.length; ++i) {
        assert.equal(vector.Keys[i].Frame, vector.Frames[i]);
        assert.equal(vector.Keys[i].Vector.length, vector.VectorSize);

        for (let j = 0; j < vector.VectorSize; ++j) {
            assert.equal(vector.Keys[i].Vector[j], vector.Values[i * vector.VectorSize + j]);
        }
    }

    // Keys are views, so writing through one reaches the flat storage.
    vector.Keys[1].Vector[0] = 123.5;
    assert.equal(vector.Values[vector.VectorSize], 123.5);
});

runWith('BHolyWings.mdx', 'assigning AnimVector.Keys rebuilds the flat storage', () => {
    const model = readModel('BHolyWings.mdx');
    const vector = collectAnimVectors(model).find(it => it.Frames.length > 3);
    const kept = vector.Keys.slice(0, 2).map(key => ({ Frame: key.Frame, Vector: key.Vector.slice() }));

    vector.Keys = kept;

    assert.equal(vector.Frames.length, 2);
    assert.equal(vector.Values.length, 2 * vector.VectorSize);
    assert.equal(vector.Keys.length, 2);
    assert.equal(vector.Keys[0].Frame, kept[0].Frame);
    assert.equal(vector.Keys[1].Vector[0], kept[1].Vector[0]);

    // The rebuilt Keys must be views on the new storage, not the arrays that were assigned.
    vector.Keys[0].Vector[0] = -7.5;
    assert.equal(vector.Values[0], -7.5);
});

console.log('all tests passed');
