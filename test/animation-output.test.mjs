import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseMDX, generateMDX, ModelRenderer } from '../dist/es/war3-model.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

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

function readBuffer(fileName) {
    const buffer = fs.readFileSync(path.join(repoRoot, fileName));
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

/**
 * Run the animation path headlessly and record everything it produces.
 *
 * ModelRenderer.update() drives node hierarchy interpolation and geoset alpha without touching
 * GL, so the whole interpolator can be exercised without a context. rendererData is internal, but
 * reading it here is what lets this test compare interpolation output rather than just types.
 */
function sampleAnimation(model, { sequences = 12, steps = 24, delta = 37 } = {}) {
    const renderer = new ModelRenderer(model);
    const samples = [];
    const count = Math.min(model.Sequences.length, sequences);

    for (let sequence = 0; sequence < count; ++sequence) {
        renderer.setSequence(sequence);

        for (let step = 0; step < steps; ++step) {
            renderer.update(delta);

            for (const node of renderer.rendererData.nodes) {
                if (node) {
                    samples.push(...node.matrix);
                }
            }
            samples.push(...renderer.rendererData.geosetAlpha);
        }
    }

    return samples;
}

const models = fs.readdirSync(repoRoot).filter(name => /\.mdx$/i.test(name));

runWithModels('animation output is deterministic and finite', () => {
    for (const name of models) {
        const buffer = readBuffer(name);
        const first = sampleAnimation(parseMDX(buffer));
        const second = sampleAnimation(parseMDX(buffer));

        assert.ok(first.length > 0, `${name}: produced no samples`);
        assert.equal(second.length, first.length, `${name}: sample count differs between runs`);

        for (let i = 0; i < first.length; ++i) {
            assert.ok(Number.isFinite(first[i]), `${name}: non-finite value at ${i}`);
            assert.equal(second[i], first[i], `${name}: value ${i} differs between runs`);
        }
    }
});

runWithModels('animation output survives a generateMDX round trip unchanged', () => {
    for (const name of models) {
        const model = parseMDX(readBuffer(name));
        const expected = sampleAnimation(model);
        const actual = sampleAnimation(parseMDX(generateMDX(model)));

        assert.equal(actual.length, expected.length, `${name}: sample count`);

        for (let i = 0; i < expected.length; ++i) {
            assert.equal(actual[i], expected[i], `${name}: value ${i}`);
        }
    }
});

runWithModels('non-looping sequences remain on their final frame', () => {
    for (const name of models) {
        const model = parseMDX(readBuffer(name));
        for (let sequence = 0; sequence < model.Sequences.length; ++sequence) {
            const animation = model.Sequences[sequence];
            if (!animation.NonLooping) {
                continue;
            }

            const renderer = new ModelRenderer(model);
            renderer.setSequence(sequence);
            renderer.update(animation.Interval[1] - animation.Interval[0] + 1);
            assert.equal(renderer.getFrame(), animation.Interval[1], `${name}: sequence ${sequence} did not stop`);
            renderer.update(1000);
            assert.equal(renderer.getFrame(), animation.Interval[1], `${name}: sequence ${sequence} looped`);
        }
    }
});

console.log('all tests passed');
