import assert from 'node:assert/strict';
import {mat4} from 'gl-matrix';
import {parseMDL, parseMDX, generateMDX, ModelRenderer} from '../dist/es/war3-model.mjs';

const source = `Version { FormatVersion 800, }
Model "Geoset test" { BlendTime 150, }
Sequences 2 {
    Anim "Stand" { Interval { 0, 1000 }, }
    Anim "Other" { Interval { 2000, 3000 }, }
}
GeosetAnim {
    Alpha 2 { Linear, 0: 1, 1000: 0, }
    Color 2 { Linear, 0: { 1, 0, 0 }, 1000: { 0, 0, 1 }, }
    GeosetId 0,
}
GeosetAnim { static Alpha 0.75, static Color { 0, 1, 0 }, GeosetId 1, }
`;

function fixture(hd = false) {
    const model = parseMDL(source);
    for (const extent of [model.Info, ...model.Sequences]) {
        extent.MinimumExtent = new Float32Array(3);
        extent.MaximumExtent = new Float32Array(3);
    }
    model.Textures = [{Image: 'test.blp', Flags: 0}];
    model.Materials = [{Layers: [{FilterMode: 2, Shading: 0, TextureID: 0, Alpha: 0.5,
        ...(hd ? {ShaderTypeId: 1, NormalTextureID: 0, ORMTextureID: 0} : {})}]}];
    model.Geosets = Array.from({length: 3}, () => ({
        MaterialID: 0, SelectionGroup: 0, Unselectable: false,
        Vertices: new Float32Array(9), Normals: new Float32Array(9),
        TVertices: [new Float32Array(6)], VertexGroup: new Uint8Array(3),
        Faces: new Uint16Array([0, 1, 2]), Groups: [], TotalGroupsCount: 0, Anims: [],
        MinimumExtent: new Float32Array(3), MaximumExtent: new Float32Array(3), BoundsRadius: 0,
        ...(hd ? {SkinWeights: new Uint8Array(24), Tangents: new Float32Array(12)} : {})
    }));
    return model;
}

// Hand-authored MDX GEOA bytes assert the wire/API RGB contract independently of generators.
{
    const bytes = new ArrayBuffer(40);
    const view = new DataView(bytes);
    const ascii = (offset, text) => [...text].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
    ascii(0, 'MDLX'); ascii(4, 'GEOA');
    view.setUint32(8, 28, true); view.setUint32(12, 28, true);
    view.setFloat32(16, 1, true); view.setUint32(20, 2, true);
    view.setFloat32(24, 1, true); // RGB red, not MDL BGR blue.
    view.setUint32(36, 2, true); // Sparse GeosetId: this is the third geoset.
    const original = fixture();
    original.GeosetAnims = parseMDX(bytes).GeosetAnims;
    const renderer = new ModelRenderer(original);
    assert.deepEqual([...renderer.rendererData.geosetColor[0]], [1, 1, 1]);
    assert.deepEqual([...renderer.rendererData.geosetColor[2]], [1, 0, 0]);
}

for (const roundTrip of [false, true]) {
    const original = fixture();
    assert.equal(original.GeosetAnims[0].Flags & 2, 2, 'MDL enables animated color');
    assert.equal(original.GeosetAnims[1].Flags & 2, 2, 'MDL enables static color');
    const model = roundTrip ? parseMDX(generateMDX(original)) : original;
    const renderer = new ModelRenderer(model);
    assert.deepEqual([...renderer.rendererData.geosetColor[0]], [0, 0, 1]);
    renderer.update(250);
    assert.equal(renderer.rendererData.geosetAlpha[0], 0.75);
    assert.deepEqual([...renderer.rendererData.geosetColor[0]], [0.25, 0, 0.75]);
    assert.deepEqual([...renderer.rendererData.geosetColor[1]], [0, 1, 0]);
    assert.deepEqual([...renderer.rendererData.geosetColor[2]], [1, 1, 1]);
    assert.equal(renderer.rendererData.geosetAlpha[2], 1);
    renderer.setSequence(1);
    renderer.update(0);
    assert.deepEqual([...renderer.rendererData.geosetColor[0]], [1, 1, 1], 'missing tracks reset tint');
    assert.equal(renderer.rendererData.geosetAlpha[0], 1);
    model.GeosetAnims[1].Flags = 0;
    renderer.update(0);
    assert.deepEqual([...renderer.rendererData.geosetColor[1]], [1, 1, 1], 'disabled color stays white');
}

for (const stepped of [false, true]) {
    const model = fixture();
    model.GlobalSequences = [1000];
    for (const track of [model.GeosetAnims[0].Color, model.GeosetAnims[0].Alpha]) {
        track.GlobalSeqId = 0;
        if (stepped) track.LineType = 0;
    }
    const renderer = new ModelRenderer(model);
    renderer.setSequence(1);
    renderer.update(250);
    assert.equal(renderer.rendererData.geosetAlpha[0], stepped ? 1 : 0.75);
    assert.deepEqual([...renderer.rendererData.geosetColor[0]], stepped ? [0, 0, 1] : [0.25, 0, 0.75]);
}

// Capture actual draw-time uniforms: interpolation alone did not detect the original bug.
for (const hd of [false, true]) {
    const renderer = new ModelRenderer(fixture(hd));
    const draws = [];
    let color;
    let layerAlpha;
    renderer.gl = new Proxy({}, {get: (_, name) => {
        if (/^[A-Z_0-9]+$/.test(name)) return 1;
        if (name === 'uniform4f') return (location, ...values) => { if (location === 'geoset') color = values; };
        if (name === 'uniform1f') return (location, value) => { if (location === 'layer') layerAlpha = value; };
        if (name === 'drawElements') return () => draws.push({color: [...color], layerAlpha});
        return () => {};
    }});
    renderer.shaderProgramLocations.geosetColorUniform = 'geoset';
    renderer.shaderProgramLocations.layerAlphaUniform = 'layer';
    renderer.shaderProgramLocations.nodesMatricesAttributes = [null];
    renderer.ensureGeosetBuffers = () => {};
    renderer.useVAO = true;
    renderer.update(250);
    renderer.render(mat4.create(), mat4.create(), {});
    assert.deepEqual(draws, [
        {color: [0.25, 0, 0.75, 0.75], layerAlpha: 0.5},
        {color: [0, 1, 0, 0.75], layerAlpha: 0.5},
        {color: [1, 1, 1, 1], layerAlpha: 0.5},
    ], `${hd ? 'HD' : 'SD'} draws receive independent geoset tint and alpha`);
    draws.length = 0;
    renderer.update(750);
    renderer.render(mat4.create(), mat4.create(), {});
    assert.equal(draws.length, 2, 'zero-alpha geosets are skipped');
}

console.log('ok - geoset colors and alpha reach SD/HD draws and survive MDX round trips');
