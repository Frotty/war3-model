import assert from 'node:assert/strict';
import {mat4, vec3} from 'gl-matrix';
import {parseMDL, ModelRenderer, fitThumbnailCamera} from '../dist/es/war3-model.mjs';

function fixture() {
    const model = parseMDL(`Version { FormatVersion 800, }
        Model "Display test" { BlendTime 150, }
        Sequences 1 { Anim "Stand" { Interval { 0, 10000 }, } }
        Bone "Root" { ObjectId 0, GeosetId Multiple, GeosetAnimId None, }
        PivotPoints 1 { { 0, 0, 0 }, }
    `);
    model.Textures = [{Image: 'test.blp', Flags: 0}];
    model.Materials = [{PriorityPlane: 0, Layers: [{FilterMode: 2, Shading: 0, TextureID: 0, Alpha: 1}]}];
    model.Geosets = [{MaterialID: 0, Vertices: new Float32Array([
        -1,-1,-2, 1,-1,-2, 0,1,-2, -1,-1,-8, 1,-1,-8, 0,1,-8, 9000,9000,9000,
    ]), Normals: new Float32Array(21), TVertices: [new Float32Array(14), new Float32Array(14).fill(0.7)],
    VertexGroup: new Uint8Array(7), Groups: [[0]], Faces: new Uint16Array([0,1,2,3,4,5]), Anims: []}];
    return model;
}

function mockGL(renderer, draws = []) {
    let bound;
    const uploads = [];
    renderer.gl = new Proxy({}, {get: (_, name) => {
        if (/^[A-Z_0-9]+$/.test(name)) return name;
        if (name === 'isContextLost') return () => false;
        if (name === 'fenceSync') return undefined;
        if (name === 'createBuffer') return () => ({});
        if (name === 'deleteBuffer') return buffer => uploads.push({target: 'DELETE', bound: buffer});
        if (name === 'bindTexture') return (_target, texture) => uploads.push({target: 'TEXTURE', bound: texture});
        if (name === 'bindBuffer') return (_target, buffer) => {bound = buffer;};
        if (name === 'bufferData') return (target, data) => uploads.push({target, bound, data: [...data]});
        if (name === 'drawElements') return () => draws.push('mesh');
        return () => {};
    }});
    renderer.shaderProgramLocations.nodesMatricesAttributes = [null];
    renderer.ensureGeosetBuffers = () => {};
    renderer.useVAO = true;
    return uploads;
}

const track = values => ({Frames: new Uint32Array([0]), Values: new Float32Array(values),
    VectorSize: values.length, LineType: 0, GlobalSeqId: null});
for (const [flags, translation, scale] of [[0,10,2], [1,0,2], [4,10,1], [5,0,1]]) {
    const model = fixture();
    model.Nodes[0].Translation = track([10,0,0]);
    model.Nodes[0].Scaling = track([2,2,2]);
    model.Nodes.push({ObjectId: 1, Parent: 0, Flags: flags, PivotPoint: new Float32Array(3)});
    const renderer = new ModelRenderer(model);
    renderer.update(0);
    const matrix = renderer.rendererData.nodes[1].matrix;
    assert.equal(matrix[12], translation, 'translation inheritance flag');
    assert.equal(matrix[0], scale, 'scale inheritance flag');
}
{
    const model = fixture();
    model.Nodes[0].Rotation = track([0,0,Math.SQRT1_2,Math.SQRT1_2]);
    model.Nodes.push({ObjectId: 1, Parent: 0, Flags: 2, PivotPoint: new Float32Array(3)});
    const renderer = new ModelRenderer(model);
    renderer.update(0);
    assert.ok(Math.abs(renderer.rendererData.nodes[1].matrix[0] - 1) < 1e-6, 'rotation inheritance flag');
    model.Nodes[1].Flags = 128;
    renderer.setCamera([3,4,5], [0,0,0,1]);
    renderer.update(0);
    assert.deepEqual([...renderer.rendererData.nodes[1].matrix.slice(12,15)], [3,4,5], 'camera anchored node follows camera position');
}
{
    const model = fixture();
    const light = {ObjectId: 1, Flags: 0, PivotPoint: new Float32Array([1,2,3]), LightType: 0,
        Color: new Float32Array([0,0,1]), Intensity: 2, AmbColor: new Float32Array([0,1,0]), AmbIntensity: 1,
        AttenuationStart: 10, AttenuationEnd: 100};
    model.Nodes.push(light);
    model.Lights = [light];
    const renderer = new ModelRenderer(model);
    renderer.update(0);
    assert.equal(renderer.lighting.count, 1);
    assert.ok(Math.abs(renderer.lighting.colors[0] - 0.4) < 1e-6, 'light color converts BGR and scales intensity');
    assert.deepEqual([...renderer.lighting.positions.slice(0,4)], [1,2,3,1]);
    assert.ok(Math.abs(renderer.lighting.ambient[1] - 1) < 1e-6, 'model ambient contributes to bright defaults');
    light.Visibility = 0;
    renderer.update(0);
    assert.equal(renderer.lighting.count, 0, 'hidden model light is ignored');
}
{
    const model = fixture();
    model.Nodes[0].Rotation = track([0,0,Math.SQRT1_2,Math.SQRT1_2]);
    model.Nodes[0].Scaling = track([2,3,4]);
    model.Geosets[0].Normals.set([0,1,0]);
    model.Materials[0].Layers.push({...model.Materials[0].Layers[0]});
    const renderer = new ModelRenderer(model);
    mockGL(renderer);
    renderer.softwareSkinning = true;
    renderer.vertices[0] = new Float32Array(model.Geosets[0].Vertices.length);
    const generate = renderer.generateGeosetVertices.bind(renderer);
    let skins = 0;
    renderer.generateGeosetVertices = index => {skins++; generate(index);};
    renderer.update(0);
    renderer.render(mat4.create(), mat4.create(), {});
    assert.equal(skins, 1, 'multiple layers skin their shared geometry once');
    assert.ok(Math.abs(renderer.vertices[0][0] - 3) < 1e-5);
    assert.ok(Math.abs(renderer.vertices[0][1] + 2) < 1e-5);
    assert.ok(Math.abs(renderer.posedNormals[0][0] + 1) < 1e-5, 'software skinning transforms normals');
}

{
    const model = fixture();
    model.Materials[0].RenderMode = 16;
    model.Materials[0].Layers[0].CoordId = 1;
    const renderer = new ModelRenderer(model);
    const uploads = mockGL(renderer);
    renderer.update(0);
    renderer.render(mat4.create(), mat4.create(), {});
    assert.deepEqual(uploads.find(u => u.target === 'ELEMENT_ARRAY_BUFFER').data, [3,4,5,0,1,2], 'far triangles draw first');
    assert.deepEqual(uploads.find(u => u.target === 'ARRAY_BUFFER').data, [...model.Geosets[0].TVertices[1]], 'CoordId selects the second UV set');
    uploads.length = 0;
    const reversed = mat4.fromScaling(mat4.create(), [1,1,-1]);
    renderer.render(reversed, mat4.create(), {});
    assert.deepEqual(uploads.find(u => u.target === 'ELEMENT_ARRAY_BUFFER').data, [0,1,2,3,4,5], 'sorting follows camera depth');
}

{
    const model = fixture();
    model.Materials.push({PriorityPlane: 99, Layers: [{FilterMode: 0, TextureID: 0, Alpha: 1}]});
    model.Geosets.push({...model.Geosets[0], MaterialID: 1});
    model.ParticleEmitters2 = [{PriorityPlane: -2, EmissionRate: 0, LifeSpan: 1}];
    model.RibbonEmitters = [{MaterialID: 0, EmissionRate: 0, LifeSpan: 1}];
    const renderer = new ModelRenderer(model);
    assert.deepEqual(renderer.drawOrder.map(b => [b.kind, b.index]), [['geoset',1], ['particle',0], ['geoset',0], ['ribbon',0]], 'opaque first, then shared translucent priorities');
    const draws = [];
    mockGL(renderer, draws);
    renderer.particlesController.render = () => draws.push('particle');
    renderer.ribbonsController.render = () => draws.push('ribbon');
    renderer.render(mat4.create(), mat4.create(), {});
    assert.deepEqual(draws, ['mesh','particle','mesh','ribbon'], 'draw order interleaves effects without duplicate draws');
}

{
    const model = fixture();
    model.Materials[0].Layers.push({FilterMode: 0, TextureID: 0, Alpha: 1});
    model.Materials.push({PriorityPlane: 99, Layers: [{FilterMode: 0, TextureID: 0, Alpha: 1}]});
    model.Geosets.push({...model.Geosets[0], MaterialID: 1});
    model.ParticleEmitters2 = [{PriorityPlane: -2, EmissionRate: 0, LifeSpan: 1}];
    const renderer = new ModelRenderer(model);
    assert.deepEqual(renderer.drawOrder.map(b => [b.kind, b.index, b.layer]),
        [['geoset',1,0], ['particle',0,0], ['geoset',0,0], ['geoset',0,1]],
        'mixed-filter materials keep authored passes adjacent and ordered');
}

{
    const model = fixture();
    model.GlobalSequences = [300];
    const renderer = new ModelRenderer(model);
    renderer.setPose(0, 501);
    assert.equal(renderer.getFrame(), 501);
    assert.equal(renderer.rendererData.globalSequencesFrames[0], 201, 'global clocks preserve overshoot');
    renderer.update(0);
    assert.equal(renderer.getFrame(), 501, 'frozen updates preserve the pose');
    renderer.setPose(0, 1);
    assert.equal(renderer.rendererData.globalSequencesFrames[0], 1, 'setting a pose resets global clocks');
    const bounds = renderer.getVisibleBounds();
    assert.deepEqual([...bounds.minimum], [-1,-1,-8]);
    assert.deepEqual([...bounds.maximum], [1,1,-2], 'unused outlying vertices do not affect bounds');
    const camera = fitThumbnailCamera(bounds, 300, 100, [0,0,1]);
    assert.throws(() => fitThumbnailCamera({minimum: [0,0,0], maximum: [Infinity,1,1]}, 100, 100), /Invalid/);
    const vp = mat4.mul(mat4.create(), camera.projection, camera.view);
    for (const point of [bounds.minimum, bounds.maximum]) {
        const projected = vec3.transformMat4(vec3.create(), point, vp);
        assert.ok(Math.abs(projected[0]) <= 0.85 && Math.abs(projected[1]) <= 0.85);
    }
    renderer.rendererData.geosetAlpha[0] = 0;
    assert.equal(renderer.getVisibleBounds(), null);
    renderer.rendererData.geosetAlpha[0] = 1;
    model.Geosets[0].LevelOfDetail = 1;
    assert.equal(renderer.getVisibleBounds(), null);
    assert.ok(renderer.getVisibleBounds({levelOfDetail: 1}));
}

function ribbonFixture() {
    const model = fixture();
    const emitter = {...model.Nodes[0], EmissionRate: 10, LifeSpan: 10, Gravity: 20,
        HeightAbove: 1, HeightBelow: 1, MaterialID: 0, Rows: 2, Columns: 3, TextureSlot: 4};
    model.RibbonEmitters = [emitter];
    return new ModelRenderer(model);
}
for (const animatedTexture of [false, true]) {
    const model = fixture();
    model.Textures.unshift({Image: '', ReplaceableId: 1, Flags: 0});
    const textureID = animatedTexture ? {Frames: new Uint32Array([0]), Values: new Uint32Array([0]),
        VectorSize: 1, LineType: 0, GlobalSeqId: null} : 0;
    model.Materials[0].Layers = [{FilterMode: 2, Shading: 0, TextureID: textureID, Alpha: 0},
        {FilterMode: 2, Shading: 0, TextureID: 1, Alpha: 1}];
    const renderer = new ModelRenderer(model);
    renderer.setPose(0, 0);
    const draws = [];
    mockGL(renderer, draws);
    renderer.render(mat4.create(), mat4.create(), {});
    assert.deepEqual(draws, [], 'mask layers are consumed rather than drawn independently');
    assert.equal(renderer.getVisibleBounds(), null, 'bounds follow mask exclusions from actual draws');
    model.Materials[0].Layers[0].Alpha = 1;
    assert.ok(renderer.getVisibleBounds(), 'visible replaceable passes still contribute bounds');
}
{
    const model = fixture();
    model.ParticleEmitters2 = [{EmissionRate: 0, LifeSpan: 1, TextureID: 0, FilterMode: 0}];
    const renderer = new ModelRenderer(model);
    const uploads = mockGL(renderer);
    const fallback = renderer.rendererData.whiteTexture = {};
    const particles = renderer.particlesController;
    particles.gl = renderer.gl;
    particles.setLayerProps(particles.emitters[0]);
    const ribbons = ribbonFixture().ribbonsController;
    ribbons.gl = renderer.gl;
    ribbons.rendererData.whiteTexture = fallback;
    ribbons.setLayerProps(model.Materials[0].Layers[0], 0);
    assert.equal(uploads.filter(upload => upload.target === 'TEXTURE' && upload.bound === fallback).length, 2,
        'both effect controllers bind the shared fallback when an image is missing');
}
{
    const renderer = ribbonFixture();
    const uploads = mockGL(renderer);
    const controller = renderer.ribbonsController;
    controller.gl = renderer.gl;
    renderer.rendererData.model.Geosets = [];
    renderer.update(100);
    const emitter = controller.emitters[0];
    assert.equal(renderer.getVisibleBounds(), null, 'a single ribbon point does not draw');
    const buffers = [emitter.vertexBuffer, emitter.texCoordBuffer];
    renderer.update(100);
    assert.ok(renderer.getVisibleBounds(), 'a live ribbon contributes to visible bounds');
    emitter.props.Alpha = 0;
    assert.equal(renderer.getVisibleBounds(), null, 'hidden ribbons do not distort framing');
    emitter.props.Alpha = 1;
    renderer.update(0);
    controller.render(mat4.create(), mat4.create());
    assert.ok(!uploads.some(upload => upload.target === 'DELETE' && buffers.includes(upload.bound)), 'ribbon buffers survive subsequent updates and draws');
    controller.destroy();
    for (const buffer of buffers) assert.ok(uploads.some(upload => upload.target === 'DELETE' && upload.bound === buffer), 'ribbon destroy releases GL buffers');
    const gpu = ribbonFixture().ribbonsController;
    let destroyed = 0;
    gpu.emitters[0].vertexGPUBuffer = {destroy: () => destroyed++};
    gpu.emitters[0].texCoordGPUBuffer = {destroy: () => destroyed++};
    gpu.update(0);
    assert.equal(destroyed, 0, 'GPU ribbon updates keep their buffers alive');
    gpu.destroy();
    assert.equal(destroyed, 2, 'GPU ribbon teardown owns destruction');
}
{
    const renderer = ribbonFixture();
    const controller = renderer.ribbonsController;
    renderer.update(100);
    const emitter = controller.emitters[0];
    assert.equal(emitter.creationTimes.length, 1, 'missing visibility defaults to visible');
    assert.equal(emitter.texCoords[1], 0.5, 'atlas row uses column count');
    emitter.props.EmissionRate = 0;
    renderer.update(1000);
    assert.equal(emitter.vertices[2], -10, 'gravity falls along world Z');
    const vertices = [...emitter.vertices];
    renderer.update(0);
    assert.deepEqual([...emitter.vertices], vertices, 'paused trails do not fall');
    const split = ribbonFixture();
    split.update(100);
    split.ribbonsController.emitters[0].props.EmissionRate = 0;
    for (let i = 0; i < 10; ++i) split.update(100);
    assert.ok(Math.abs(split.ribbonsController.emitters[0].vertices[2] + 10) < 1e-5, 'fall does not depend on timestep');
    renderer.setPose(0, 0);
    assert.equal(emitter.creationTimes.length, 0, 'setting a pose clears trails');
}

{
    const renderer = new ModelRenderer(fixture());
    mockGL(renderer);
    await assert.rejects(renderer.whenReady({timeoutMs: 5}), /timed out/);
    assert.equal(renderer.textureListeners.size, 0, 'timed-out listeners are removed');
    const abort = new globalThis.AbortController();
    const pending = renderer.whenReady({signal: abort.signal});
    abort.abort();
    await assert.rejects(pending, {name: 'AbortError'});
    const ready = renderer.whenReady();
    renderer.adoptTexture('test.blp', {});
    await ready;
    const rendered = renderer.whenRendered();
    renderer.render(mat4.create(), mat4.create(), {});
    await rendered;
    const fresh = new ModelRenderer(fixture());
    mockGL(fresh);
    const waiting = fresh.whenReady();
    fresh.destroy();
    await assert.rejects(waiting, /destroyed/);
}

console.log('Renderer display, sorting, bounds, ribbons and readiness checks passed.');
