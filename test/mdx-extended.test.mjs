import assert from 'node:assert/strict';
import {Buffer} from 'node:buffer';
import {parseMDX, generateMDX, parseMDL, generateMDL} from '../dist/es/war3-model.mjs';

function chunk(tag, data) {
    const header = Buffer.alloc(8);
    header.write(tag); header.writeUInt32LE(data.length, 4);
    return Buffer.concat([header, data]);
}
function track(tag, value) {
    const bytes = Buffer.alloc(24);
    bytes.write(tag); bytes.writeUInt32LE(1, 4); bytes.writeInt32LE(-1, 12);
    bytes.writeUInt32LE(123, 16); bytes.writeFloatLE(value, 20);
    return bytes;
}
function model(version, ...chunks) {
    const vers = Buffer.alloc(4); vers.writeUInt32LE(version);
    return Buffer.concat([Buffer.from('MDLX'), chunk('VERS', vers), chunk('MODL', Buffer.alloc(372)), ...chunks]);
}
const parse = bytes => parseMDX(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

// Independent binary fixture: v1800's SKIN count is an element count, while the
// payload contains uint16 bone IDs and weights. UVAS must follow all 16 bytes.
function skinGeoset(version) {
    const uint32 = (...values) => {
        const bytes = Buffer.alloc(values.length * 4);
        values.forEach((value, index) => bytes.writeUInt32LE(value, index * 4));
        return bytes;
    };
    const skin = [version >= 1800 ? 300 : 3, 2, 1, 0, 122, 107, 26, 0];
    const array = (tag, count, bytes) => Buffer.concat([Buffer.from(tag), uint32(count), bytes]);
    const skinBytes = Buffer.alloc(version >= 1800 ? 16 : 8);
    skin.forEach((value, index) => version >= 1800 ? skinBytes.writeUInt16LE(value, index * 2) : skinBytes.writeUInt8(value, index));
    const payload = Buffer.concat([
        array('VRTX', 1, Buffer.alloc(12)), array('NRMS', 1, Buffer.alloc(12)),
        array('PTYP', 1, uint32(4)), array('PCNT', 1, uint32(3)), array('PVTX', 3, Buffer.alloc(6)),
        array('GNDX', 0, Buffer.alloc(0)), array('MTGC', 1, uint32(1)), array('MATS', 1, uint32(0)),
        uint32(0, 0, 0, 0), Buffer.alloc(80), Buffer.alloc(28), uint32(0),
        // chunk() normally writes a byte length; SKIN intentionally writes 8 here.
        Buffer.from('SKIN'), uint32(8), skinBytes,
        Buffer.from('UVAS'), uint32(1), array('UVBS', 1, Buffer.alloc(8)),
    ]);
    return {bytes: model(version, chunk('GEOS', Buffer.concat([uint32(payload.length + 4), payload]))), skin};
}
for (const version of [1200, 1800]) {
    const fixture = skinGeoset(version);
    const parsed = parse(fixture.bytes);
    assert.deepEqual([...parsed.Geosets[0].SkinWeights], fixture.skin, `v${version} skin elements`);
    assert.equal(parsed.Geosets[0].SkinWeights.BYTES_PER_ELEMENT, version >= 1800 ? 2 : 1);
    assert.equal(parsed.Geosets[0].TVertices[0].length, 2, 'UVAS starts after the entire skin payload');
    assert.deepEqual([...parseMDX(generateMDX(parsed)).Geosets[0].SkinWeights], fixture.skin, 'MDX round trip preserves wide bone IDs');
    assert.deepEqual([...parseMDL(generateMDL(parsed)).Geosets[0].SkinWeights], fixture.skin, 'MDL round trip preserves wide bone IDs');
    console.log(`ok - v${version} skin storage and UV alignment survive MDX/MDL round trips`);
}

const light = Buffer.alloc(172);
light.writeUInt32LE(light.length + 24, 0); light.writeUInt32LE(96, 4);
light.write('Modern light', 8); light.writeInt32LE(-1, 92); light.writeUInt32LE(512, 96);
light.writeUInt32LE(7, 104);
for (const [offset, value] of [[108, 80], [112, 200], [116, 1], [120, 0.5], [124, 0.25], [128, 18],
    [132, 0.1], [136, 0.2], [140, 0.3], [144, 2], [148, 0.4], [152, 3], [156, 4], [160, 0.0005], [164, 6], [168, 0.00001]]) light.writeFloatLE(value, offset);
const camera = Buffer.alloc(120);
camera.writeUInt32LE((0x83000000 | (120 + 3 * 24)) >>> 0, 0); camera.write('Modern camera', 4);
function checkExtended(parsed) {
    assert.equal(parsed.Lights[0].ReforgedFlags, 7);
    assert.equal(parsed.Lights[0].AttenuationStart, 80);
    assert.equal(parsed.Lights[0].AttenuationEnd, 200);
    assert.deepEqual([...parsed.Lights[0].Color], [1, 0.5, 0.25]);
    assert.equal(parsed.Lights[0].Intensity, 18);
    assert.equal(parsed.Lights[0].ShadowIntensity, Math.fround(0.4));
    assert.deepEqual([...parsed.Lights[0].ReforgedData], [3, 4, Math.fround(0.0005), 6, Math.fround(0.00001)]);
    assert.equal(parsed.Lights[0].Visibility.Values[0], 0.75);
    assert.equal(parsed.Cameras[0].Flags, 0x83);
    assert.equal(parsed.Cameras[0].AdditionalTracks.IDUF.Values[0], 10);
    assert.equal(parsed.Cameras[0].AdditionalTracks.ELAF.Values[0], 8);
    assert.equal(parsed.Cameras[0].AdditionalTracks.PTSF.Values[0], 35);
    assert.equal(parsed.Lights[0].Visibility.Frames[0], 123);
}

for (const version of [800, 1000, 1100, 1200]) {
    const staticLight = Buffer.concat([light.subarray(0, 104), light.subarray(108, version >= 1200 ? 152 : 148)]);
    const start = track('KLAS', 80.25);
    const end = track('KLAE', 200.5);
    staticLight.writeUInt32LE(staticLight.length + 72, 0);
    const bytes = model(version, chunk('LITE', Buffer.concat([staticLight, track('KLAV', 0.75), start, end])));
    for (const parsed of [parse(bytes), parseMDX(generateMDX(parse(bytes)))]) {
        const parsedLight = parsed.Lights[0];
        assert.equal(parsedLight.ReforgedFlags, undefined);
        assert.equal(parsedLight.ReforgedData, undefined);
        assert.deepEqual([...parsedLight.Color], [1, 0.5, 0.25]);
        assert.equal(parsedLight.ShadowIntensity, version >= 1200 ? Math.fround(0.4) : undefined);
        assert.equal(parsedLight.Visibility.Values[0], 0.75);
        assert.equal(parsedLight.AttenuationStart.Values[0], 80.25);
        assert.equal(parsedLight.AttenuationEnd.Values[0], 200.5);
    }
    console.log(`ok - v${version} light layout and fractional attenuation survive MDX round trips`);
}
for (const version of [1600, 1700, 1800]) {
    const bytes = model(version, chunk('LITE', Buffer.concat([light, track('KLAV', 0.75)])),
        chunk('CAMS', Buffer.concat([camera, track('IDUF', 10), track('ELAF', 8), track('PTSF', 35)])));
    const parsed = parse(bytes);
    checkExtended(parsed);
    const reparsed = parseMDX(generateMDX(parsed));
    checkExtended(reparsed);
    assert.deepEqual(new Uint8Array(generateMDX(reparsed)), new Uint8Array(generateMDX(parsed)));
    console.log(`ok - v${version} light fields, camera flags and extra tracks survive MDX round trips`);
}
