struct VSUniforms {
    mvMatrix: mat4x4f,
    pMatrix: mat4x4f,
    nodesMatrices: array<mat4x4f, ${MAX_NODES}>,
}

struct FSUniforms {
    replaceableColor: vec3f,
    replaceableType: u32,
    discardAlphaLevel: f32,
    layerAlpha: f32,
    wireframe: u32,
    useReplaceableMask: u32,
    tVertexAnim: mat3x3f,
    geosetColor: vec4f,
    lighting: vec4f,
    ambient: vec4f,
    lightPos: vec4f,
    lightColor: vec4f,
    modelLightPositions: array<vec4f, 8>,
    modelLightColors: array<vec4f, 8>,
    modelLightAttenuation: array<vec4f, 8>,
}

@group(0) @binding(0) var<uniform> vsUniforms: VSUniforms;
@group(1) @binding(0) var<uniform> fsUniforms: FSUniforms;
@group(1) @binding(1) var fsUniformSampler: sampler;
@group(1) @binding(2) var fsUniformTexture: texture_2d<f32>;
@group(1) @binding(3) var fsMaskSampler: sampler;
@group(1) @binding(4) var fsMaskTexture: texture_2d<f32>;

struct VSIn {
    @location(0) vertexPosition: vec3f,
    @location(1) normal: vec3f,
    @location(2) textureCoord: vec2f,
    @location(3) group: vec4<u32>,
}

struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) textureCoord: vec2f,
    @location(2) worldPosition: vec3f,
}

@vertex fn vs(
    in: VSIn
) -> VSOut {
    var position: vec4f = vec4f(in.vertexPosition, 1.0);
    var count: i32 = 1;
    var skin = vsUniforms.nodesMatrices[in.group[0]];

    if (in.group[1] < ${MAX_NODES}) {
        skin += vsUniforms.nodesMatrices[in.group[1]];
        count += 1;
    }
    if (in.group[2] < ${MAX_NODES}) {
        skin += vsUniforms.nodesMatrices[in.group[2]];
        count += 1;
    }
    if (in.group[3] < ${MAX_NODES}) {
        skin += vsUniforms.nodesMatrices[in.group[3]];
        count += 1;
    }
    skin *= 1.0 / f32(count);
    position = skin * position;
    position.w = 1.;

    var out: VSOut;
    out.position = vsUniforms.pMatrix * vsUniforms.mvMatrix * position;
    out.textureCoord = in.textureCoord;
    let cof0 = cross(skin[1].xyz, skin[2].xyz);
    let cof1 = cross(skin[2].xyz, skin[0].xyz);
    let cof2 = cross(skin[0].xyz, skin[1].xyz);
    let determinant = dot(skin[0].xyz, cof0);
    out.normal = in.normal;
    if (abs(determinant) > 0.000001) {
        out.normal = (cof0 * in.normal.x + cof1 * in.normal.y + cof2 * in.normal.z) / determinant;
    }
    out.worldPosition = position.xyz;
    return out;
}

fn hypot(z: vec2f) -> f32 {
    var t: f32 = 0;
    var x: f32 = abs(z.x);
    let y: f32 = abs(z.y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    if (z.x == 0.0 && z.y == 0.0) {
        return 0.0;
    }
    return x * sqrt(1.0 + t * t);
}

@fragment fn fs(
    in: VSOut
) -> @location(0) vec4f {
    if (fsUniforms.wireframe > 0) {
        return vec4f(1);
    }

    let texCoord: vec2f = (fsUniforms.tVertexAnim * vec3f(in.textureCoord.x, in.textureCoord.y, 1.)).xy;
    var color: vec4f = vec4f(0.0);
    var diffuseAlpha: f32 = 0.0;
    var teamMask: f32 = 1.0;

    if (fsUniforms.useReplaceableMask != 0) {
        diffuseAlpha = textureSample(fsMaskTexture, fsMaskSampler, texCoord).a;
        teamMask = 1.0 - diffuseAlpha;
    }

    if (fsUniforms.replaceableType != 0 && fsUniforms.useReplaceableMask != 0 && teamMask <= 0.001) {
        discard;
    }

    if (fsUniforms.replaceableType == 0) {
        color = textureSample(fsUniformTexture, fsUniformSampler, texCoord);
    } else if (fsUniforms.replaceableType == 1) {
        color = vec4f(fsUniforms.replaceableColor, 1.0);
    } else if (fsUniforms.replaceableType == 2) {
        let dist: f32 = hypot(texCoord - vec2(0.5, 0.5)) * 2.;
        let truncateDist: f32 = clamp(1. - dist * 1.4, 0., 1.);
        let alpha: f32 = sin(truncateDist);
        color = vec4f(fsUniforms.replaceableColor * alpha * teamMask, alpha * teamMask);
    }

    color *= fsUniforms.layerAlpha;
    color *= fsUniforms.geosetColor;
    if (fsUniforms.lighting.y < 0.5) {
        let normal = normalize(in.normal + vec3f(0., 0., 0.000001));
        var light = fsUniforms.ambient.rgb;
        if (fsUniforms.lighting.z < 0.5) {
            light += fsUniforms.lightColor.rgb * fsUniforms.lighting.x * max(dot(normal, normalize(fsUniforms.lightPos.xyz - in.worldPosition)), 0.);
        }
        for (var i: u32 = 0; i < u32(fsUniforms.lighting.z); i++) {
            let delta = fsUniforms.modelLightPositions[i].xyz - in.worldPosition;
            let distance = length(delta);
            var direction = fsUniforms.modelLightPositions[i].xyz;
            var weight = 1.;
            if (fsUniforms.modelLightPositions[i].w > 0.5) {
                direction = delta / max(distance, 0.000001);
                let attenuation = fsUniforms.modelLightAttenuation[i].xy;
                weight = clamp((attenuation.y - distance) / max(attenuation.y - attenuation.x, 0.000001), 0., 1.);
            }
            light += fsUniforms.modelLightColors[i].rgb * weight * max(dot(normal, direction), 0.);
        }
        color = vec4f(color.rgb * clamp(light, vec3f(0.), vec3f(1.5)), color.a);
    }

    // A negative threshold means "discard near-black texels" for additive color-keyed effects.
    if (fsUniforms.discardAlphaLevel < 0.0) {
        if (max(color.r, max(color.g, color.b)) < -fsUniforms.discardAlphaLevel) {
            discard;
        }
    } else if (color.a < fsUniforms.discardAlphaLevel) {
        discard;
    }

    return color;
}
