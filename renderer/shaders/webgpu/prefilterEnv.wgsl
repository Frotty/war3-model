const PI: f32 = 3.14159265359;
const gamma: f32 = 2.2;

struct VSUniforms {
    mvMatrix: mat4x4f,
    pMatrix: mat4x4f,
}

struct FSUniforms {
    roughness: f32,
}

@group(0) @binding(0) var<uniform> vsUniforms: VSUniforms;
@group(1) @binding(0) var<uniform> fsUniforms: FSUniforms;
@group(1) @binding(1) var fsUniformSampler: sampler;
@group(1) @binding(2) var fsUniformTexture: texture_cube<f32>;

struct VSIn {
    @location(0) vertexPosition: vec3f,
}

struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) localPos: vec3f,
}

@vertex fn vs(
    in: VSIn
) -> VSOut {
    var out: VSOut;
    out.position = vsUniforms.pMatrix * vsUniforms.mvMatrix * vec4f(in.vertexPosition, 1);
    out.localPos = in.vertexPosition;
    return out;
}

fn RadicalInverse_VdC(bits: u32) -> f32 {
    var res: u32 = bits;
    res = (res << 16u) | (res >> 16u);
    res = ((res & 0x55555555u) << 1u) | ((res & 0xAAAAAAAAu) >> 1u);
    res = ((res & 0x33333333u) << 2u) | ((res & 0xCCCCCCCCu) >> 2u);
    res = ((res & 0x0F0F0F0Fu) << 4u) | ((res & 0xF0F0F0F0u) >> 4u);
    res = ((res & 0x00FF00FFu) << 8u) | ((res & 0xFF00FF00u) >> 8u);
    return f32(res) * 2.3283064365386963e-10; // / 0x100000000
}

fn Hammersley(i: u32, N: u32) -> vec2f {
    return vec2f(f32(i)/f32(N), RadicalInverse_VdC(i));
}

fn ImportanceSampleGGX(Xi: vec2f, N: vec3f, roughness: f32) -> vec3f {
    let a: f32 = roughness * roughness;

    let phi: f32 = 2.0 * PI * Xi.x;
    let cosTheta: f32 = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    let sinTheta: f32 = sqrt(1.0 - cosTheta*cosTheta);

    // from spherical coordinates to cartesian coordinates
    var H: vec3f;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // from tangent-space vector to world-space sample vector
    var up: vec3f;
    if (abs(N.z) < 0.999) {
        up = vec3f(0.0, 0.0, 1.0);
    } else {
        up = vec3f(1.0, 0.0, 0.0);
    }
    let tangent: vec3f   = normalize(cross(up, N));
    let bitangent: vec3f = cross(N, tangent);

    let sampleVec: vec3f = tangent * H.x + bitangent * H.y + N * H.z;

    return normalize(sampleVec);
}

fn DistributionGGX(NdotH: f32, roughness: f32) -> f32 {
    let a: f32 = roughness * roughness;
    let a2: f32 = a * a;
    let denom: f32 = NdotH * NdotH * (a2 - 1.0) + 1.0;

    return a2 / (PI * denom * denom);
}

@fragment fn fs(
    in: VSOut
) -> @location(0) vec4f {
    let N: vec3f = normalize(in.localPos);
    let R: vec3f = N;
    let V: vec3f = R;

    // See the WebGL prefilterEnv shader: 256 samples read from the mip level matching each
    // sample's solid angle, instead of 1024 all read from level 0.
    const SAMPLE_COUNT: u32 = 256u;
    const resolution: f32 = ${ENV_MAP_SIZE}.0;
    const saTexel: f32 = 4.0 * PI / (6.0 * resolution * resolution);

    var totalWeight: f32 = 0.0;
    var prefilteredColor: vec3f = vec3f(0.0);
    for(var i: u32 = 0u; i < SAMPLE_COUNT; i++)
    {
        let Xi: vec2f = Hammersley(i, SAMPLE_COUNT);
        let H: vec3f  = ImportanceSampleGGX(Xi, N, fsUniforms.roughness);
        let L: vec3f  = normalize(2.0 * dot(V, H) * H - V);

        let NdotL: f32 = max(dot(N, L), 0.0);
        if(NdotL > 0.0) {
            let NdotH: f32 = max(dot(N, H), 0.0);
            let HdotV: f32 = max(dot(H, V), 0.0);
            let pdf: f32 = DistributionGGX(NdotH, fsUniforms.roughness) * NdotH / (4.0 * HdotV) + 0.0001;
            let saSample: f32 = 1.0 / (f32(SAMPLE_COUNT) * pdf + 0.0001);
            var mipLevel: f32 = 0.0;
            if (fsUniforms.roughness > 0.0) {
                mipLevel = 0.5 * log2(saSample / saTexel);
            }

            prefilteredColor += pow(textureSampleLevel(fsUniformTexture, fsUniformSampler, L, mipLevel).rgb, vec3f(gamma)) * NdotL;
            totalWeight      += NdotL;
        }
    }
    prefilteredColor = prefilteredColor / totalWeight;

    return vec4f(prefilteredColor, 1.0);
}
