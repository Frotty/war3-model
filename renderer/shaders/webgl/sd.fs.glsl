precision mediump float;

varying vec3 vNormal;
varying vec3 vFragPos;
varying vec2 vTextureCoord;

uniform sampler2D uSampler;
uniform sampler2D uMaskSampler;
uniform vec3 uReplaceableColor;
uniform float uReplaceableType;
uniform float uLayerAlpha;
uniform vec4 uGeosetColor;
uniform vec4 uLighting;
uniform vec3 uAmbient;
uniform vec3 uLightPos;
uniform vec3 uLightColor;
#if MAX_MODEL_LIGHTS > 0
uniform vec4 uModelLightPositions[MAX_MODEL_LIGHTS];
uniform vec4 uModelLightColors[MAX_MODEL_LIGHTS];
uniform vec4 uModelLightAttenuation[MAX_MODEL_LIGHTS];
#endif
uniform float uDiscardAlphaLevel;
uniform float uCaptureAlpha;
uniform float uUseReplaceableMask;
uniform float uWireframe;

float hypot (vec2 z) {
    float t;
    float x = abs(z.x);
    float y = abs(z.y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return (z.x == 0.0 && z.y == 0.0) ? 0.0 : x * sqrt(1.0 + t * t);
}

void main(void) {
    if (uWireframe > 0.) {
        gl_FragColor = vec4(1.);
        return;
    }

    vec2 texCoord = vTextureCoord;
    vec4 maskColor = uUseReplaceableMask > 0. ? texture2D(uMaskSampler, texCoord) : vec4(0.0);
    float diffuseAlpha = maskColor.a;
    float teamMask = uUseReplaceableMask > 0. ? 1.0 - diffuseAlpha : 1.0;

    if (uReplaceableType == 0.) {
        gl_FragColor = texture2D(uSampler, texCoord);
    } else if (uReplaceableType == 1.) {
        if (uUseReplaceableMask > 0.) {
            gl_FragColor = vec4(mix(uReplaceableColor, maskColor.rgb, diffuseAlpha), 1.0);
        } else {
            gl_FragColor = vec4(uReplaceableColor, 1.0);
        }
    } else if (uReplaceableType == 2.) {
        float dist = hypot(texCoord - vec2(0.5, 0.5)) * 2.;
        float truncateDist = clamp(1. - dist * 1.4, 0., 1.);
        float alpha = sin(truncateDist);
        if (uUseReplaceableMask > 0.) {
            gl_FragColor = vec4(mix(uReplaceableColor * alpha, maskColor.rgb, diffuseAlpha), alpha);
        } else {
            gl_FragColor = vec4(uReplaceableColor * alpha * teamMask, alpha * teamMask);
        }
    }

    gl_FragColor *= uLayerAlpha;
    gl_FragColor *= uGeosetColor;
    if (uLighting.y < 0.5) {
        vec3 normal = normalize(vNormal + vec3(0., 0., 0.000001));
        vec3 light = uAmbient;
        if (uLighting.z < 0.5) light += uLightColor * uLighting.x * max(dot(normal, normalize(uLightPos - vFragPos)), 0.);
#if MAX_MODEL_LIGHTS > 0
        for (int i = 0; i < MAX_MODEL_LIGHTS; ++i) {
            if (float(i) >= uLighting.z) break;
            vec3 delta = uModelLightPositions[i].xyz - vFragPos;
            float distance = length(delta);
            vec3 direction = uModelLightPositions[i].w > 0.5 ? delta / max(distance, 0.000001) : uModelLightPositions[i].xyz;
            vec2 attenuation = uModelLightAttenuation[i].xy;
            float weight = uModelLightPositions[i].w > 0.5 ? clamp((attenuation.y - distance) / max(attenuation.y - attenuation.x, 0.000001), 0., 1.) : 1.;
            light += uModelLightColors[i].rgb * weight * max(dot(normal, direction), 0.);
        }
#endif
        gl_FragColor.rgb *= clamp(light, vec3(0.), vec3(1.5));
    }

    // A negative threshold means "discard near-black texels" for additive color-keyed effects.
    if (uDiscardAlphaLevel < 0.) {
        if (max(gl_FragColor.r, max(gl_FragColor.g, gl_FragColor.b)) < -uDiscardAlphaLevel) {
            discard;
        }
    } else if (gl_FragColor[3] < uDiscardAlphaLevel) {
        discard;
    }
    if (uCaptureAlpha > 0.5) {
        float coverage = clamp(max(gl_FragColor.r, max(gl_FragColor.g, gl_FragColor.b)), 0., 1.);
        if (coverage > 0.) gl_FragColor.rgb /= coverage;
        gl_FragColor.a *= coverage;
    }
}
