import {vec3} from 'gl-matrix';
import {LightType} from '../model';
import {RendererData} from './rendererData';
import {ModelInterp} from './modelInterp';

export const MAX_MODEL_LIGHTS = 8;
export interface LightingData {
    ambient: vec3;
    positions: Float32Array<ArrayBuffer>;
    colors: Float32Array<ArrayBuffer>;
    attenuation: Float32Array<ArrayBuffer>;
    count: number;
}
export function createLighting (): LightingData {
    return {ambient: vec3.create(), positions: new Float32Array(MAX_MODEL_LIGHTS * 4),
        colors: new Float32Array(MAX_MODEL_LIGHTS * 4), attenuation: new Float32Array(MAX_MODEL_LIGHTS * 4), count: 0};
}
const tempColor = vec3.create();
const tempAmbient = vec3.create();
const tempPosition = vec3.create();

export function evaluateLighting (out: LightingData, data: RendererData, interp: ModelInterp, ambient: number, diffuse: number): void {
    vec3.set(out.ambient, ambient, ambient, ambient);
    out.positions.fill(0); out.colors.fill(0); out.attenuation.fill(0);
    out.count = 0;
    for (const light of data.model.Lights) {
        const visibility = interp.animVectorVal(light.Visibility, 1);
        if (visibility <= 0) continue;
        const ambientColor = light.AmbColor instanceof Float32Array ? light.AmbColor : interp.vec3(tempAmbient, light.AmbColor);
        const ambientIntensity = interp.animVectorVal(light.AmbIntensity, 0) * diffuse * visibility;
        if (ambientColor) for (let j = 0; j < 3; ++j) out.ambient[j] += ambientColor[j] * ambientIntensity;
        if (light.LightType === LightType.Ambient || out.count >= MAX_MODEL_LIGHTS) continue;
        const color = light.Color instanceof Float32Array ? light.Color : interp.vec3(tempColor, light.Color);
        const intensity = interp.animVectorVal(light.Intensity, 1) * diffuse * visibility;
        const matrix = data.nodes[light.ObjectId]?.matrix;
        if (!color || !matrix || intensity <= 0) continue;
        const offset = out.count++ * 4;
        if (light.LightType === LightType.Omnidirectional) {
            vec3.transformMat4(tempPosition, light.PivotPoint as vec3, matrix);
            out.positions.set(tempPosition, offset);
            out.positions[offset + 3] = 1;
        } else {
            vec3.set(tempPosition, matrix[8], matrix[9], matrix[10]);
            vec3.normalize(tempPosition, tempPosition);
            out.positions.set(tempPosition, offset);
        }
        for (let j = 0; j < 3; ++j) out.colors[offset + j] = color[j] * intensity;
        out.attenuation[offset] = interp.animVectorVal(light.AttenuationStart, 0);
        out.attenuation[offset + 1] = interp.animVectorVal(light.AttenuationEnd, 1e10);
    }
}
