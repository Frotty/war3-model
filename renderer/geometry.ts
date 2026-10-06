import {mat4, vec3} from 'gl-matrix';
import {Geoset} from '../model';
import {NodeWrapper} from './rendererData';

export interface VisibleBounds {
    minimum: vec3;
    maximum: vec3;
}

/** Evaluate the same bone weights used by the mesh shaders, including HD's byte weights. */
export function skinVertex (out: vec3, geoset: Geoset, nodes: NodeWrapper[], index: number): vec3 {
    const source = geoset.Vertices;
    const x = source[index * 3], y = source[index * 3 + 1], z = source[index * 3 + 2];
    vec3.set(out, 0, 0, 0);
    let total = 0;
    const skin = geoset.SkinWeights;
    const group = skin?.length ? null : geoset.Groups[geoset.VertexGroup[index]];
    const count = skin?.length ? 4 : group?.length || 0;
    for (let j = 0; j < count; ++j) {
        const weight = skin?.length ? skin[index * 8 + j + 4] / 255 : 1 / count;
        if (!weight) continue;
        const matrix = nodes[skin?.length ? skin[index * 8 + j] : group[j]]?.matrix;
        if (!matrix) continue;
        out[0] += (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) * weight;
        out[1] += (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) * weight;
        out[2] += (matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]) * weight;
        total += weight;
    }
    if (!total) vec3.set(out, x, y, z);
    return out;
}

export function expandBounds (bounds: VisibleBounds, point: vec3): void {
    if (!Number.isFinite(point[0]) || !Number.isFinite(point[1]) || !Number.isFinite(point[2])) return;
    vec3.min(bounds.minimum, bounds.minimum, point);
    vec3.max(bounds.maximum, bounds.maximum, point);
}

export function emptyBounds (): VisibleBounds {
    return {minimum: vec3.fromValues(Infinity, Infinity, Infinity), maximum: vec3.fromValues(-Infinity, -Infinity, -Infinity)};
}

export function boundsCorners (bounds: VisibleBounds): vec3[] {
    const result: vec3[] = [];
    for (let i = 0; i < 8; ++i) result.push(vec3.fromValues(
        i & 1 ? bounds.maximum[0] : bounds.minimum[0],
        i & 2 ? bounds.maximum[1] : bounds.minimum[1],
        i & 4 ? bounds.maximum[2] : bounds.minimum[2]));
    return result;
}

export function sortedFaces (geoset: Geoset, nodes: NodeWrapper[], view: mat4): Uint16Array<ArrayBuffer> {
    const point = vec3.create();
    const depths = new Float32Array(geoset.Vertices.length / 3);
    for (let i = 0; i < depths.length; ++i) {
        skinVertex(point, geoset, nodes, i);
        depths[i] = view[2] * point[0] + view[6] * point[1] + view[10] * point[2] + view[14];
    }
    const triangles = Array.from({length: geoset.Faces.length / 3}, (_, i) => i);
    const depth = (i: number): number => depths[geoset.Faces[i * 3]] + depths[geoset.Faces[i * 3 + 1]] + depths[geoset.Faces[i * 3 + 2]];
    triangles.sort((a, b) => depth(a) - depth(b) || a - b);
    // GPUQueue.writeBuffer requires a size divisible by four bytes.
    const faces = new Uint16Array(Math.ceil(geoset.Faces.length / 2) * 2);
    triangles.forEach((triangle, i) => faces.set(geoset.Faces.subarray(triangle * 3, triangle * 3 + 3), i * 3));
    return faces;
}
