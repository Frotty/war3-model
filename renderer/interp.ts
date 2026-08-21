import {LineType, AnimVector} from '../model';
import {vec3, quat} from 'gl-matrix';

/**
 * A pair of keyframe indices into an AnimVector's flat arrays, plus the frame to interpolate at.
 * Indices rather than keyframe objects: the objects only exist if something asks for
 * `AnimVector.Keys`, and the renderer never does.
 */
export interface KeyframeRange {
    frame: number;
    left: number;
    right: number;
}

const findKeyframesRes: KeyframeRange = {
    frame: 0,
    left: 0,
    right: 0
};

// Scratch for pulling one keyframe's components out of the flat arrays. gl-matrix only reads
// indices 0..3 of its inputs, so four slots covers every vector size in the format.
const leftVector = new Float32Array(4);
const rightVector = new Float32Array(4);
const leftOutTan = new Float32Array(4);
const rightInTan = new Float32Array(4);

export function lerp (left: number, right: number, t: number): number {
    return left * (1 - t) + right * t;
}

function bezier (left: number, outTan: number, inTan: number, right: number, t: number): number {
    const inverseFactor = 1 - t,
        inverseFactorTimesTwo = inverseFactor * inverseFactor,
        factorTimes2 = t * t,
        factor1 = inverseFactorTimesTwo * inverseFactor,
        factor2 = 3 * t * inverseFactorTimesTwo,
        factor3 = 3 * factorTimes2 * inverseFactor,
        factor4 = factorTimes2 * t;

    return left * factor1 + outTan * factor2 + inTan * factor3 + right * factor4;
}

function hermite (left: number, outTan: number, inTan: number, right: number, t: number): number {
    const factorTimes2 = t * t,
        factor1 = factorTimes2 * (2 * t - 3) + 1,
        factor2 = factorTimes2 * (t - 2) + t,
        factor3 = factorTimes2 * (t - 1),
        factor4 = factorTimes2 * (3 - 2 * t);

    return left * factor1 + outTan * factor2 + inTan * factor3 + right * factor4;
}

function copyInto (out: Float32Array, source: Float32Array|Int32Array, index: number, size: number): Float32Array {
    const base = index * size;

    for (let i = 0; i < size; ++i) {
        out[i] = source[base + i];
    }

    return out;
}

export function findKeyframes (animVector: AnimVector, frame: number, from: number, to: number): KeyframeRange|null {
    if (!animVector) {
        return null;
    }

    const frames = animVector.Frames;
    let first = 0;
    let count = frames.length;

    if (count === 0) {
        return null;
    }

    if (frames[0] > to) {
        return null;
    } else if (frames[count - 1] < from) {
        return null;
    }

    while (count > 0) {
        const step = count >> 1;
        if (frames[first + step] <= frame) {
            first = first + step + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }

    if (first === frames.length || frames[first] > to) {
        if (first > 0 && frames[first - 1] >= from) {
            findKeyframesRes.frame = frame;
            findKeyframesRes.left = first - 1;
            findKeyframesRes.right = first - 1;

            return findKeyframesRes;
        } else {
            return null;
        }
    }
    if (first === 0 || frames[first - 1] < from) {
        if (frames[first] <= to) {
            findKeyframesRes.frame = frame;
            findKeyframesRes.left = first;
            findKeyframesRes.right = first;

            return findKeyframesRes;
        } else {
            return null;
        }
    }

    findKeyframesRes.frame = frame;
    findKeyframesRes.left = first - 1;
    findKeyframesRes.right = first;

    return findKeyframesRes;
}

export function interpNum (animVector: AnimVector, frame: number, left: number, right: number): number {
    const frames = animVector.Frames;
    const values = animVector.Values;
    const size = animVector.VectorSize;
    const leftBase = left * size;
    const rightBase = right * size;

    if (frames[left] === frames[right]) {
        return values[leftBase];
    }

    const t = (frame - frames[left]) / (frames[right] - frames[left]);

    if (animVector.LineType === LineType.DontInterp) {
        return values[leftBase];
    } else if (animVector.LineType === LineType.Bezier) {
        return bezier(values[leftBase], animVector.OutTans[leftBase], animVector.InTans[rightBase], values[rightBase], t);
    } else if (animVector.LineType === LineType.Hermite) {
        return hermite(values[leftBase], animVector.OutTans[leftBase], animVector.InTans[rightBase], values[rightBase], t);
    } else {
        // Linear
        return lerp(values[leftBase], values[rightBase], t);
    }
}

export function interpVec3 (out: vec3, animVector: AnimVector, frame: number, left: number, right: number): vec3 {
    const frames = animVector.Frames;
    const values = animVector.Values;
    const size = animVector.VectorSize;

    // Always writes through `out`. Callers hold several results at once — updateNode keeps
    // translation, rotation and scaling live together — so returning a shared scratch buffer
    // would alias them.
    if (frames[left] === frames[right] || animVector.LineType === LineType.DontInterp) {
        return copyInto(out as Float32Array, values, left, size) as unknown as vec3;
    }

    const a = copyInto(leftVector, values, left, size);
    const b = copyInto(rightVector, values, right, size);
    const t = (frame - frames[left]) / (frames[right] - frames[left]);

    if (animVector.LineType === LineType.Bezier) {
        return vec3.bezier(out, a as unknown as vec3,
            copyInto(leftOutTan, animVector.OutTans, left, size) as unknown as vec3,
            copyInto(rightInTan, animVector.InTans, right, size) as unknown as vec3,
            b as unknown as vec3, t);
    } else if (animVector.LineType === LineType.Hermite) {
        return vec3.hermite(out, a as unknown as vec3,
            copyInto(leftOutTan, animVector.OutTans, left, size) as unknown as vec3,
            copyInto(rightInTan, animVector.InTans, right, size) as unknown as vec3,
            b as unknown as vec3, t);
    } else {
        return vec3.lerp(out, a as unknown as vec3, b as unknown as vec3, t);
    }
}

export function interpQuat (out: quat, animVector: AnimVector, frame: number, left: number, right: number): quat {
    const frames = animVector.Frames;
    const values = animVector.Values;
    const size = animVector.VectorSize;

    if (frames[left] === frames[right] || animVector.LineType === LineType.DontInterp) {
        return copyInto(out as Float32Array, values, left, size) as unknown as quat;
    }

    const a = copyInto(leftVector, values, left, size);
    const b = copyInto(rightVector, values, right, size);
    const t = (frame - frames[left]) / (frames[right] - frames[left]);

    if (animVector.LineType === LineType.Hermite || animVector.LineType === LineType.Bezier) {
        return quat.sqlerp(out, a as unknown as quat,
            copyInto(leftOutTan, animVector.OutTans, left, size) as unknown as quat,
            copyInto(rightInTan, animVector.InTans, right, size) as unknown as quat,
            b as unknown as quat, t);
    } else {
        return quat.slerp(out, a as unknown as quat, b as unknown as quat, t);
    }
}
