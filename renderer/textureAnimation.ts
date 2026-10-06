import {mat3, mat4, quat, vec3} from 'gl-matrix';
import {Layer, Model} from '../model';
import {ModelInterp} from './modelInterp';

const translation = vec3.create();
const rotation = quat.create();
const scaling = vec3.create();
const transform = mat4.create();
const matrix = mat3.create();
const identity = mat3.create();
const zero = vec3.create();
const one = vec3.fromValues(1, 1, 1);
const noRotation = quat.create();

export function textureAnimationMatrix (interp: ModelInterp, model: Model, layer: Layer): mat3 {
    const animation = model.TextureAnims[layer.TVertexAnimId];
    if (!animation) return identity;
    mat4.fromRotationTranslationScale(transform,
        interp.quat(rotation, animation.Rotation) || noRotation,
        interp.vec3(translation, animation.Translation) || zero,
        interp.vec3(scaling, animation.Scaling) || one);
    mat3.set(matrix, transform[0], transform[1], 0, transform[4], transform[5], 0, transform[12], transform[13], 0);
    return matrix;
}
