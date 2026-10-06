attribute vec3 aVertexPosition;
attribute vec3 aNormal;
attribute vec2 aTextureCoord;
attribute vec4 aGroup;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat3 uTVertexAnim;
uniform mat4 uNodesMatrices[${MAX_NODES}];

varying vec3 vNormal;
varying vec3 vFragPos;
varying vec2 vTextureCoord;

void main(void) {
    vec4 position = vec4(aVertexPosition, 1.0);
    int count = 1;
    mat4 skin = uNodesMatrices[int(aGroup[0])];

    if (aGroup[1] < ${MAX_NODES}.) {
        skin += uNodesMatrices[int(aGroup[1])];
        count += 1;
    }
    if (aGroup[2] < ${MAX_NODES}.) {
        skin += uNodesMatrices[int(aGroup[2])];
        count += 1;
    }
    if (aGroup[3] < ${MAX_NODES}.) {
        skin += uNodesMatrices[int(aGroup[3])];
        count += 1;
    }
    skin /= float(count);
    position = skin * position;
    position.w = 1.;

    gl_Position = uPMatrix * uMVMatrix * position;
    vTextureCoord = (uTVertexAnim * vec3(aTextureCoord, 1.)).xy;
    mat3 linear = mat3(skin);
    vec3 cof0 = cross(linear[1], linear[2]);
    vec3 cof1 = cross(linear[2], linear[0]);
    vec3 cof2 = cross(linear[0], linear[1]);
    float determinant = dot(linear[0], cof0);
    vNormal = abs(determinant) > 0.000001 ? (cof0 * aNormal.x + cof1 * aNormal.y + cof2 * aNormal.z) / determinant : aNormal;
    vFragPos = position.xyz;
}
