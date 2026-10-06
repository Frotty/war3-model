attribute vec3 aVertexPosition;
attribute vec3 aNormal;
attribute vec2 aTextureCoord;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat3 uTVertexAnim;

varying vec3 vNormal;
varying vec3 vFragPos;
varying vec2 vTextureCoord;

void main(void) {
    vec4 position = vec4(aVertexPosition, 1.0);
    gl_Position = uPMatrix * uMVMatrix * position;
    vTextureCoord = (uTVertexAnim * vec3(aTextureCoord, 1.)).xy;
    vNormal = aNormal;
    vFragPos = aVertexPosition;
}
