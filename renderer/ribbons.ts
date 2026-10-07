/// <reference types="vite/client" />
/// <reference types="@webgpu/types" />

import {getSharedProgram} from './util';
import {RendererData} from './rendererData';
import {VisibleBounds, expandBounds} from './geometry';
import {ModelInterp} from './modelInterp';
import {FilterMode, Layer, LayerShading, Material, RibbonEmitter} from '../model';
import {mat4, vec3} from 'gl-matrix';
import {textureAnimationMatrix} from './textureAnimation';
import vertexShader from './shaders/webgl/ribbon.vs.glsl?raw';
import fragmentShader from './shaders/webgl/ribbon.fs.glsl?raw';
import ribbonShader from './shaders/webgpu/ribbons.wgsl?raw';

interface RibbonEmitterWrapper {
    index: number;

    emission: number;
    props: RibbonEmitter;
    capacity: number;
    baseCapacity: number;
    creationTimes: number[];

    // xyz
    vertices: Float32Array<ArrayBuffer>;
    vertexBuffer: WebGLBuffer;
    vertexGPUBuffer: GPUBuffer;
    // xy
    texCoords: Float32Array<ArrayBuffer>;
    texCoordBuffer: WebGLBuffer;
    texCoordGPUBuffer: GPUBuffer;

    fsUnifrmsPerLayer: GPUBuffer[];
}

export class RibbonsController {
    private gl: WebGL2RenderingContext | WebGLRenderingContext;
    private shaderProgram: WebGLProgram;

    private device: GPUDevice;
    private gpuShaderModule: GPUShaderModule;
    private gpuPipelineLayout: GPUPipelineLayout;
    private gpuPipelines: GPURenderPipeline[];
    private vsBindGroupLayout: GPUBindGroupLayout | null;
    private fsBindGroupLayout: GPUBindGroupLayout | null;
    private gpuVSUniformsBuffer: GPUBuffer;
    private gpuVSUniformsBindGroup: GPUBindGroup;

    private shaderProgramLocations: {
        vertexPositionAttribute: number,
        textureCoordAttribute: number,
        pMatrixUniform: WebGLUniformLocation | null,
        mvMatrixUniform: WebGLUniformLocation | null,
        samplerUniform: WebGLUniformLocation | null,
        replaceableColorUniform: WebGLUniformLocation | null,
        replaceableTypeUniform: WebGLUniformLocation | null,
        discardAlphaLevelUniform: WebGLUniformLocation | null,
        colorUniform: WebGLUniformLocation | null,
        tVertexAnimUniform: WebGLUniformLocation | null
    };

    private interp: ModelInterp;
    private rendererData: RendererData;
    private emitters: RibbonEmitterWrapper[];
    private elapsed = 0;

    constructor (interp: ModelInterp, rendererData: RendererData) {
        this.shaderProgramLocations = {
            vertexPositionAttribute: null,
            textureCoordAttribute: null,
            pMatrixUniform: null,
            mvMatrixUniform: null,
            samplerUniform: null,
            replaceableColorUniform: null,
            replaceableTypeUniform: null,
            discardAlphaLevelUniform: null,
            colorUniform: null,
            tVertexAnimUniform: null
        };

        this.interp = interp;
        this.rendererData = rendererData;
        this.emitters = [];

        if (rendererData.model.RibbonEmitters.length) {
            for (let i = 0; i < rendererData.model.RibbonEmitters.length; ++i) {
                const ribbonEmitter = rendererData.model.RibbonEmitters[i];

                const emitter: RibbonEmitterWrapper = {
                    index: i,

                    emission: 0,
                    props: ribbonEmitter,
                    capacity: 0,
                    baseCapacity: 0,
                    creationTimes: [],
                    vertices: null,
                    vertexBuffer: null,
                    vertexGPUBuffer: null,
                    texCoords: null,
                    texCoordBuffer: null,
                    texCoordGPUBuffer: null,
                    fsUnifrmsPerLayer: []
                };

                emitter.baseCapacity = Math.ceil(
                    ModelInterp.maxAnimVectorVal(emitter.props.EmissionRate) * emitter.props.LifeSpan
                ) + 1; // extra points

                this.emitters.push(emitter);
            }
        }
    }

    public destroy (): void {
        if (this.shaderProgram) {
            this.shaderProgram = null;
        }
        if (this.gpuVSUniformsBuffer) {
            this.gpuVSUniformsBuffer.destroy();
            this.gpuVSUniformsBuffer = null;
        }
        for (const emitter of this.emitters) {
            if (this.gl) {
                this.gl.deleteBuffer(emitter.vertexBuffer);
                this.gl.deleteBuffer(emitter.texCoordBuffer);
            }
            emitter.vertexGPUBuffer?.destroy();
            emitter.texCoordGPUBuffer?.destroy();
            for (const buffer of emitter.fsUnifrmsPerLayer) {
                buffer.destroy();
            }
        }
        this.emitters = [];
    }

    public initGL (glContext: WebGLRenderingContext): void {
        this.gl = glContext;

        // See ParticlesController.initGL — nothing to draw means nothing to compile.
        if (!this.emitters.length) {
            return;
        }

        this.initShaders();
    }

    public initGPUDevice (device: GPUDevice): void {
        this.device = device;

        // Same reasoning as initGL: no emitters, no pipelines.
        if (!this.emitters.length) {
            return;
        }

        this.gpuShaderModule = device.createShaderModule({
            label: 'ribbons shader module',
            code: ribbonShader
        });

        this.vsBindGroupLayout = this.device.createBindGroupLayout({
            label: 'ribbons vs bind group layout',
            entries: [ {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {
                    type: 'uniform',
                    hasDynamicOffset: false,
                    minBindingSize: 128
                }
            }] as const
        });
        this.fsBindGroupLayout = this.device.createBindGroupLayout({
            label: 'ribbons bind group layout2',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                    type: 'uniform',
                        hasDynamicOffset: false,
                        minBindingSize: 96
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                        sampler: {
                        type: 'filtering'
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType: 'float',
                        viewDimension: "2d",
                        multisampled: false
                    }
                }
            ] as const
        });

        this.gpuPipelineLayout = this.device.createPipelineLayout({
            label: 'ribbons pipeline layout',
            bindGroupLayouts: [
                this.vsBindGroupLayout,
                this.fsBindGroupLayout
            ]
        });

        const createPipeline = (name: string, blend: GPUBlendState, depth: GPUDepthStencilState) => {
            return device.createRenderPipeline({
                label: `ribbons pipeline ${name}`,
                layout: this.gpuPipelineLayout,
                vertex: {
                    module: this.gpuShaderModule,
                    buffers: [{
                        arrayStride: 12,
                        attributes: [{
                            shaderLocation: 0,
                            offset: 0,
                            format: 'float32x3' as const
                        }]
                    }, {
                        arrayStride: 8,
                        attributes: [{
                            shaderLocation: 1,
                            offset: 0,
                            format: 'float32x2' as const
                        }]
                    }]
                },
                fragment: {
                    module: this.gpuShaderModule,
                    targets: [{
                        format: navigator.gpu.getPreferredCanvasFormat(),
                        blend
                    }]
                },
                depthStencil: depth,
                primitive: {
                    topology: 'triangle-strip'
                }
            });
        };

        this.gpuPipelines = [
            createPipeline('none', {
                color: {
                    operation: 'add',
                    srcFactor: 'one',
                    dstFactor: 'zero'
                },
                alpha: {
                    operation: 'add',
                    srcFactor: 'one',
                    dstFactor: 'zero'
                }
            }, {
                depthWriteEnabled: true,
                depthCompare: 'less-equal',
                format: 'depth24plus'
            }),
            createPipeline('transparent', {
                color: {
                    operation: 'add',
                    srcFactor: 'src-alpha',
                    dstFactor: 'one-minus-src-alpha'
                },
                alpha: {
                    operation: 'add',
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha'
                }
            }, {
                depthWriteEnabled: true,
                depthCompare: 'less-equal',
                format: 'depth24plus'
            }),
            createPipeline('blend', {
                color: {
                    operation: 'add',
                    srcFactor: 'src-alpha',
                    dstFactor: 'one-minus-src-alpha'
                },
                alpha: {
                    operation: 'add',
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha'
                }
            }, {
                depthWriteEnabled: false,
                depthCompare: 'less-equal',
                format: 'depth24plus'
            }),
            createPipeline('additive', {
                color: {
                    operation: 'add',
                    srcFactor: 'src',
                    dstFactor: 'one'
                },
                alpha: {
                    operation: 'add',
                    srcFactor: 'zero',
                    dstFactor: 'one'
                }
            }, {
                depthWriteEnabled: false,
                depthCompare: 'less-equal',
                format: 'depth24plus'
            }),
            createPipeline('addAlpha', {
                color: {
                    operation: 'add',
                    srcFactor: 'src-alpha',
                    dstFactor: 'one'
                },
                alpha: {
                    operation: 'add',
                    srcFactor: 'zero',
                    dstFactor: 'one'
                }
            }, {
                depthWriteEnabled: false,
                depthCompare: 'less-equal',
                format: 'depth24plus'
            }),
            createPipeline('modulate', {
                color: {
                    operation: 'add',
                    srcFactor: 'zero',
                    dstFactor: 'src'
                },
                alpha: {
                    operation: 'add',
                    srcFactor: 'zero',
                    dstFactor: 'one'
                }
            }, {
                depthWriteEnabled: false,
                depthCompare: 'less-equal',
                format: 'depth24plus'
            }),
            createPipeline('modulate2x', {
                color: {
                    operation: 'add',
                    srcFactor: 'dst',
                    dstFactor: 'src'
                },
                alpha: {
                    operation: 'add',
                    srcFactor: 'zero',
                    dstFactor: 'one'
                }
            }, {
                depthWriteEnabled: false,
                depthCompare: 'less-equal',
                format: 'depth24plus'
            })
        ];

        this.gpuVSUniformsBuffer = this.device.createBuffer({
            label: 'ribbons vs uniforms',
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.gpuVSUniformsBindGroup = this.device.createBindGroup({
            layout: this.vsBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.gpuVSUniformsBuffer }
                }
            ]
        });
    }

    public update (delta: number): void {
        this.elapsed += delta;
        for (const emitter of this.emitters) {
            this.updateEmitter(emitter, delta);
        }
    }

    public reset (): void {
        this.elapsed = 0;
        for (const emitter of this.emitters) {
            emitter.creationTimes.length = 0;
            emitter.emission = 0;
        }
    }

    public expandBounds (bounds: VisibleBounds): void {
        const point = vec3.create();
        for (const emitter of this.emitters) {
            if (!this.visibleEmitter(emitter)) continue;
            for (let i = 0; i < emitter.creationTimes.length * 6; i += 3) {
                vec3.set(point, emitter.vertices[i], emitter.vertices[i + 1], emitter.vertices[i + 2]);
                expandBounds(bounds, point);
            }
        }
    }

    private visibleEmitter (emitter: RibbonEmitterWrapper): boolean {
        return emitter.creationTimes.length >= 2 && this.interp.animVectorVal(emitter.props.Alpha, 1) > 1e-6 &&
            this.rendererData.model.Materials[emitter.props.MaterialID].Layers.some(layer => this.interp.animVectorVal(layer.Alpha, 1) > 1e-6);
    }

    public render (mvMatrix: mat4, pMatrix: mat4, emitterIndex?: number): void {
        if (!this.emitters.length) {
            return;
        }

        this.gl.useProgram(this.shaderProgram);

        this.gl.uniformMatrix4fv(this.shaderProgramLocations.pMatrixUniform, false, pMatrix);
        this.gl.uniformMatrix4fv(this.shaderProgramLocations.mvMatrixUniform, false, mvMatrix);

        this.gl.enableVertexAttribArray(this.shaderProgramLocations.vertexPositionAttribute);
        this.gl.enableVertexAttribArray(this.shaderProgramLocations.textureCoordAttribute);

        for (const emitter of this.emitters) {
            if (emitterIndex !== undefined && emitter.index !== emitterIndex) continue;
            if (!this.visibleEmitter(emitter)) {
                continue;
            }

            this.setGeneralBuffers(emitter);
            const materialID: number = emitter.props.MaterialID;
            const material: Material = this.rendererData.model.Materials[materialID];
            for (let j = 0; j < material.Layers.length; ++j) {
                const color = emitter.props.Color || new Float32Array([1, 1, 1]);
                const layerAlpha = this.interp.animVectorVal(material.Layers[j].Alpha, 1);
                if (layerAlpha < 1e-6) continue;
                this.gl.uniform4f(this.shaderProgramLocations.colorUniform,
                    color[2] * layerAlpha, color[1] * layerAlpha, color[0] * layerAlpha,
                    this.interp.animVectorVal(emitter.props.Alpha, 1) * layerAlpha);
                this.setLayerProps(material.Layers[j], this.rendererData.materialLayerTextureID[materialID][j]);
                this.renderEmitter(emitter);
            }
        }

        this.gl.disableVertexAttribArray(this.shaderProgramLocations.vertexPositionAttribute);
        this.gl.disableVertexAttribArray(this.shaderProgramLocations.textureCoordAttribute);
    }

    public renderGPU (pass: GPURenderPassEncoder, mvMatrix: mat4, pMatrix: mat4, emitterIndex?: number): void {
        if (!this.emitters.length) {
            return;
        }

        const VSUniformsValues = new ArrayBuffer(128);
        const VSUniformsViews = {
            mvMatrix: new Float32Array(VSUniformsValues, 0, 16),
            pMatrix: new Float32Array(VSUniformsValues, 64, 16)
        };
        VSUniformsViews.mvMatrix.set(mvMatrix);
        VSUniformsViews.pMatrix.set(pMatrix);
        this.device.queue.writeBuffer(this.gpuVSUniformsBuffer, 0, VSUniformsValues);

        for (const emitter of this.emitters) {
            if (emitterIndex !== undefined && emitter.index !== emitterIndex) continue;
            if (!this.visibleEmitter(emitter)) {
                continue;
            }

            this.device.queue.writeBuffer(emitter.vertexGPUBuffer, 0, emitter.vertices);
            this.device.queue.writeBuffer(emitter.texCoordGPUBuffer, 0, emitter.texCoords);

            pass.setVertexBuffer(0, emitter.vertexGPUBuffer);
            pass.setVertexBuffer(1, emitter.texCoordGPUBuffer);

            pass.setBindGroup(0, this.gpuVSUniformsBindGroup);

            const materialID: number = emitter.props.MaterialID;
            const material: Material = this.rendererData.model.Materials[materialID];

            for (let j = 0; j < material.Layers.length; ++j) {
                const textureID = this.rendererData.materialLayerTextureID[materialID][j];
                const texture = this.rendererData.model.Textures[textureID];
                const layer = material.Layers[j];
                const layerAlpha = this.interp.animVectorVal(layer.Alpha, 1);
                if (layerAlpha < 1e-6) continue;

                const pipeline = this.gpuPipelines[layer.FilterMode] || this.gpuPipelines[0];
                pass.setPipeline(pipeline);

                const fsUniformsValues = new ArrayBuffer(96);
                const fsUniformsViews = {
                    replaceableColor: new Float32Array(fsUniformsValues, 0, 3),
                    replaceableType: new Uint32Array(fsUniformsValues, 12, 1),
                    discardAlphaLevel: new Float32Array(fsUniformsValues, 16, 1),
                    color: new Float32Array(fsUniformsValues, 32, 4),
                    tVertexAnim: new Float32Array(fsUniformsValues, 48, 12),
                };

                fsUniformsViews.replaceableColor.set(this.rendererData.teamColor);
                fsUniformsViews.replaceableType.set([texture.ReplaceableId || 0]);
                fsUniformsViews.discardAlphaLevel.set([layer.FilterMode === FilterMode.Transparent ? .75 : 0]);
                const color = emitter.props.Color || new Float32Array([1, 1, 1]);
                fsUniformsViews.color.set([color[2] * layerAlpha, color[1] * layerAlpha, color[0] * layerAlpha,
                    this.interp.animVectorVal(emitter.props.Alpha, 1) * layerAlpha]);
                const texAnim = textureAnimationMatrix(this.interp, this.rendererData.model, layer);
                for (let row = 0; row < 3; ++row) fsUniformsViews.tVertexAnim.set(texAnim.slice(row * 3, row * 3 + 3), row * 4);

                if (!emitter.fsUnifrmsPerLayer[j]) {
                    emitter.fsUnifrmsPerLayer[j] = this.device.createBuffer({
                        label: `ribbons fs uniforms ${emitter.index} layer ${j}`,
                        size: 96,
                        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
                    });
                }
                const fsUniformsBuffer = emitter.fsUnifrmsPerLayer[j];

                this.device.queue.writeBuffer(fsUniformsBuffer, 0, fsUniformsValues);

                const fsUniformsBindGroup = this.device.createBindGroup({
                    label: `ribbons fs uniforms ${emitter.index}`,
                    layout: this.fsBindGroupLayout,
                    entries: [
                        {
                            binding: 0,
                            resource: { buffer: fsUniformsBuffer }
                        },
                        {
                            binding: 1,
                            resource: this.rendererData.gpuSamplers[textureID]
                        },
                        {
                            binding: 2,
                            resource: (this.rendererData.gpuTextures[texture.Image] || this.rendererData.gpuEmptyTexture).createView()
                        }
                    ]
                });

                pass.setBindGroup(1, fsUniformsBindGroup);

                pass.draw(emitter.creationTimes.length * 2);
            }
        }
    }

    private initShaders (): void {
        const shaderProgram = this.shaderProgram = getSharedProgram(this.gl, vertexShader, fragmentShader);

        this.gl.useProgram(shaderProgram);

        this.shaderProgramLocations.vertexPositionAttribute =
            this.gl.getAttribLocation(shaderProgram, 'aVertexPosition');
        this.shaderProgramLocations.textureCoordAttribute =
            this.gl.getAttribLocation(shaderProgram, 'aTextureCoord');

        this.shaderProgramLocations.pMatrixUniform = this.gl.getUniformLocation(shaderProgram, 'uPMatrix');
        this.shaderProgramLocations.mvMatrixUniform = this.gl.getUniformLocation(shaderProgram, 'uMVMatrix');
        this.shaderProgramLocations.samplerUniform = this.gl.getUniformLocation(shaderProgram, 'uSampler');
        this.shaderProgramLocations.replaceableColorUniform =
            this.gl.getUniformLocation(shaderProgram, 'uReplaceableColor');
        this.shaderProgramLocations.replaceableTypeUniform =
            this.gl.getUniformLocation(shaderProgram, 'uReplaceableType');
        this.shaderProgramLocations.discardAlphaLevelUniform =
            this.gl.getUniformLocation(shaderProgram, 'uDiscardAlphaLevel');
        this.shaderProgramLocations.colorUniform =
            this.gl.getUniformLocation(shaderProgram, 'uColor');
        this.shaderProgramLocations.tVertexAnimUniform = this.gl.getUniformLocation(shaderProgram, 'uTVertexAnim');
    }

    private resizeEmitterBuffers (emitter: RibbonEmitterWrapper, size: number): void {
        if (size <= emitter.capacity) {
            return;
        }

        size = Math.min(size, emitter.baseCapacity);

        const vertices = new Float32Array(size * 2 * 3);  // 2 vertices * xyz
        const texCoords = new Float32Array(size * 2 * 2); // 2 vertices * xy

        if (emitter.vertices) {
            vertices.set(emitter.vertices);
        }

        emitter.vertices = vertices;
        emitter.texCoords = texCoords;

        emitter.capacity = size;

        if (this.gl) {
            if (!emitter.vertexBuffer) {
                emitter.vertexBuffer = this.gl.createBuffer();
                emitter.texCoordBuffer = this.gl.createBuffer();
            }
        } else if (this.device) {
            emitter.vertexGPUBuffer?.destroy();
            emitter.texCoordGPUBuffer?.destroy();

            emitter.vertexGPUBuffer = this.device.createBuffer({
                label: `ribbon vertex buffer ${emitter.index}`,
                size: vertices.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            emitter.texCoordGPUBuffer = this.device.createBuffer({
                label: `ribbon texCoord buffer ${emitter.index}`,
                size: texCoords.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
        }
    }

    private updateEmitter (emitter: RibbonEmitterWrapper, delta: number): void {
        const now = this.elapsed;
        const visibility = this.interp.animVectorVal(emitter.props.Visibility, 1);

        // Apply gravity to existing points only. A frozen update(0) must not move the trail.
        const gravity = emitter.props.Gravity || 0;
        for (let i = 0; i < emitter.creationTimes.length; ++i) {
            const age = (now - emitter.creationTimes[i]) / 1000;
            const previousAge = Math.max(0, age - delta / 1000);
            const fall = 0.5 * gravity * (age * age - previousAge * previousAge);
            emitter.vertices[i * 6 + 2] -= fall;
            emitter.vertices[i * 6 + 5] -= fall;
        }

        if (visibility > 0) {
            const emissionRate = emitter.props.EmissionRate;

            emitter.emission += emissionRate * delta;

            if (emitter.emission >= 1000) {
                // only once per tick
                emitter.emission = emitter.emission % 1000;

                if (emitter.creationTimes.length + 1 > emitter.capacity) {
                    this.resizeEmitterBuffers(emitter, emitter.creationTimes.length + 1);
                }

                this.appendVertices(emitter);

                emitter.creationTimes.push(now);
            }
        }

        if (emitter.creationTimes.length) {
            while (emitter.creationTimes.length && emitter.creationTimes[0] + emitter.props.LifeSpan * 1000 < now) {
                emitter.creationTimes.shift();
                for (let i = 0; i + 6 + 5 < emitter.vertices.length; i += 6) {
                    emitter.vertices[i]     = emitter.vertices[i + 6];
                    emitter.vertices[i + 1] = emitter.vertices[i + 7];
                    emitter.vertices[i + 2] = emitter.vertices[i + 8];
                    emitter.vertices[i + 3] = emitter.vertices[i + 9];
                    emitter.vertices[i + 4] = emitter.vertices[i + 10];
                    emitter.vertices[i + 5] = emitter.vertices[i + 11];
                }
            }
        }

        // still exists
        if (emitter.creationTimes.length) {
            this.updateEmitterTexCoords(emitter, now);
        }
    }

    private appendVertices (emitter: RibbonEmitterWrapper): void {
        const first: vec3 = vec3.clone(emitter.props.PivotPoint as vec3);
        const second: vec3 = vec3.clone(emitter.props.PivotPoint as vec3);

        first[1] -= this.interp.animVectorVal(emitter.props.HeightBelow, 0);
        second[1] += this.interp.animVectorVal(emitter.props.HeightAbove, 0);

        const emitterMatrix: mat4 = this.rendererData.nodes[emitter.props.ObjectId].matrix;
        vec3.transformMat4(first, first, emitterMatrix);
        vec3.transformMat4(second, second, emitterMatrix);

        const currentSize = emitter.creationTimes.length;
        emitter.vertices[currentSize * 6]     = first[0];
        emitter.vertices[currentSize * 6 + 1] = first[1];
        emitter.vertices[currentSize * 6 + 2] = first[2];
        emitter.vertices[currentSize * 6 + 3] = second[0];
        emitter.vertices[currentSize * 6 + 4] = second[1];
        emitter.vertices[currentSize * 6 + 5] = second[2];
    }

    private updateEmitterTexCoords (emitter: RibbonEmitterWrapper, now: number): void {
        for (let i = 0; i < emitter.creationTimes.length; ++i) {
            let relativePos = (now - emitter.creationTimes[i]) / (emitter.props.LifeSpan * 1000);
            const textureSlot = this.interp.animVectorVal(emitter.props.TextureSlot, 0);

            const texCoordX = textureSlot % emitter.props.Columns;
            const texCoordY = Math.floor(textureSlot / emitter.props.Columns);
            const cellWidth = 1 / emitter.props.Columns;
            const cellHeight = 1 / emitter.props.Rows;

            relativePos = texCoordX * cellWidth + relativePos * cellWidth;

            emitter.texCoords[i * 2 * 2]     = relativePos;
            emitter.texCoords[i * 2 * 2 + 1] = texCoordY * cellHeight;
            emitter.texCoords[i * 2 * 2 + 2] = relativePos;
            emitter.texCoords[i * 2 * 2 + 3] = (1 + texCoordY) * cellHeight;
        }
    }

    private setLayerProps (layer: Layer, textureID: number): void {
        this.gl.uniformMatrix3fv(this.shaderProgramLocations.tVertexAnimUniform, false,
            textureAnimationMatrix(this.interp, this.rendererData.model, layer));
        const texture = this.rendererData.model.Textures[textureID];

        if (layer.Shading & LayerShading.TwoSided) {
            this.gl.disable(this.gl.CULL_FACE);
        } else {
            this.gl.enable(this.gl.CULL_FACE);
        }

        if (layer.FilterMode === FilterMode.Transparent) {
            this.gl.uniform1f(this.shaderProgramLocations.discardAlphaLevelUniform, 0.75);
        } else {
            this.gl.uniform1f(this.shaderProgramLocations.discardAlphaLevelUniform, 0.);
        }

        if (layer.FilterMode === FilterMode.None) {
            this.gl.disable(this.gl.BLEND);
            this.gl.enable(this.gl.DEPTH_TEST);
            // this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
            this.gl.depthMask(true);
        } else if (layer.FilterMode === FilterMode.Transparent) {
            this.gl.enable(this.gl.BLEND);
            this.gl.enable(this.gl.DEPTH_TEST);
            this.gl.blendFuncSeparate(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA, this.gl.ONE, this.gl.ONE_MINUS_SRC_ALPHA);
            this.gl.depthMask(true);
        } else if (layer.FilterMode === FilterMode.Blend) {
            this.gl.enable(this.gl.BLEND);
            this.gl.enable(this.gl.DEPTH_TEST);
            this.gl.blendFuncSeparate(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA, this.gl.ONE, this.gl.ONE_MINUS_SRC_ALPHA);
            this.gl.depthMask(false);
        } else if (layer.FilterMode === FilterMode.Additive) {
            this.gl.enable(this.gl.BLEND);
            this.gl.enable(this.gl.DEPTH_TEST);
            this.gl.blendFuncSeparate(this.gl.SRC_ALPHA, this.gl.ONE,
                this.rendererData.captureAlpha ? this.gl.ONE : this.gl.ZERO,
                this.rendererData.captureAlpha ? this.gl.ONE_MINUS_SRC_ALPHA : this.gl.ONE);
            this.gl.depthMask(false);
        } else if (layer.FilterMode === FilterMode.AddAlpha) {
            this.gl.enable(this.gl.BLEND);
            this.gl.enable(this.gl.DEPTH_TEST);
            this.gl.blendFuncSeparate(this.gl.SRC_ALPHA, this.gl.ONE,
                this.rendererData.captureAlpha ? this.gl.ONE : this.gl.ZERO,
                this.rendererData.captureAlpha ? this.gl.ONE_MINUS_SRC_ALPHA : this.gl.ONE);
            this.gl.depthMask(false);
        } else if (layer.FilterMode === FilterMode.Modulate) {
            this.gl.enable(this.gl.BLEND);
            this.gl.enable(this.gl.DEPTH_TEST);
            this.gl.blendFuncSeparate(this.gl.ZERO, this.gl.SRC_COLOR, this.gl.ZERO, this.gl.ONE);
            this.gl.depthMask(false);
        } else if (layer.FilterMode === FilterMode.Modulate2x) {
            this.gl.enable(this.gl.BLEND);
            this.gl.enable(this.gl.DEPTH_TEST);
            this.gl.blendFuncSeparate(this.gl.DST_COLOR, this.gl.SRC_COLOR, this.gl.ZERO, this.gl.ONE);
            this.gl.depthMask(false);
        }

        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.rendererData.textures[texture.Image] || this.rendererData.whiteTexture);
        this.gl.uniform1i(this.shaderProgramLocations.samplerUniform, 0);
        this.gl.uniform1f(this.shaderProgramLocations.replaceableTypeUniform, 0);
        if (texture.Image) {
            this.gl.uniform1f(this.shaderProgramLocations.replaceableTypeUniform, 0);
        } else if (texture.ReplaceableId === 1 || texture.ReplaceableId === 2) {
            this.gl.uniform3fv(this.shaderProgramLocations.replaceableColorUniform, this.rendererData.teamColor);
            this.gl.uniform1f(this.shaderProgramLocations.replaceableTypeUniform, texture.ReplaceableId);
        }

        if (layer.Shading & LayerShading.NoDepthTest) {
            this.gl.disable(this.gl.DEPTH_TEST);
        }
        if (layer.Shading & LayerShading.NoDepthSet) {
            this.gl.depthMask(false);
        }

    }

    private setGeneralBuffers (emitter: RibbonEmitterWrapper): void {
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, emitter.texCoordBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, emitter.texCoords, this.gl.DYNAMIC_DRAW);
        this.gl.vertexAttribPointer(this.shaderProgramLocations.textureCoordAttribute, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, emitter.vertexBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, emitter.vertices, this.gl.DYNAMIC_DRAW);
        this.gl.vertexAttribPointer(this.shaderProgramLocations.vertexPositionAttribute, 3, this.gl.FLOAT, false, 0, 0);
    }

    private renderEmitter (emitter: RibbonEmitterWrapper): void {
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, emitter.creationTimes.length * 2);
    }
}
