export interface ModelInfo {
    Name: string;
    MinimumExtent: Float32Array;
    MaximumExtent: Float32Array;
    BoundsRadius: number;
    BlendTime: number;
    NumGeosets?: number;
    NumGeosetAnims?: number;
    NumBones?: number;
    NumLights?: number;
    NumAttachments?: number;
    NumEvents?: number;
    NumParticleEmitters?: number;
    NumParticleEmitters2?: number;
    NumRibbonEmitters?: number;
}

export interface Sequence {
    Name: string;
    Interval: Uint32Array;
    NonLooping: boolean;
    MinimumExtent: Float32Array;
    MaximumExtent: Float32Array;
    BoundsRadius: number;
    MoveSpeed: number;
    Rarity: number;
}

export enum TextureFlags {
    WrapWidth = 1,
    WrapHeight = 2
}

export interface Texture {
    Image: string;
    ReplaceableId?: number;
    Flags?: TextureFlags;
}

export enum FilterMode {
    None = 0,
    Transparent = 1,
    Blend = 2,
    Additive = 3,
    AddAlpha = 4,
    Modulate = 5,
    Modulate2x = 6
}

export enum LineType {
    DontInterp = 0,
    Linear = 1,
    Hermite = 2,
    Bezier = 3
}

export interface AnimKeyframe {
    Frame: number;
    Vector: Float32Array|Int32Array;
    InTan?: Float32Array|Int32Array;
    OutTan?: Float32Array|Int32Array;
}

export interface AnimVector {
    LineType: LineType;
    GlobalSeqId?: number;
    /**
     * Keyframes as objects, in frame order.
     *
     * Materialised from the flat arrays below on first access and cached from then on; the
     * `Vector`, `InTan` and `OutTan` of each keyframe are views into those arrays, so writing
     * through them updates the vector. Assigning a new array replaces the flat arrays.
     *
     * The renderer reads the flat arrays directly and never touches this, which is the point: a
     * Reforged hero carries around half a million keyframes, and building an object and a
     * separate typed array for each of them dominated parse time.
     */
    Keys: AnimKeyframe[];
    /** Keyframe times. Its length is the keyframe count. */
    Frames: Int32Array;
    /** Keyframe vectors, `VectorSize` components each, laid out end to end. */
    Values: Float32Array|Int32Array;
    /** In/out tangents, same layout as `Values`. Only present for Hermite and Bezier. */
    InTans?: Float32Array|Int32Array;
    OutTans?: Float32Array|Int32Array;
    /** Components per keyframe: 1, 3 or 4. */
    VectorSize: number;
}

function allocateLike (source: Float32Array|Int32Array, length: number): Float32Array|Int32Array {
    return source instanceof Int32Array ? new Int32Array(length) : new Float32Array(length);
}

/**
 * Backing implementation for AnimVector.
 *
 * `Keys` is a prototype accessor rather than a per-object one: every AnimVector then shares a
 * single hidden class, which keeps the flat-array reads in the interpolator monomorphic.
 * Defining the accessor per instance instead drops each object into dictionary mode and costs
 * more per frame than the lazy materialisation saves.
 */
class AnimVectorImpl implements AnimVector {
    public LineType: LineType;
    public GlobalSeqId: number;
    public VectorSize: number;
    public Frames: Int32Array;
    public Values: Float32Array|Int32Array;
    public InTans?: Float32Array|Int32Array;
    public OutTans?: Float32Array|Int32Array;

    private keysCache: AnimKeyframe[] = null;

    constructor (
        lineType: LineType,
        globalSeqId: number,
        vectorSize: number,
        frames: Int32Array,
        values: Float32Array|Int32Array,
        inTans?: Float32Array|Int32Array,
        outTans?: Float32Array|Int32Array
    ) {
        this.LineType = lineType;
        this.GlobalSeqId = globalSeqId;
        this.VectorSize = vectorSize;
        this.Frames = frames;
        this.Values = values;
        this.InTans = inTans;
        this.OutTans = outTans;
    }

    public get Keys (): AnimKeyframe[] {
        if (!this.keysCache) {
            const size = this.VectorSize;
            const keys: AnimKeyframe[] = new Array(this.Frames.length);

            for (let i = 0; i < keys.length; ++i) {
                const keyframe: AnimKeyframe = {
                    Frame: this.Frames[i],
                    Vector: this.Values.subarray(i * size, (i + 1) * size)
                };

                if (this.InTans) {
                    keyframe.InTan = this.InTans.subarray(i * size, (i + 1) * size);
                    keyframe.OutTan = this.OutTans.subarray(i * size, (i + 1) * size);
                }

                keys[i] = keyframe;
            }

            this.keysCache = keys;
        }

        return this.keysCache;
    }

    public set Keys (value: AnimKeyframe[]) {
        const size = this.VectorSize;
        const hasTans = value.length > 0 ? Boolean(value[0].InTan) : Boolean(this.InTans);

        const frames = new Int32Array(value.length);
        const values = allocateLike(this.Values, value.length * size);
        const inTans = hasTans ? allocateLike(this.Values, value.length * size) : undefined;
        const outTans = hasTans ? allocateLike(this.Values, value.length * size) : undefined;

        for (let i = 0; i < value.length; ++i) {
            frames[i] = value[i].Frame;
            values.set(value[i].Vector, i * size);
            if (hasTans) {
                inTans.set(value[i].InTan, i * size);
                outTans.set(value[i].OutTan, i * size);
            }
        }

        this.Frames = frames;
        this.Values = values;
        this.InTans = inTans;
        this.OutTans = outTans;
        // The incoming keyframes point at the caller's arrays, not the storage just built, so
        // they have to be rebuilt as views on next read.
        this.keysCache = null;
    }
}

/**
 * Build an AnimVector over flat keyframe storage. `values`, `inTans` and `outTans` hold
 * `vectorSize` components per keyframe, laid out end to end and in the same order as `frames`.
 */
export function createAnimVector (
    lineType: LineType,
    globalSeqId: number,
    vectorSize: number,
    frames: Int32Array,
    values: Float32Array|Int32Array,
    inTans?: Float32Array|Int32Array,
    outTans?: Float32Array|Int32Array
): AnimVector {
    return new AnimVectorImpl(lineType, globalSeqId, vectorSize, frames, values, inTans, outTans);
}

/** Replace an AnimVector's keyframes, rebuilding its flat storage to match. */
export function setAnimVectorKeys (animVector: AnimVector, value: AnimKeyframe[]): void {
    animVector.Keys = value;
}

export enum LayerShading {
    Unshaded = 1,
    SphereEnvMap = 2,
    TwoSided = 16,
    Unfogged = 32,
    NoDepthTest = 64,
    NoDepthSet = 128
}

export interface Layer {
    FilterMode?: FilterMode;
    Shading?: number;
    TextureID?: AnimVector|number;
    TVertexAnimId?: number;
    CoordId: number;
    Alpha?: AnimVector|number;
    /* Since Version: 900 */
    EmissiveGain?: AnimVector|number;
    /* Since Version: 1000 */
    FresnelColor?: AnimVector|Float32Array;
    /* Since Version: 1000 */
    FresnelOpacity?: AnimVector|number;
    /* Since Version: 1000 */
    FresnelTeamColor?: AnimVector|number;
    /* Since version: 1100 */
    ShaderTypeId?: number;
    NormalTextureID?: AnimVector|number;
    ORMTextureID?: AnimVector|number;
    EmissiveTextureID?: AnimVector|number;
    TeamColorTextureID?: AnimVector|number;
    ReflectionsTextureID?: AnimVector|number;
}

export enum MaterialRenderMode {
    ConstantColor = 1,
    SortPrimsFarZ = 16,
    FullResolution = 32,
}

export interface Material {
    PriorityPlane?: number;
    RenderMode?: number;
    Layers: Layer[];
    /* Since Version: 900 */
    Shader?: string;
}

export interface GeosetAnimInfo {
    MinimumExtent: Float32Array;
    MaximumExtent: Float32Array;
    BoundsRadius: number;
}

export interface Geoset {
    Vertices: Float32Array;
    Normals: Float32Array;
    TVertices: Float32Array[];
    VertexGroup: Uint8Array;
    Faces: Uint16Array;
    Groups: number[][];
    TotalGroupsCount: number;
    MinimumExtent: Float32Array;
    MaximumExtent: Float32Array;
    BoundsRadius: number;
    Anims: GeosetAnimInfo[];
    MaterialID: number;
    SelectionGroup: number;
    Unselectable: boolean;
    /* Since Version: 900 */
    LevelOfDetail?: number;
    /* Since Version: 900 */
    Name?: string;
    /* Since Version: 900 */
    Tangents?: Float32Array;
    /* Since Version: 900 */
    SkinWeights?: Uint8Array;
}

export enum GeosetAnimFlags {
    DropShadow = 1,
    Color = 2
}

export interface GeosetAnim {
    GeosetId: number;
    Alpha: AnimVector|number;
    Color: AnimVector|Float32Array;
    Flags: number;
}

export enum NodeFlags {
    DontInheritTranslation = 1,
    DontInheritRotation = 2,
    DontInheritScaling = 4,
    Billboarded = 8,
    BillboardedLockX = 16,
    BillboardedLockY = 32,
    BillboardedLockZ = 64,
    CameraAnchored = 128
}

export enum NodeType {
    Helper = 0,
    Bone = 256,
    Light = 512,
    EventObject = 1024,
    Attachment = 2048,
    ParticleEmitter = 4096, // ParticleEmitter | ParticleEmitter2 | ParticleEmitterPopcorn
    CollisionShape = 8192,
    RibbonEmitter = 16384
}

export interface Node {
    Name: string;
    ObjectId: number;
    Parent?: number|null;
    PivotPoint: Float32Array;
    Flags: number;

    Translation?: AnimVector;
    Rotation?: AnimVector;
    Scaling?: AnimVector;
}

export interface Bone extends Node {
    GeosetId?: number;
    GeosetAnimId?: number;
}

export type Helper = Node

export interface Attachment extends Node {
    Path?: string;
    AttachmentID?: number;
    Visibility?: AnimVector;
}

export interface EventObject extends Node {
    EventTrack: Uint32Array;
}

export enum CollisionShapeType {
    Box = 0,
    Sphere = 2
}

export interface CollisionShape extends Node {
    Shape: CollisionShapeType;
    Vertices: Float32Array;
    BoundsRadius?: number;
}

export enum ParticleEmitterFlags {
    EmitterUsesMDL = 32768,
    EmitterUsesTGA = 65536
}

export interface ParticleEmitter extends Node {
    EmissionRate: AnimVector|number;
    Gravity: AnimVector|number;
    Longitude: AnimVector|number;
    Latitude: AnimVector|number;
    Path: string;
    LifeSpan: AnimVector|number;
    InitVelocity: AnimVector|number;
    Visibility: AnimVector;
}

export enum ParticleEmitter2Flags {
    Unshaded = 32768,
    SortPrimsFarZ = 65536,
    LineEmitter = 131072,
    Unfogged = 262144,
    ModelSpace = 524288,
    XYQuad = 1048576
}

export enum ParticleEmitter2FilterMode {
    Blend = 0,
    Additive = 1,
    Modulate = 2,
    Modulate2x = 3,
    AlphaKey = 4
}

// Not actually mapped to mdx flags (0: Head, 1: Tail, 2: Both)
export enum ParticleEmitter2FramesFlags {
    Head = 1,
    Tail = 2
}

export interface ParticleEmitter2 extends Node {
    Speed?: AnimVector|number;
    Variation?: AnimVector|number;
    Latitude?: AnimVector|number;
    Gravity?: AnimVector|number;
    Visibility?: AnimVector|number;
    Squirt?: boolean;
    LifeSpan?: number;
    EmissionRate?: AnimVector|number;
    Width?: AnimVector|number;
    Length?: AnimVector|number;
    FilterMode?: ParticleEmitter2FilterMode;
    Rows?: number;
    Columns?: number;
    FrameFlags: number;
    TailLength?: number;
    Time?: number;
    SegmentColor?: Float32Array[];
    Alpha?: Uint8Array;
    ParticleScaling?: Float32Array;
    LifeSpanUVAnim?: Uint32Array;
    DecayUVAnim?: Uint32Array;
    TailUVAnim?: Uint32Array;
    TailDecayUVAnim?: Uint32Array;
    TextureID?: number;
    ReplaceableId?: number;
    PriorityPlane?: number;
}

export interface Camera {
    Name: string;
    Position: Float32Array;
    FieldOfView: number;
    NearClip: number;
    FarClip: number;
    TargetPosition: Float32Array;
    TargetTranslation?: AnimVector;
    Translation?: AnimVector;
    Rotation?: AnimVector;
}

export enum LightType {
    Omnidirectional = 0,
    Directional = 1,
    Ambient = 2
}

export interface Light extends Node {
    LightType: LightType;

    AttenuationStart?: AnimVector|number;
    AttenuationEnd?: AnimVector|number;

    Color?: AnimVector|Float32Array;
    Intensity?: AnimVector|number;
    AmbIntensity?: AnimVector|number;
    AmbColor?: AnimVector|Float32Array;

    Visibility?: AnimVector|number;
}

export interface RibbonEmitter extends Node {
    HeightAbove?: AnimVector|number;
    HeightBelow?: AnimVector|number;
    Alpha?: AnimVector|number;
    // todo support KRCO
    Color?: Float32Array;
    LifeSpan?: number;
    TextureSlot?: AnimVector|number;
    EmissionRate?: number;
    Rows?: number;
    Columns?: number;
    MaterialID?: number;
    Gravity?: number;

    Visibility?: AnimVector;
}

export interface TVertexAnim {
    Translation?: AnimVector;
    Rotation?: AnimVector;
    Scaling?: AnimVector;
}

/* Since Version: 900 */
export interface FaceFX {
    Name: string;
    Path: string;
}

/* Since Version: 900 */
export interface BindPose {
    Matrices: Float32Array[];
}

/* Since Version: 900 */
export enum ParticleEmitterPopcornFlags {
    Unshaded = 32768,
    SortPrimsFarZ = 65536,
    Unfogged = 262144
}

/* Since Version: 900 */
export interface ParticleEmitterPopcorn extends Node {
    LifeSpan?: AnimVector|number;
    EmissionRate?: AnimVector|number;
    Speed?: AnimVector|number;
    Color?: AnimVector|Float32Array;
    Alpha?: AnimVector|number;
    ReplaceableId?: number;
    Path?: string;
    AnimVisibilityGuide?: string;
    Visibility?: AnimVector;
}

export interface Model {
    Version: number;
    Info: ModelInfo;
    Sequences: Sequence[];
    Textures: Texture[];
    Materials: Material[];
    Geosets: Geoset[];
    GeosetAnims: GeosetAnim[];
    Bones: Bone[];
    Helpers: Helper[];
    Attachments: Attachment[];
    Nodes: Node[];
    PivotPoints: Float32Array[];
    EventObjects: EventObject[];
    CollisionShapes: CollisionShape[];
    GlobalSequences: number[];
    ParticleEmitters: ParticleEmitter[];
    ParticleEmitters2: ParticleEmitter2[];
    Cameras: Camera[];
    Lights: Light[];
    RibbonEmitters: RibbonEmitter[];
    TextureAnims: TVertexAnim[];
    /* Since Version: 900 */
    FaceFX?: FaceFX[];
    /* Since Version: 900 */
    BindPoses?: BindPose[];
    /* Since Version: 900 */
    ParticleEmitterPopcorns?: ParticleEmitterPopcorn[];
}
