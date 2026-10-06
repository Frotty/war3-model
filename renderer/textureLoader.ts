import type {DdsInfo} from 'dds-parser';
import type {Texture} from '../model';
import type {DDS_FORMAT} from './modelRenderer';

export type TextureSource =
    {type: 'imageData'; imageData: ImageData[]} |
    {type: 'image'; image: HTMLImageElement} |
    {type: 'blp'; buffer: ArrayBuffer} |
    {type: 'dds'; buffer: ArrayBuffer; format: DDS_FORMAT; info: DdsInfo};

/** Resolve the model's virtual texture path through the consumer's archive/file service. */
export type TextureLoader = (texture: Texture, index: number, signal?: AbortSignal) => Promise<TextureSource>;
