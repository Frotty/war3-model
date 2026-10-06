export interface WaitOptions {
    signal?: AbortSignal;
    timeoutMs?: number;
}

export function abortError (): Error {
    const error = new Error('Rendering was aborted');
    error.name = 'AbortError';
    return error;
}

/** Add cancellation without leaving a timeout or abort listener attached after settlement. */
export function cancellable<T> (promise: Promise<T>, {signal, timeoutMs}: WaitOptions = {}): Promise<T> {
    if (signal?.aborted) return Promise.reject(abortError());
    if (timeoutMs !== undefined && (!Number.isFinite(timeoutMs) || timeoutMs <= 0)) {
        return Promise.reject(new Error('timeoutMs must be positive'));
    }
    return new Promise<T>((resolve, reject) => {
        let timer: ReturnType<typeof setTimeout>;
        const abort = (): void => finish(undefined, abortError(), true);
        const finish = (value?: T, error?: unknown, failed = false): void => {
            clearTimeout(timer);
            signal?.removeEventListener('abort', abort);
            if (failed) reject(error); else resolve(value);
        };
        signal?.addEventListener('abort', abort, {once: true});
        if (timeoutMs !== undefined) timer = setTimeout(() => finish(undefined, new Error('Rendering timed out'), true), timeoutMs);
        promise.then(value => finish(value), error => finish(undefined, error, true));
    });
}

/** A fence waits for submitted work rather than guessing a number of animation frames. */
export function waitForGL (gl: WebGLRenderingContext | WebGL2RenderingContext, options: WaitOptions = {}): Promise<void> {
    if (gl.isContextLost()) return Promise.reject(new Error('WebGL context was lost'));
    if (!('fenceSync' in gl)) {
        gl.finish();
        return Promise.resolve();
    }
    const gl2 = gl as WebGL2RenderingContext;
    const fence = gl2.fenceSync(gl2.SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!fence) return Promise.reject(new Error('Unable to create WebGL completion fence'));
    gl2.flush();
    let timer: ReturnType<typeof setTimeout>;
    let stopped = false;
    const work = new Promise<void>((resolve, reject) => {
        const poll = (): void => {
            if (stopped) return;
            const status = gl2.clientWaitSync(fence, 0, 0);
            if (gl2.isContextLost() || status === gl2.WAIT_FAILED) {
                reject(new Error('WebGL work did not complete'));
            } else if (status === gl2.TIMEOUT_EXPIRED) {
                timer = setTimeout(poll, 4);
            } else {
                resolve();
            }
        };
        poll();
    });
    return cancellable(work, options).finally(() => {
        stopped = true;
        clearTimeout(timer);
        gl2.deleteSync(fence);
    });
}
