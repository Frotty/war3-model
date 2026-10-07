// Opt-in integration test: proprietary game files stay on the local machine.
/* global window, fetch */
import {Buffer} from 'node:buffer';
import {performance} from 'node:perf_hooks';
import {createRequire} from 'node:module';
import {resolve, dirname, basename} from 'node:path';
import {mkdir, writeFile, open} from 'node:fs/promises';
import {createServer} from 'node:http';
import {fileURLToPath, URL} from 'node:url';
import {chromium} from 'playwright';
import {createServer as createViteServer} from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const parseOnly = args.includes('--parse-only');
const option = (name, fallback) => {
    const at = args.indexOf(`--${name}`);
    if (at >= 0 && (!args[at + 1] || args[at + 1].startsWith('--'))) throw new Error(`Missing value for --${name}`);
    return at >= 0 ? args[at + 1] : fallback;
};
const game = option('game', process.env.WC3_GAME_PATH);
if (!game) throw new Error('Set WC3_GAME_PATH or pass --game <Warcraft III directory>.');
const limit = Number(option('limit', Infinity));
const size = Number(option('size', 128));
const minCoverage = Number(option('min-coverage', 0.02));
const minSpan = Number(option('min-span', 0.5));
if (!(limit > 0 && (limit === Infinity || Number.isInteger(limit))) || !Number.isInteger(size) || size < 16 || size > 2048 ||
    !(minCoverage >= 0 && minCoverage <= 1) || !(minSpan >= 0 && minSpan <= 1)) throw new Error('Invalid test options.');
const output = resolve(option('output', resolve(root, parseOnly ? '_build/game-data-parser' : '_build/game-data-thumbnails')));
const requireCasc = createRequire(resolve(option('casc', resolve(root, '../casc-ts')), 'package.json'));
const {CascStorage, closeAllSegments} = requireCasc('./dist/index.js');
const {decodeTga} = requireCasc('./dist/formats/index.js');
const storage = await CascStorage.openAsync(resolve(game), console.log);
let assetServer, vite, browser, journal;
let models = [], selected = [];
const results = [];
try {
    await mkdir(output, {recursive: true});
    const expanded = new Set();
    // listFiles initially contains only TVFS roots. Expand every discovered game namespace,
    // including SD, HD, locales and optional variants before taking the model census.
    for (;;) {
        const containers = storage.listFiles().filter(path => /\.w3mod$/i.test(path) && !expanded.has(path));
        if (!containers.length) break;
        for (const path of containers) {
            expanded.add(path);
            try { await storage.readFileAsync(path); }
            catch (error) { console.warn(`Container unavailable: ${path}: ${error.message}`); }
        }
    }
    models = storage.listFiles().filter(path => /\.(?:mdx|mdl)$/i.test(path));
    if (!models.length) throw new Error('No models discovered; the game archive census was empty.');
    selected = models.filter(path => path.toLowerCase().includes(String(option('filter', '')).toLowerCase())).slice(0, limit);
    if (!selected.length) throw new Error('No models match the requested filter.');
    console.log(`Discovered ${models.length} models; testing ${selected.length}. Output: ${output}`);
    let currentPath = '';
    let modelBuffer;
    const missing = new Set();
    const textureSources = new Map();
    let sourceBytes = 0;
    function candidates(path) {
        const normalized = path.replace(/\\/g, '/').replace(/^\/+/, '');
        const parts = currentPath.slice(0, currentPath.lastIndexOf(':')).replace(/^\/+/, '').split(':');
        const prefixes = parts.map((_, index) => parts.slice(0, parts.length - index).join(':') + ':');
        const variants = /\.[^./]+$/i.test(normalized)
            ? [normalized, ...['dds', 'blp', 'tga'].map(ext => normalized.replace(/\.[^.]+$/, `.${ext}`))] : [normalized];
        return [...new Set(prefixes.flatMap(prefix => variants.map(value => prefix + value)))];
    }
    journal = await open(resolve(output, 'results.jsonl'), 'w');
    const parser = parseOnly ? await import('../dist/es/war3-model.mjs') : undefined;
    let page;
    if (!parseOnly) {
        assetServer = createServer(async (req, res) => {
            try {
                const url = new URL(req.url, 'http://localhost');
                if (url.pathname === '/model') { res.end(modelBuffer); return; }
                if (url.pathname !== '/texture') { res.writeHead(404).end(); return; }
                const path = url.searchParams.get('path');
                const key = JSON.stringify([currentPath.slice(0, currentPath.lastIndexOf(':')), path]);
                let source = textureSources.get(key);
                if (!source) {
                    for (const candidate of candidates(path)) {
                        try { source = {bytes: await storage.readFileAsync(candidate), path: candidate}; break; } catch { /* next extension/root */ }
                    }
                    // Bound the host-side compressed source cache as well as the session GPU cache.
                    if (source) {
                        if (source.bytes.length <= 64 * 1024 * 1024) {
                            textureSources.set(key, source);
                            sourceBytes += source.bytes.length;
                        }
                        while (textureSources.size > 256 || sourceBytes > 64 * 1024 * 1024) {
                            const oldest = textureSources.keys().next().value;
                            sourceBytes -= textureSources.get(oldest).bytes.length;
                            textureSources.delete(oldest);
                        }
                    }
                }
                if (!source) { missing.add(path); res.writeHead(404).end(); return; }
                res.setHeader('X-Texture-Path', source.path);
                if (/\.tga$/i.test(source.path)) {
                    const image = decodeTga(source.bytes);
                    res.setHeader('X-Texture-Raster', 'true');
                    res.end(JSON.stringify(image));
                    return;
                }
                res.end(source.bytes);
            } catch (error) { res.writeHead(500).end(error.message); }
        });
        await new Promise(resolve => assetServer.listen(0, '127.0.0.1', resolve));
        const assetOrigin = `http://127.0.0.1:${assetServer.address().port}`;
        vite = await createViteServer({root, configFile: false, server: {host: '127.0.0.1', port: 0, watch: null, hmr: false},
            plugins: [{name: 'game-assets', configureServer(server) {
                server.middlewares.use('/game-assets', (req, res) => {
                    fetch(assetOrigin + req.url).then(async response => {
                        res.statusCode = response.status;
                        const source = response.headers.get('X-Texture-Path');
                        if (source) res.setHeader('X-Texture-Path', source);
                        const raster = response.headers.get('X-Texture-Raster');
                        if (raster) res.setHeader('X-Texture-Raster', raster);
                        res.end(Buffer.from(await response.arrayBuffer()));
                    }).catch(error => { res.statusCode = 500; res.end(error.message); });
                });
            }}]});
        await vite.listen();
        browser = await chromium.launch({headless: true});
        page = await browser.newPage();
        await page.goto(`http://127.0.0.1:${vite.httpServer.address().port}/test/game-data-thumbnails.html`);
        await page.waitForFunction(() => typeof window.captureGameModel === 'function');
    }
    for (const [index, path] of selected.entries()) {
        currentPath = path;
        missing.clear();
        const started = performance.now();
        let result;
        try {
            modelBuffer = await storage.readFileAsync(path);
            if (parseOnly) {
                const model = /\.mdl$/i.test(path) ? parser.parseMDL(modelBuffer.toString('utf8')) :
                    parser.parseMDX(modelBuffer.buffer.slice(modelBuffer.byteOffset, modelBuffer.byteOffset + modelBuffer.byteLength));
                result = {pass: true, version: model.Version};
            } else result = await page.evaluate(options => window.captureGameModel(options), {path, size, minCoverage, minSpan});
            if (result.png) {
                const filename = `${String(index).padStart(5, '0')}-${basename(path.replace(/\\/g, '/')).replace(/[^a-z0-9_.-]/gi, '_')}.png`;
                await writeFile(resolve(output, filename), Buffer.from(result.png, 'base64'));
                delete result.png;
                result.thumbnail = filename;
            }
        } catch (error) { result = {pass: false, error: error.message}; }
        result = {path, ...result, missingTextures: [...missing], ms: Math.round(performance.now() - started)};
        results.push(result);
        console.log(`[${index + 1}/${selected.length}] ${result.pass ? 'PASS' : 'FAIL'} ${path} ${result.coverage !== undefined ? `${(result.coverage * 100).toFixed(1)}% pixels` : result.version !== undefined ? `v${result.version}` : result.error}`);
        // Append rather than rewriting a growing report once per model (quadratic disk traffic).
        await journal.write(JSON.stringify(result) + '\n');
    }
} finally {
    const cleanup = await Promise.allSettled([browser?.close(), vite?.close(),
        assetServer ? new Promise(resolve => assetServer.close(resolve)) : undefined, closeAllSegments(), journal?.close()]);
    for (const result of cleanup) if (result.status === 'rejected') console.warn('Cleanup failed:', result.reason);
    if (journal) await writeFile(resolve(output, 'results.json'), JSON.stringify({game: resolve(game), discovered: models.length,
        tested: results.length, selected: selected.length, parseOnly, size, minCoverage, minSpan, results}, null, 2));
}
const failed = results.filter(result => !result.pass);
console.log(`${results.length - failed.length} passed, ${failed.length} failed. Report: ${resolve(output, 'results.json')}`);
if (failed.length) process.exitCode = 1;
