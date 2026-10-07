/* global document */
import assert from 'node:assert/strict';
import {createServer} from 'vite';
import {chromium} from 'playwright';

const server = await createServer({configFile: false, server: {host: '127.0.0.1', port: 0, watch: null, hmr: false}});
let browser;
try {
    await server.listen();
    browser = await chromium.launch({headless: true, args: process.argv.includes('--webgpu') ?
        ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] : []});
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/test/browser-renderer.html`);
    await page.waitForFunction(() => document.body.dataset.result, undefined, {timeout: 60000});
    const report = await page.locator('#status').textContent();
    console.log(report);
    assert.equal(await page.locator('body').getAttribute('data-result'), 'pass', report);
} finally {
    await browser?.close();
    await server.close();
}
