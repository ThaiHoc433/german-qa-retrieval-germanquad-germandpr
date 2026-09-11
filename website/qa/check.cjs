// QA only; not loaded by the website. Usage: node qa/check.cjs <playwright-module-path>
const { chromium } = require(process.argv[2] || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const url = pathToFileURL(path.join(root, 'index.html')).href;

(async () => {
  const browser = await chromium.launch({headless: true});
  const report = { url: 'index.html (file://)', viewports: [], consoleErrors: [] };
  try {
    for (const width of [375, 768, 1440]) {
      const page = await browser.newPage({ viewport: {width, height: 1000}, reducedMotion: 'reduce' });
      page.on('pageerror', error => report.consoleErrors.push(error.message));
      await page.goto(url);
      await page.evaluate(() => document.fonts.ready);
      const inspect = () => page.evaluate(() => ({
        width: innerWidth,
        documentWidth: document.documentElement.scrollWidth,
        overflow: Array.from(document.querySelectorAll('main *')).filter(el => {
          const r = el.getBoundingClientRect();
          return r.width && (r.right > innerWidth + 1 || r.left < -1);
        }).map(el => `${el.tagName}.${el.className}`).slice(0, 20),
        missingLinks: Array.from(document.querySelectorAll('a[href^="#"]')).filter(a => !document.getElementById(a.hash.slice(1))).map(a=>a.hash),
        duplicateIds: Array.from(document.querySelectorAll('[id]')).map(el=>el.id).filter((id,i,ids)=>ids.indexOf(id)!==i),
        brokenAssets: Array.from(document.images).filter(i => !i.complete || i.naturalWidth === 0).length
      }));
      const initial = await inspect();
      assert.equal(initial.documentWidth, width, `horizontal overflow ${width}`);
      assert.deepEqual(initial.overflow, []);
      assert.deepEqual(initial.missingLinks, []);
      assert.deepEqual(initial.duplicateIds, []);
      assert.equal(await page.locator('.walk-step:visible').count(), 1);
      assert.equal(await page.locator('#previous').isDisabled(), true);
      for (let i=1;i<5;i++) {
        await page.locator('#next').click();
        assert.equal(await page.locator(`#walk-${i}`).isVisible(),true);
      }
      assert.equal(await page.locator('#next').isDisabled(),true);
      await page.locator('#previous').click();
      assert.equal(await page.locator('#walk-3').isVisible(),true);
      for(let i=0;i<5;i++) {
        await page.locator(`[data-step="${i}"]`).click();
        assert.equal(await page.locator(`#walk-${i}`).isVisible(),true);
      }
      await page.locator('[data-step="0"]').click();
      const summaries = page.locator('summary');
      for (let i=0;i<await summaries.count();i++) {
        const summary=summaries.nth(i);
        await summary.focus();
        await page.keyboard.press('Enter');
        assert.equal(await summary.evaluate(s => s.parentElement.open),true);
        await page.keyboard.press('Space');
        assert.equal(await summary.evaluate(s => s.parentElement.open),false);
      }
      for(const id of ['overview','architecture','data','evaluation','modules','roadmap']) {
        await page.locator(`.nav-links a[href="#${id}"]`).click();
        await page.waitForTimeout(100);
        assert.equal(await page.locator(`.nav-links a[href="#${id}"]`).getAttribute('aria-current'),'location');
      }
      await page.locator('.back-top').click();
      await page.waitForTimeout(100);
      assert.ok(await page.evaluate(()=>scrollY)<5);
      await page.screenshot({path:path.join(__dirname,`${width}-full.png`),fullPage:true});
      await page.screenshot({path:path.join(__dirname,`${width}-hero.png`)});
      await page.locator('.nav-links a[href="#architecture"]').click();
      await page.waitForTimeout(100);
      await page.screenshot({path:path.join(__dirname,`${width}-pipeline.png`)});
      if(width===375 || width===1440) {
        for (const id of ['data','modules','errors']) {
          // Isolated section capture: suppress sticky/fixed chrome only in the screenshot.
          await page.locator(`#${id}`).screenshot({path:path.join(__dirname,`${width}-${id}.png`), style:'.site-header, .skip-link { visibility: hidden !important; }'});
        }
      }
      await page.evaluate(()=>document.querySelectorAll('details').forEach(d=>d.open=true));
      const expanded=await inspect();
      assert.deepEqual(expanded.overflow,[],`expanded overflow ${width}`);
      assert.equal(expanded.documentWidth,width);
      report.viewports.push({width,initial,expanded,walkthrough:'passed all steps, next/previous and endpoints',keyboardDetails:await summaries.count(),navigation:'all six anchors + back-to-top passed'});
      await page.close();
    }
    const noJS = await browser.newPage({javaScriptEnabled:false,viewport:{width:375,height:1000},reducedMotion:'reduce'});
    await noJS.goto(url);
    assert.equal(await noJS.locator('.walk-step:visible').count(),5);
    assert.equal(await noJS.locator('.walk-controls').isVisible(),false);
    await noJS.locator('.technical summary').first().click();
    assert.equal(await noJS.locator('.technical details').first().getAttribute('open'),'');
    await noJS.screenshot({path:path.join(__dirname,'375-no-js.png'),fullPage:true});
    report.noJavaScript = 'All 5 walkthrough steps visible; native details opens; content available.';
    const zoom = await browser.newPage({viewport:{width:768,height:1000}});
    await zoom.goto(url);
    await zoom.addStyleTag({content:'html { font-size: 200%; }'});
    console.log('text enlargement diagnostics', await zoom.evaluate(()=>Array.from(document.querySelectorAll('body *')).filter(e=>{const r=e.getBoundingClientRect();return r.width && (r.right>innerWidth+1 || r.left< -1)}).map(e=>({tag:e.tagName,cls:e.className,text:e.textContent.slice(0,80),width:e.getBoundingClientRect().width})).slice(0,20)));
    assert.equal(await zoom.evaluate(()=>document.documentElement.scrollWidth),768,'200% text overflow');
    report.textEnlargement = '200% root font at 768px, no page horizontal overflow';
    assert.deepEqual(report.consoleErrors,[]);
    fs.writeFileSync(path.join(__dirname,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
