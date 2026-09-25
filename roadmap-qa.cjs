const {chromium}=require('C:/Users/PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true});
 const errors=[];
 try{
  for(const width of [380,768,1440]){
   const context=await browser.newContext({viewport:{width,height:1000}});
   const page=await context.newPage();
   page.on('pageerror',e=>errors.push(e.message));
   page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
   await page.goto(pathToFileURL(path.resolve('roadmap-preview.html')).href);
   assert.equal(await page.locator('[data-lesson]').count(),32);
   assert.deepEqual(await page.locator('[data-lesson]').evaluateAll(es=>es.map(e=>Number(e.dataset.lesson))),Array.from({length:32},(_,i)=>i+1));
   assert.equal(await page.locator('.stage-detail').count(),10);
   assert.equal(await page.locator('.week-block').count(),12);
   assert.equal(await page.locator('#tuan-11 .lesson').count(),2);
   assert.equal(await page.locator('#tuan-12 .lesson').count(),0);
   assert.equal(await page.locator('input:checked').count(),0);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),width);
   await page.locator('[data-complete="1"]').check();
   await page.reload();
   assert.equal(await page.locator('[data-complete="1"]').isChecked(),true);
   await page.locator('#stage-filter').selectOption('5');
   assert.equal(await page.locator('.lesson:visible').count(),6);
   await page.locator('#clear').click();
   await page.locator('#env-filter').selectOption('T4');
   assert.equal(await page.locator('.lesson:visible').count(),10);
   await page.locator('#jump').selectOption('11');
   assert.equal(await page.locator('.lesson:visible').count(),32);
   await page.locator('#chang-8 summary').focus();
   await page.keyboard.press('Enter');
   assert.equal(await page.locator('#chang-8').getAttribute('open'),'');
   await page.keyboard.press('Space');
   assert.equal(await page.locator('#chang-8').getAttribute('open'),null);
   await page.emulateMedia({colorScheme:'dark'});
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),width);
   if(width===380)await page.screenshot({path:'roadmap-mobile.png',fullPage:false});
   if(width===1440){
    await page.emulateMedia({colorScheme:'light'});
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.screenshot({path:'roadmap-desktop.png',fullPage:false});
    await page.locator('#stage-filter').selectOption('5');
    await page.evaluate(()=>dispatchEvent(new Event('beforeprint')));
    assert.equal(await page.locator('.lesson:visible').count(),32);
    assert.equal(await page.locator('.stage-detail[open]').count(),10);
    await page.emulateMedia({media:'print'});
    await page.pdf({path:'roadmap-print.pdf'});
    await page.evaluate(()=>dispatchEvent(new Event('afterprint')));
    await page.emulateMedia({media:'screen'});
    assert.equal(await page.locator('.lesson:visible').count(),6);
   }
   await context.close();
  }
  const context=await browser.newContext();
  await context.addInitScript(()=>Object.defineProperty(window,'localStorage',{get(){throw Error('blocked');}}));
  const page=await context.newPage();
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve('roadmap-preview.html')).href);
  await page.locator('[data-complete="1"]').check();
  assert.equal(await page.locator('#progress').getAttribute('value'),'1');
  assert.match(await page.locator('#storage-note').textContent(),/phiên hiện tại/);
  assert.deepEqual(errors,[]);
  console.log('PASS: 32 lessons, 10 stages, 12 weeks, 380/768/1440px, dark mode, filters, keyboard, persistence, blocked storage, print restore; no console errors.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
