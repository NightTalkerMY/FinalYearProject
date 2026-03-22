import puppeteer from 'puppeteer';

(async () => {
  console.log("Launching Headless Hologram Engine with GPU Force...");

  const browser = await puppeteer.launch({
    headless: "new", 
    args: [
      '--autoplay-policy=no-user-gesture-required',
      '--ignore-gpu-blocklist',        // Force Chrome to use GPU even if it thinks it shouldn't
      '--enable-gpu-rasterization',
      '--enable-webgl',
      '--use-gl=angle',                // Use the Windows ANGLE graphics layer
      '--use-angle=d3d11',             // Specifically use Direct3D 11 for Windows
      '--disable-dev-shm-usage',
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      '--no-sandbox',
    ]
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 }); // WAS 1280,720

  // Navigate to your React App
  await page.goto('http://localhost:5173');

  console.log("Rendering Engine initialized.");
  
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes("STREAM") || text.includes("tracks")) {
      console.log('BROWSER:', text);
    }
  });

  page.on('pageerror', err => console.error('BROWSER ERROR:', err));
})();


// import puppeteer from 'puppeteer';
// import fs from 'fs';
// import path from 'path';

// (async () => {
//   console.log("Launching Headless Hologram Engine with GPU Force...");

//   // Put all Chromium profile/cache data on SSD
//   const userDataDir = 'D:/FinalYearProject/puppeteer-data';
//   const diskCacheDir = 'D:/FinalYearProject/puppeteer-cache';

//   fs.mkdirSync(userDataDir, { recursive: true });
//   fs.mkdirSync(diskCacheDir, { recursive: true });

//   const browser = await puppeteer.launch({
//     headless: "new",
//     userDataDir,
//     args: [
//       '--autoplay-policy=no-user-gesture-required',

//       // GPU / rendering
//       '--ignore-gpu-blocklist',
//       '--enable-gpu-rasterization',
//       '--enable-webgl',
//       '--use-gl=angle',
//       '--use-angle=d3d11',

//       // Performance / stability
//       '--disable-dev-shm-usage',
//       '--disable-background-timer-throttling',
//       '--disable-backgrounding-occluded-windows',
//       '--disable-renderer-backgrounding',
//       '--no-sandbox',

//       // Force disk cache away from C:
//       `--disk-cache-dir=${diskCacheDir}`,

//       // Optional: reduce extra browser noise
//       '--no-first-run',
//       '--no-default-browser-check',
//       '--disable-features=Translate,BackForwardCache',
//     ]
//   });

//   const page = await browser.newPage();
//   await page.setViewport({ width: 1280, height: 720 });

//   page.on('console', msg => {
//     const text = msg.text();
//     if (text.includes("STREAM") || text.includes("tracks")) {
//       console.log('BROWSER:', text);
//     }
//   });

//   page.on('pageerror', err => console.error('BROWSER ERROR:', err));

//   await page.goto('http://localhost:5173', {
//     waitUntil: 'networkidle2'
//   });

//   console.log("Rendering Engine initialized.");
// })();