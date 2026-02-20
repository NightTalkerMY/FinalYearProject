import puppeteer from 'puppeteer';

(async () => {
  console.log("🚀 Launching Headless Hologram Engine with GPU Force...");

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

  console.log("✅ Rendering Engine initialized.");
  
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes("STREAM") || text.includes("tracks")) {
      console.log('BROWSER:', text);
    }
  });

  page.on('pageerror', err => console.error('BROWSER ERROR:', err));
})();
