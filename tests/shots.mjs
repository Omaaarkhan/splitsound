import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { webkit } from "playwright-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const distDir = path.join(rootDir, "dist");
const shotsDir = path.join(rootDir, "test-results", "shots");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".json": "application/json",
};

const STRICT_CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self'",
  "font-src 'self'",
  "img-src 'self' data:",
  "connect-src 'self'",
].join("; ");

const SCENES = [
  { name: "home", width: 1240, height: 800 },
  { name: "home-empty", width: 1240, height: 800 },
  { name: "home-first-run", width: 1240, height: 800 },
  { name: "imported", width: 1240, height: 800 },
  { name: "consent", width: 1240, height: 800 },
  { name: "scan-downloading", width: 1240, height: 800 },
  { name: "scan-unavailable", width: 1240, height: 800 },
  { name: "splitting", width: 1240, height: 800 },
  { name: "ready", width: 1240, height: 800 },
  { name: "ready-solo", width: 1240, height: 800 },
  { name: "loop", width: 1240, height: 800 },
  { name: "export", width: 1240, height: 800 },
  { name: "settings", width: 1240, height: 800 },
  { name: "update", width: 1240, height: 800 },
  { name: "error", width: 1240, height: 800 },
  { name: "ready", fileName: "ready-980x640", width: 980, height: 640 },
];

function startStaticServer() {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const urlPath = new URL(req.url || "/", "http://127.0.0.1").pathname;
      const safePath = path.normalize(urlPath).replace(/^(\.\.[/\\])+/, "");
      let filePath = path.join(distDir, safePath === "/" ? "index.html" : safePath);
      if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        filePath = path.join(distDir, "index.html");
      }
      const ext = path.extname(filePath).toLowerCase();
      res.writeHead(200, {
        "Content-Type": MIME[ext] || "application/octet-stream",
        "Content-Security-Policy": STRICT_CSP,
        "Cache-Control": "no-store",
      });
      fs.createReadStream(filePath).pipe(res);
    });
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      resolve({ server, port: addr.port });
    });
  });
}

async function main() {
  if (!process.argv.includes("--no-build")) {
    execFileSync("npm", ["run", "build"], { cwd: rootDir, stdio: "inherit" });
  }

  fs.mkdirSync(shotsDir, { recursive: true });
  const { server, port } = await startStaticServer();
  const baseUrl = `http://127.0.0.1:${port}`;
  const browser = await webkit.launch();

  const failures = [];

  try {
    const captureEntry = async (entry) => {
      const outName = entry.fileName ?? entry.name;
      const context = await browser.newContext({
        viewport: { width: entry.width, height: entry.height },
        deviceScaleFactor: 2,
      });
      const page = await context.newPage();

      const errors = [];
      let capturing = false;
      page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
      page.on("console", (msg) => {
        if (msg.type() === "error") {
          // Playwright WebKit injects a temporary stylesheet during page.screenshot() itself.
          // Ignore that harness-only message while failing on any CSP/console error during app execution.
          if (capturing && msg.text().includes("Refused to apply a stylesheet")) {
            return;
          }
          errors.push(`console.error: ${msg.text()}`);
        }
      });
      page.on("request", (req) => {
        const url = req.url();
        if (!url.startsWith(baseUrl) && !url.startsWith("data:")) {
          errors.push(`external request blocked: ${url}`);
        }
      });

      await page.goto(`${baseUrl}/?scene=${entry.name}`, { waitUntil: "networkidle" });
      await page.evaluate(async () => {
        await document.fonts.ready;
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      });
      // Allow ResizeObserver, async peaks decode, and 180ms panel entrance to settle
      await page.waitForTimeout(260);

      const outPath = path.join(shotsDir, `${outName}.png`);
      capturing = true;
      await page.screenshot({ path: outPath, caret: "initial" });
      capturing = false;
      console.log(`Captured ${outName}.png (${entry.width}x${entry.height} @2x)`);

      if (errors.length > 0) {
        failures.push({ scene: outName, errors });
      }
      await context.close();
    };

    const batchSize = 4;
    for (let i = 0; i < SCENES.length; i += batchSize) {
      await Promise.all(SCENES.slice(i, i + batchSize).map(captureEntry));
    }
  } finally {
    await browser.close();
    server.close();
  }

  if (failures.length > 0) {
    console.error(
      "Errors detected while capturing screenshots:",
      JSON.stringify(failures, null, 2),
    );
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
