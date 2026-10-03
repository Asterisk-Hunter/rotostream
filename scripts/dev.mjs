#!/usr/bin/env node
/**
 * Dev launcher: runs the FastAPI backend and the Next.js frontend together with
 * prefixed, interleaved logs and a single Ctrl-C to stop both.
 *
 *   node scripts/dev.mjs               # both
 *   node scripts/dev.mjs --api-only
 *   node scripts/dev.mjs --web-only
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const isWin = process.platform === "win32";
const envPath = path.join(root, ".env");
if (existsSync(envPath)) process.loadEnvFile(envPath);

const args = new Set(process.argv.slice(2));
const apiOnly = args.has("--api-only");
const webOnly = args.has("--web-only");
if ((apiOnly && webOnly) || [...args].some((arg) => !["--api-only", "--web-only"].includes(arg))) {
  console.error("Usage: node scripts/dev.mjs [--api-only | --web-only]");
  process.exit(1);
}

// 8010, not 8000: plenty of other dev servers sit on 8000 and a silent port
// collision shows up as a confusing 404 from whatever is already listening.
const API_HOST = process.env.ROTOSTREAM_HOST ?? "127.0.0.1";
const API_PORT = Number(process.env.ROTOSTREAM_PORT ?? 8010);
const WEB_PORT = Number(process.env.WEB_PORT ?? 3000);
for (const [name, port] of [["ROTOSTREAM_PORT", API_PORT], ["WEB_PORT", WEB_PORT]]) {
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    console.error(`${name} must be an integer between 1 and 65535`);
    process.exit(1);
  }
}

/** True when something is already listening on host:port. */
function portInUse(port, host) {
  return new Promise((resolve) => {
    const probe = net.createServer();
    probe.once("error", (err) => {
      if (err.code === "EADDRINUSE") resolve(true);
      else {
        console.error(`Cannot bind ${host}:${port}: ${err.message}`);
        resolve(true);
      }
    });
    probe.once("listening", () => probe.close(() => resolve(false)));
    probe.listen(port, host);
  });
}

// Prefer the project venv, fall back to whatever `python` is on PATH.
const venvPython = isWin
  ? path.join(root, ".venv", "Scripts", "python.exe")
  : path.join(root, ".venv", "bin", "python");
const python = existsSync(venvPython) ? venvPython : isWin ? "python" : "python3";
if (!existsSync(venvPython)) {
  console.warn(`\x1b[33m[dev]\x1b[0m no venv at .venv - using "${python}" from PATH`);
}

const children = [];
let shuttingDown = false;

function pipeWithPrefix(stream, tag, isErr) {
  let buffer = "";
  stream.on("data", (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      process[isErr ? "stderr" : "stdout"].write(`${tag} ${line}\n`);
    }
  });
  stream.on("end", () => {
    if (buffer) process[isErr ? "stderr" : "stdout"].write(`${tag} ${buffer}\n`);
  });
}

function run(name, colour, command, commandArgs, cwd) {
  const tag = `\x1b[${colour}m[${name}]\x1b[0m`;
  const child = spawn(command, commandArgs, {
    cwd,
    env: {
      ...process.env,
      NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL ?? `http://${API_HOST === "0.0.0.0" ? "127.0.0.1" : API_HOST}:${API_PORT}`,
    },
    shell: false,
    detached: !isWin,
  });
  pipeWithPrefix(child.stdout, tag, false);
  pipeWithPrefix(child.stderr, tag, true);
  child.on("error", (err) => {
    process.stderr.write(`${tag} failed to start: ${err.message}\n`);
    shutdown(1);
  });
  child.on("exit", (code, signal) => {
    if (!shuttingDown) {
      process.stdout.write(`${tag} exited (code=${code} signal=${signal})\n`);
      shutdown(code ?? (signal ? 1 : 0));
    }
  });
  children.push(child);
  return child;
}

function shutdown(exitCode = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    try {
      if (!child.pid) continue;
      if (isWin) {
        // uvicorn --reload starts a child too; stop only this launcher's process tree.
        spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore", windowsHide: true });
      } else {
        process.kill(-child.pid, "SIGTERM");
      }
    } catch {
      /* already gone */
    }
  }
  setTimeout(() => process.exit(exitCode), 1500);
}

process.on("SIGINT", () => shutdown());
process.on("SIGTERM", () => shutdown());

async function main() {
  const nextCli = path.join(root, "web", "node_modules", "next", "dist", "bin", "next");
  if (!apiOnly && !existsSync(nextCli)) {
    console.error("Frontend dependencies are missing. Run pnpm install first.");
    process.exit(1);
  }
  if (!webOnly && (await portInUse(API_PORT, API_HOST))) {
    process.stderr.write(
      `\x1b[31m[dev]\x1b[0m port ${API_PORT} is already in use (another uvicorn? your own app?).\n` +
        `      Set ROTOSTREAM_PORT to something free, then point the web app at it:\n` +
        `        ROTOSTREAM_PORT=8020 node scripts/dev.mjs\n` +
        `        NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8020   # in web/.env.local\n`,
    );
    process.exit(1);
  }
  if (!apiOnly && (await portInUse(WEB_PORT, "127.0.0.1"))) {
    console.error(`[dev] port ${WEB_PORT} is already in use. Set WEB_PORT to a free port.`);
    process.exit(1);
  }

  if (!webOnly) {
    run(
      "api",
      "36",
      python,
      [
        "-m",
        "uvicorn",
        "app.main:app",
        "--reload",
        "--host",
        API_HOST,
        "--port",
        String(API_PORT),
      ],
      path.join(root, "api"),
    );
  }

  if (!apiOnly) {
    run(
      "web",
      "35",
      process.execPath,
      [nextCli, "dev", "--port", String(WEB_PORT)],
      path.join(root, "web"),
    );
  }

  process.stdout.write(
    `\x1b[2m[dev]\x1b[0m api http://${API_HOST}:${API_PORT}/docs   web http://localhost:${WEB_PORT}\n`,
  );
}

main().catch((error) => {
  console.error(`[dev] ${error.message}`);
  shutdown(1);
});
