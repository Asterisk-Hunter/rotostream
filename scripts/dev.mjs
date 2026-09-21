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
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const isWin = process.platform === "win32";

const args = new Set(process.argv.slice(2));
const apiOnly = args.has("--api-only");
const webOnly = args.has("--web-only");

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
}

function run(name, colour, command, commandArgs, cwd) {
  const tag = `\x1b[${colour}m[${name}]\x1b[0m`;
  const child = spawn(command, commandArgs, { cwd, env: process.env, shell: false });
  pipeWithPrefix(child.stdout, tag, false);
  pipeWithPrefix(child.stderr, tag, true);
  child.on("error", (err) => {
    process.stderr.write(`${tag} failed to start: ${err.message}\n`);
  });
  child.on("exit", (code, signal) => {
    if (!shuttingDown) {
      process.stdout.write(`${tag} exited (code=${code} signal=${signal})\n`);
      shutdown();
    }
  });
  children.push(child);
  return child;
}

function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    try {
      child.kill();
    } catch {
      /* already gone */
    }
  }
  setTimeout(() => process.exit(0), 150);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

if (!webOnly) {
  run(
    "api",
    "36",
    python,
    ["-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
    path.join(root, "api"),
  );
}

if (!apiOnly) {
  run("web", "35", isWin ? "pnpm.cmd" : "pnpm", ["dev"], path.join(root, "web"));
}

process.stdout.write(
  `\x1b[2m[dev]\x1b[0m api http://127.0.0.1:8000/docs   web http://localhost:3000\n`,
);
