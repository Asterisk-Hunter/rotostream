#!/usr/bin/env node
/** Run a project Python command without swallowing its failure on another platform. */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const windows = process.platform === "win32";
const venv = path.join(root, ".venv", windows ? "Scripts/python.exe" : "bin/python");
const python = process.env.ROTOSTREAM_PYTHON ?? (existsSync(venv) ? venv : windows ? "python" : "python3");
const child = spawn(python, process.argv.slice(2), { cwd: root, stdio: "inherit", shell: false });
child.on("error", (error) => {
  console.error(`Cannot start Python: ${error.message}`);
  process.exitCode = 1;
});
child.on("exit", (code, signal) => {
  process.exitCode = code ?? (signal ? 1 : 0);
});
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}

