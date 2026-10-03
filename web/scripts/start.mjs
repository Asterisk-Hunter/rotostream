import { cpSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { spawn } from "node:child_process";

const webDir = fileURLToPath(new URL("..", import.meta.url));
const standaloneDir = path.join(webDir, ".next", "standalone", "web");
const server = path.join(standaloneDir, "server.js");
if (!existsSync(server)) {
  process.stderr.write("Production build missing. Run pnpm --dir web build first.\n");
  process.exit(1);
}
// Next intentionally leaves static/public assets out of standalone output.
cpSync(path.join(webDir, ".next", "static"), path.join(standaloneDir, ".next", "static"), { recursive: true });
const publicDir = path.join(webDir, "public");
if (existsSync(publicDir)) cpSync(publicDir, path.join(standaloneDir, "public"), { recursive: true });
const child = spawn(process.execPath, [server], { stdio: "inherit", env: process.env });
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => child.kill(signal));
child.on("error", (error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
child.on("exit", (code) => { process.exitCode = code ?? 1; });
