"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

function safeReturnPath(): string {
  const requested = new URLSearchParams(window.location.search).get("next");
  if (!requested || !requested.startsWith("/")) return "/";
  try {
    const target = new URL(requested, window.location.origin);
    if (target.origin !== window.location.origin
      || target.pathname === "/login" || target.pathname.startsWith("/auth/")) return "/";
    return `${target.pathname}${target.search}${target.hash}`;
  } catch {
    return "/";
  }
}

export function SignInForm() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError(null);

    try {
      const response = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        cache: "no-store",
        body: JSON.stringify({ username, password }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null) as { detail?: unknown } | null;
        setError(typeof payload?.detail === "string" ? payload.detail : "Sign-in failed. Check your details and try again.");
        return;
      }
      router.replace(safeReturnPath());
      router.refresh();
    } catch {
      setError("Couldn’t reach the sign-in service. Check your connection and retry.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div>
        <label htmlFor="username" className="mb-2 block text-[11px] font-medium text-ink-300">Username</label>
        <input
          autoComplete="username"
          autoCapitalize="none"
          spellCheck={false}
          required
          id="username"
          name="username"
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          className="h-11 w-full rounded-lg border border-ink-700 bg-ink-950/80 px-3.5 text-[13px] text-ink-100 outline-none transition placeholder:text-ink-600 hover:border-ink-600 focus:border-accent-400/70 focus:ring-2 focus:ring-accent-400/10"
          placeholder="Your editor username"
        />
      </div>
      <div>
        <div className="mb-2 flex items-center justify-between">
          <label htmlFor="password" className="text-[11px] font-medium text-ink-300">Password</label>
          <span className="font-mono text-[9px] tracking-wide text-ink-600">PRIVATE</span>
        </div>
        <div className="relative">
          <input
            autoComplete="current-password"
            required
            id="password"
            name="password"
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="h-11 w-full rounded-lg border border-ink-700 bg-ink-950/80 px-3.5 pr-[4.5rem] text-[13px] tracking-[0.08em] text-ink-100 outline-none transition placeholder:font-sans placeholder:tracking-normal placeholder:text-ink-600 hover:border-ink-600 focus:border-accent-400/70 focus:ring-2 focus:ring-accent-400/10"
            placeholder="Enter your password"
          />
          <button
            type="button"
            aria-label={showPassword ? "Hide password" : "Show password"}
            aria-pressed={showPassword}
            onClick={() => setShowPassword((visible) => !visible)}
            className="absolute inset-y-0 right-0 rounded-r-lg px-3 text-[10px] font-medium text-ink-500 transition hover:text-ink-200 focus-visible:text-accent-300"
          >
            {showPassword ? "Hide" : "Show"}
          </button>
        </div>
      </div>

      {error && <p role="alert" className="rounded-lg border border-negative/25 bg-negative/[0.07] px-3 py-2.5 text-[11px] leading-5 text-negative">{error}</p>}

      <button
        type="submit"
        disabled={submitting || !username || !password}
        className="group flex h-11 w-full items-center justify-center gap-2 rounded-lg border border-accent-300/25 bg-accent-400 text-[12px] font-semibold text-ink-950 shadow-[0_8px_28px_rgba(14,165,233,0.16)] transition hover:border-accent-200 hover:bg-accent-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-300 disabled:cursor-wait disabled:opacity-55"
      >
        {submitting ? "Opening workspace…" : "Enter workspace"}
        {!submitting && <span aria-hidden="true" className="transition-transform group-hover:translate-x-0.5">→</span>}
      </button>
      <p className="text-center text-[10px] leading-5 text-ink-600">Your session is protected and expires automatically after 8 hours.</p>
    </form>
  );
}
