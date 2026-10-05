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
        <label htmlFor="username" className="mb-2 block text-[13px] font-medium text-ink-200">Username</label>
        <input
          autoComplete="username"
          autoCapitalize="none"
          spellCheck={false}
          required
          id="username"
          name="username"
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          className="h-11 w-full rounded-[5px] border border-ink-700 bg-ink-950 px-3 text-[14px] text-ink-100 outline-none transition-colors placeholder:text-ink-500 hover:border-ink-600 focus:border-accent-400"
          placeholder="Username"
        />
      </div>
      <div>
        <div className="mb-2 flex items-center justify-between">
          <label htmlFor="password" className="text-[13px] font-medium text-ink-200">Password</label>
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
            className="h-11 w-full rounded-[5px] border border-ink-700 bg-ink-950 px-3 pr-[4.5rem] text-[14px] text-ink-100 outline-none transition-colors placeholder:font-sans placeholder:tracking-normal placeholder:text-ink-500 hover:border-ink-600 focus:border-accent-400"
            placeholder="Password"
          />
          <button
            type="button"
            aria-label={showPassword ? "Hide password" : "Show password"}
            aria-pressed={showPassword}
            onClick={() => setShowPassword((visible) => !visible)}
            className="absolute inset-y-0 right-0 border-l border-ink-800 px-3 text-xs text-ink-400 transition-colors hover:text-ink-100 focus-visible:text-accent-300"
          >
            {showPassword ? "Hide" : "Show"}
          </button>
        </div>
      </div>

      {error && <p role="alert" className="border border-negative/30 bg-negative/10 px-3 py-2.5 text-[13px] leading-5 text-negative">{error}</p>}

      <button
        type="submit"
        disabled={submitting || !username || !password}
        className="flex h-11 w-full items-center justify-center gap-2 rounded-[5px] bg-accent-400 text-[13px] font-semibold text-ink-950 transition-colors hover:bg-accent-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-300 disabled:cursor-wait disabled:opacity-55"
      >
        {submitting ? "Signing in…" : "Sign in"}
      </button>
      <p className="text-center text-[11px] leading-5 text-ink-500">Your sign-in expires after 8 hours.</p>
    </form>
  );
}
