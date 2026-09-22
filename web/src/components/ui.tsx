"use client";

import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
} from "react";

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

// ---------------------------------------------------------------------- panel
export function Panel({
  title,
  subtitle,
  action,
  children,
  className,
}: {
  title?: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cx(
        "rounded-panel border border-ink-700 bg-ink-850/60 backdrop-blur-sm",
        className,
      )}
    >
      {(title || action) && (
        <header className="flex items-center justify-between gap-3 border-b border-ink-700/70 px-3.5 py-2.5">
          <div className="min-w-0">
            {title && (
              <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-300">
                {title}
              </h2>
            )}
            {subtitle && <p className="mt-0.5 text-xs text-ink-400">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className="p-3.5">{children}</div>
    </section>
  );
}

// --------------------------------------------------------------------- badge
type Tone = "neutral" | "accent" | "positive" | "negative" | "warn";

const TONE_CLASS: Record<Tone, string> = {
  neutral: "border-ink-600 bg-ink-800 text-ink-300",
  accent: "border-accent-500/40 bg-accent-500/10 text-accent-300",
  positive: "border-positive/30 bg-positive/10 text-positive",
  negative: "border-negative/30 bg-negative/10 text-negative",
  warn: "border-warn/30 bg-warn/10 text-warn",
};

export function Badge({
  tone = "neutral",
  children,
  className,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
        TONE_CLASS[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

// -------------------------------------------------------------------- button
type Variant = "primary" | "secondary" | "ghost" | "danger";

const VARIANT_CLASS: Record<Variant, string> = {
  primary:
    "bg-accent-500 text-ink-950 hover:bg-accent-400 border border-transparent font-semibold",
  secondary: "border border-ink-600 bg-ink-800 text-ink-100 hover:bg-ink-750",
  ghost: "border border-transparent text-ink-300 hover:bg-ink-800 hover:text-ink-100",
  danger: "border border-negative/40 bg-negative/10 text-negative hover:bg-negative/20",
};

export function Button({
  variant = "secondary",
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-40",
        VARIANT_CLASS[variant],
        className,
      )}
    >
      {children}
    </button>
  );
}

// --------------------------------------------------------------------- meter
export function Meter({
  value,
  tone = "accent",
  className,
}: {
  value: number;
  tone?: Tone;
  className?: string;
}) {
  const fill: Record<Tone, string> = {
    neutral: "bg-ink-400",
    accent: "bg-accent-400",
    positive: "bg-positive",
    negative: "bg-negative",
    warn: "bg-warn",
  };
  return (
    <div
      className={cx("h-1.5 w-full overflow-hidden rounded-full bg-ink-800", className)}
      role="progressbar"
      aria-valuenow={Math.round(value * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={cx("h-full rounded-full transition-[width] duration-200", fill[tone])}
        style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}
      />
    </div>
  );
}

// --------------------------------------------------------------------- field
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[10px] font-medium uppercase tracking-[0.14em] text-ink-400">
        {label}
      </span>
      {children}
      {hint && <span className="mt-1 block text-[11px] leading-snug text-ink-400">{hint}</span>}
    </label>
  );
}

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...rest}
      className={cx(
        "w-full rounded-md border border-ink-600 bg-ink-800 px-2 py-1.5 text-xs text-ink-100",
        "focus:border-accent-500 focus:outline-none",
        className,
      )}
    >
      {children}
    </select>
  );
}

export function NumberInput({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      type="number"
      {...rest}
      className={cx(
        "w-full rounded-md border border-ink-600 bg-ink-800 px-2 py-1.5 font-mono text-xs text-ink-100 tnum",
        "focus:border-accent-500 focus:outline-none",
        className,
      )}
    />
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between gap-3 rounded-md border border-ink-700 bg-ink-800/60 px-2.5 py-2 text-left transition-colors hover:bg-ink-800"
    >
      <span className="min-w-0">
        <span className="block text-xs text-ink-100">{label}</span>
        {hint && <span className="mt-0.5 block text-[11px] text-ink-400">{hint}</span>}
      </span>
      <span
        className={cx(
          "relative h-4 w-7 shrink-0 rounded-full transition-colors",
          checked ? "bg-accent-500" : "bg-ink-600",
        )}
      >
        <span
          className={cx(
            "absolute top-0.5 h-3 w-3 rounded-full bg-ink-950 transition-[left]",
            checked ? "left-3.5" : "left-0.5",
          )}
        />
      </span>
    </button>
  );
}

// ---------------------------------------------------------------- empty state
export function EmptyState({
  title,
  body,
  icon,
}: {
  title: string;
  body?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 px-6 py-10 text-center">
      {icon && <div className="mb-1 text-ink-500">{icon}</div>}
      <p className="text-sm text-ink-200">{title}</p>
      {body && <p className="max-w-xs text-xs leading-relaxed text-ink-400">{body}</p>}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx(
        "inline-block h-3 w-3 animate-spin rounded-full border-[1.5px] border-ink-500 border-t-accent-400",
        className,
      )}
    />
  );
}

export function StatusDot({ ok, title }: { ok: boolean; title: string }) {
  return (
    <span
      title={title}
      className={cx(
        "inline-block h-1.5 w-1.5 rounded-full",
        ok ? "bg-positive" : "bg-negative",
      )}
    />
  );
}
