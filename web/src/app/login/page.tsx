import { SignInForm } from "@/components/SignInForm";

export const metadata = {
  title: "Sign in — RotoStream",
  description: "Sign in to your RotoStream editing workspace.",
};

export default function LoginPage() {
  return (
    <main className="min-h-screen bg-ink-950 text-ink-100">
      <div className="mx-auto flex min-h-screen w-full max-w-[1280px] flex-col px-5 sm:px-8">
        <header className="flex h-[62px] shrink-0 items-center justify-between border-b border-ink-800">
          <a href="/login" className="text-[15px] font-semibold tracking-[-0.04em]" aria-label="RotoStream home">
            Roto<span className="text-accent-400">Stream</span>
          </a>
          <span className="text-xs text-ink-500">Video editing workspace</span>
        </header>

        <div className="grid flex-1 items-center gap-12 py-12 md:grid-cols-[minmax(0,1fr)_400px] md:gap-20">
          <section className="max-w-[520px]">
            <p className="text-xs font-medium text-accent-300">Rotoscoping</p>
            <h1 className="mt-5 max-w-[480px] text-4xl font-medium leading-[1.08] tracking-[-0.05em] text-ink-100 sm:text-5xl">
              Rotoscoping,<br />frame by frame.
            </h1>
            <p className="mt-5 max-w-[430px] text-[15px] leading-7 text-ink-400">
              Mark a subject, track its mask through the clip, and export the result for your edit.
            </p>
            <div className="mt-10 grid max-w-[430px] grid-cols-3 border-y border-ink-800 py-4 text-xs">
              <div><p className="font-mono text-ink-500">01</p><p className="mt-2 text-ink-200">Mark</p></div>
              <div><p className="font-mono text-ink-500">02</p><p className="mt-2 text-ink-200">Track</p></div>
              <div><p className="font-mono text-ink-500">03</p><p className="mt-2 text-ink-200">Export</p></div>
            </div>
          </section>

          <section className="w-full max-w-[420px] md:justify-self-end" aria-labelledby="sign-in-title">
            <div className="border border-ink-800 bg-ink-900 p-6 sm:p-8">
              <p className="text-xs text-ink-500">Workspace access</p>
              <h2 id="sign-in-title" className="mt-2 text-2xl font-medium tracking-[-0.035em]">Sign in</h2>
              <p className="mt-2 text-[13px] text-ink-400">Enter your editor credentials to continue.</p>
              <div className="mt-7"><SignInForm /></div>
            </div>
          </section>
        </div>
        <footer className="flex min-h-[48px] items-center justify-between border-t border-ink-800 text-[11px] text-ink-500">
          <span>RotoStream</span><span>Private workspace</span>
        </footer>
      </div>
    </main>
  );
}
