import { SignInForm } from "@/components/SignInForm";

export const metadata = {
  title: "Sign in — RotoStream",
  description: "Sign in to your RotoStream editing workspace.",
};

export default function LoginPage() {
  return (
    <main className="relative isolate min-h-screen overflow-hidden bg-ink-950 text-ink-100">
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.035)_1px,transparent_1px)] bg-[size:64px_64px] [mask-image:linear-gradient(to_bottom,black,transparent_88%)]" />
        <div className="absolute -left-40 top-[-20rem] h-[42rem] w-[42rem] rounded-full bg-accent-500/[0.09] blur-[110px]" />
        <div className="absolute bottom-[-28rem] right-[-12rem] h-[44rem] w-[44rem] rounded-full bg-cyan-300/[0.055] blur-[120px]" />
      </div>

      <div className="mx-auto flex min-h-screen w-full max-w-[1440px] flex-col px-5 sm:px-10 lg:px-16">
        <header className="flex h-[76px] shrink-0 items-center justify-between border-b border-white/[0.075]">
          <a href="/login" className="group flex items-center gap-3 rounded-sm" aria-label="RotoStream sign in">
            <span className="relative grid h-8 w-8 place-items-center overflow-hidden rounded-[9px] border border-accent-300/25 bg-accent-400/[0.09]">
              <svg viewBox="0 0 24 24" fill="none" className="h-[18px] w-[18px] text-accent-300" aria-hidden="true">
                <path d="M4 5.5h16v13H4z" stroke="currentColor" strokeWidth="1.4" />
                <path d="M8 5.5v13M16 5.5v13M4 9h4m-4 6h4m8-6h4m-4 6h4" stroke="currentColor" strokeWidth="1.1" />
                <path d="m10 9 5 3-5 3V9Z" fill="currentColor" />
              </svg>
            </span>
            <span className="text-[14px] font-semibold tracking-[-0.04em] text-ink-100">Roto<span className="text-accent-300">Stream</span></span>
          </a>
          <span className="hidden items-center gap-2 text-[10px] font-medium uppercase tracking-[0.2em] text-ink-500 sm:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-positive shadow-[0_0_10px_rgba(74,222,128,0.55)]" />
            Private workspace
          </span>
        </header>

        <div className="grid flex-1 items-center gap-14 py-14 md:grid-cols-[minmax(0,1fr)_minmax(340px,440px)] md:gap-10 lg:gap-24 lg:py-20">
          <section className="relative hidden min-h-[470px] flex-col justify-between md:flex">
            <div className="relative z-10 max-w-[550px] pt-4 lg:pt-10">
              <p className="mb-6 flex items-center gap-3 text-[10px] font-semibold uppercase tracking-[0.24em] text-accent-300/80">
                <span className="h-px w-8 bg-accent-300/50" />
                Frame-accurate by design
              </p>
              <h1 className="max-w-[540px] text-[clamp(3.3rem,6.2vw,6.25rem)] font-medium leading-[0.94] tracking-[-0.075em] text-ink-100">
                Keep the
                <br />
                <span className="font-light text-ink-400">right</span> edge.
              </h1>
              <p className="mt-8 max-w-[390px] text-[14px] leading-7 text-ink-400">
                A focused rotoscoping workspace for shaping a selection once, then carrying it cleanly through every frame.
              </p>
            </div>

            <div className="relative mb-2 ml-2 mt-12 h-[164px] max-w-[520px] overflow-hidden rounded-xl border border-white/[0.09] bg-[#101720]/75 shadow-[0_28px_90px_rgba(0,0,0,0.3)]">
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_55%_110%,rgba(56,189,248,0.12),transparent_54%)]" />
              <div className="absolute left-5 top-4 flex items-center gap-2 font-mono text-[9px] tracking-[0.16em] text-ink-500">
                <span className="text-accent-300">MASK / 001</span><span>·</span><span>00:00:03:12</span>
              </div>
              <div className="absolute inset-x-0 bottom-[34px] h-px bg-white/[0.07]" />
              <div className="absolute bottom-5 left-5 right-5 flex h-[19px] items-center gap-[3px]">
                {Array.from({ length: 34 }, (_, index) => (
                  <span key={index} className={`h-full flex-1 rounded-[2px] ${index < 17 ? "bg-accent-400/25" : "bg-white/[0.07]"}`} />
                ))}
              </div>
              <div className="absolute bottom-[31px] left-[51%] top-[38px] w-px bg-accent-300 shadow-[0_0_9px_rgba(125,211,252,0.9)]">
                <span className="absolute -left-[3px] top-0 h-[7px] w-[7px] rounded-full bg-accent-200" />
              </div>
              <svg className="absolute right-[15%] top-[27px] h-[80px] w-[132px] text-accent-300/80" viewBox="0 0 132 80" fill="none" aria-hidden="true">
                <path d="M48 11c-12 1-18 11-18 23 0 7-8 10-8 18 0 7 8 9 12 14 4 5 11 5 16 4 8-2 16-1 21-5 5-5 2-11 0-16-2-7 6-12 5-20-1-12-12-19-28-18Z" stroke="currentColor" strokeWidth="1.4" strokeDasharray="3 3" />
                <path d="M4 43c8-8 12-14 17-17m86 6 19-9m-26 43 17 8" stroke="currentColor" strokeOpacity=".35" strokeWidth="1" />
                <circle cx="47" cy="37" r="3" fill="#4ade80" />
                <circle cx="55" cy="48" r="2.5" fill="#fb7185" />
                <path d="m45 35 2-2 2 2-2 2-2-2Z" fill="#4ade80" />
              </svg>
              <span className="absolute bottom-[52px] left-5 text-[9px] uppercase tracking-[0.17em] text-ink-500">Propagation timeline</span>
            </div>

            <div className="mt-8 flex items-center gap-2 text-[10px] tracking-wide text-ink-600">
              <span className="font-mono">ROTOSTREAM</span><span>—</span><span>One selection. Every frame.</span>
            </div>
          </section>

          <section className="mx-auto w-full max-w-[420px] md:mx-0 md:justify-self-end">
            <div className="mb-8 md:hidden">
              <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-accent-300/80">Frame-accurate by design</p>
              <h1 className="mt-4 text-5xl font-medium tracking-[-0.07em] text-ink-100">Your studio,<br /><span className="font-light text-ink-400">in focus.</span></h1>
            </div>
            <div className="relative overflow-hidden rounded-2xl border border-white/[0.1] bg-ink-900/80 p-6 shadow-[0_32px_100px_rgba(0,0,0,0.4)] backdrop-blur-xl sm:p-9">
              <div className="absolute inset-x-8 top-0 h-px bg-gradient-to-r from-transparent via-accent-300/60 to-transparent" />
              <div className="mb-8">
                <p className="font-mono text-[10px] tracking-[0.17em] text-ink-500">EDITOR ACCESS <span className="text-ink-700">/</span> 01</p>
                <h2 className="mt-3 text-[27px] font-medium tracking-[-0.055em] text-ink-100">Welcome back.</h2>
                <p className="mt-2 text-[12px] leading-5 text-ink-400">Sign in to open your video workspace.</p>
              </div>
              <SignInForm />
              <div className="mt-7 flex items-center justify-between border-t border-white/[0.07] pt-5 text-[10px] text-ink-600">
                <span>Encrypted session</span>
                <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-positive/80" /> Cloud Run connected</span>
              </div>
            </div>
            <p className="mt-5 text-center text-[10px] text-ink-600">Access is limited to the workspace editor.</p>
          </section>
        </div>
      </div>
    </main>
  );
}
