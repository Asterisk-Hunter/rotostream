import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "How to use RotoStream",
  description: "A practical guide to marking a subject, checking its mask, and exporting a cutout.",
};

const sections = [
  ["start", "Make a first cut"],
  ["mark", "Mark the subject"],
  ["track", "Track the clip"],
  ["review", "Check the result"],
  ["export", "Choose an export"],
  ["limits", "Clip size and speed"],
  ["trouble", "When something goes wrong"],
  ["remove", "Remove a clip"],
] as const;

export default function DocsPage() {
  return (
    <main className="min-h-screen bg-ink-950 text-ink-200">
      <header className="border-b border-ink-800">
        <div className="mx-auto flex h-[58px] max-w-[1180px] items-center justify-between px-5 sm:px-8">
          <Link href="/" className="text-[15px] font-semibold tracking-[-0.04em] text-ink-100">
            Roto<span className="text-accent-400">Stream</span>
          </Link>
          <Link href="/" className="text-xs text-ink-400 transition-colors hover:text-ink-100">Back to editor <span aria-hidden="true">→</span></Link>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1180px] gap-12 px-5 py-10 sm:px-8 sm:py-14 lg:grid-cols-[220px_minmax(0,700px)] lg:gap-20">
        <aside className="self-start lg:sticky lg:top-8">
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-accent-300">Field guide</p>
          <h1 className="mt-3 text-3xl font-medium tracking-[-0.045em] text-ink-100">Get a clean cutout.</h1>
          <p className="mt-3 text-[13px] leading-6 text-ink-400">Mark what stays, check where the edge slips, then export the version your next tool needs.</p>
          <nav aria-label="Guide sections" className="mt-8 hidden border-l border-ink-800 lg:block">
            {sections.map(([id, title]) => (
              <a key={id} href={`#${id}`} className="block border-l border-transparent py-2 pl-3 text-xs text-ink-400 transition-colors hover:border-accent-400 hover:text-ink-100">{title}</a>
            ))}
          </nav>
        </aside>

        <article className="min-w-0 space-y-12 text-[14px] leading-7 text-ink-300">
          <section id="start" className="scroll-mt-8">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-500">01 / Workflow</p>
            <h2 className="mt-2 text-xl font-medium tracking-[-0.025em] text-ink-100">Make a first cut</h2>
            <ol className="mt-4 list-decimal space-y-2 pl-5 marker:font-mono marker:text-accent-300">
              <li>Upload a short clip and wait for frame extraction to finish.</li>
              <li>On a clear frame, draw a box around the subject or click inside it.</li>
              <li>Use a background click only where the mask spills onto something that should be removed.</li>
              <li>Run tracking, inspect the timeline, fix problem frames, then run tracking again.</li>
              <li>Choose an export for your next step and download it from the export panel.</li>
            </ol>
            <p className="mt-4 border-l-2 border-accent-400/70 pl-4 text-[13px] text-ink-400">Start with a short, steady shot. A few accurate marks usually help more than marking every frame.</p>
          </section>

          <section id="mark" className="scroll-mt-8">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-500">02 / Mark</p>
            <h2 className="mt-2 text-xl font-medium tracking-[-0.025em] text-ink-100">Tell the tracker what belongs in the cutout</h2>
            <p className="mt-3">A <strong className="font-medium text-ink-100">foreground click</strong> means “keep this.” A <strong className="font-medium text-ink-100">background click</strong> means “remove this.” A <strong className="font-medium text-ink-100">box</strong> gives the tracker a rough region containing the subject. Marks apply to the frame shown in the viewer.</p>
            <p className="mt-3">If the subject changes shape or gets partly hidden, move to a clear frame near that moment and add another foreground mark. Use a background mark to trim spill. Undo removes the last mark; clear-frame removes marks from the current frame.</p>
          </section>

          <section id="track" className="scroll-mt-8">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-500">03 / Track</p>
            <h2 className="mt-2 text-xl font-medium tracking-[-0.025em] text-ink-100">Let the mask follow the subject</h2>
            <p className="mt-3">Tracking propagates your marks through the clip. Bidirectional tracking starts from the marks and works toward both the beginning and the end, which helps when the clearest frame is in the middle. Turn it off when you need a forward-only run.</p>
            <p className="mt-3">The CPU demo tracker is the default on the hosted service. It is useful for testing the workflow, but its cutouts are simple color-based masks and can miss boundaries. The memory tracker uses SAM 2.1 weights and produces stronger object masks, but it takes longer and needs the model environment enabled. Changing the tracker affects the next run.</p>
          </section>

          <section id="review" className="scroll-mt-8">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-500">04 / Review</p>
            <h2 className="mt-2 text-xl font-medium tracking-[-0.025em] text-ink-100">Find the frames that need your eye</h2>
            <p className="mt-3">The timeline colors summarize tracking confidence and missing masks. Use the problem arrows to jump to frames without a mask or with low confidence. Play is a sampled review preview; use the frame slider, filmstrip, or arrow keys for precise inspection.</p>
            <p className="mt-3">Click the subject to correct a frame. The preview shows your new marks against that frame; they do not change the completed run until you track again. Re-run after corrections, then inspect the changed sections before exporting.</p>
          </section>

          <section id="export" className="scroll-mt-8">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-500">05 / Export</p>
            <h2 className="mt-2 text-xl font-medium tracking-[-0.025em] text-ink-100">Pick the file for the job</h2>
            <div className="mt-4 divide-y divide-ink-800 border-y border-ink-800">
              <div className="grid gap-1 py-3 sm:grid-cols-[150px_1fr]"><strong className="font-medium text-ink-100">Replace background · MP4</strong><span>Ready-to-watch clip with the background blurred, filled with a color, or set to black or white. Keeps source audio when present.</span></div>
              <div className="grid gap-1 py-3 sm:grid-cols-[150px_1fr]"><strong className="font-medium text-ink-100">Overlay · MP4</strong><span>Original image with the mask edge drawn over it, useful for review and sharing.</span></div>
              <div className="grid gap-1 py-3 sm:grid-cols-[150px_1fr]"><strong className="font-medium text-ink-100">Transparent · WebM</strong><span>Video with an alpha channel for compositing in an editor that supports transparent WebM.</span></div>
              <div className="grid gap-1 py-3 sm:grid-cols-[150px_1fr]"><strong className="font-medium text-ink-100">Cutout frames · ZIP</strong><span>Numbered transparent PNG frames when your editor needs an image sequence.</span></div>
              <div className="grid gap-1 py-3 sm:grid-cols-[150px_1fr]"><strong className="font-medium text-ink-100">Mask files · ZIP / JSON</strong><span>Binary masks or COCO-style run-length data for technical workflows, not a finished video.</span></div>
            </div>
            <p className="mt-4 text-[13px] text-ink-400">If the timeline reports missing masks, those frames export without the subject in the cutout. Fix them and track again before making a final file.</p>
          </section>

          <section id="limits" className="scroll-mt-8">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-500">06 / Expectations</p>
            <h2 className="mt-2 text-xl font-medium tracking-[-0.025em] text-ink-100">Clip size and speed</h2>
            <p className="mt-3">The hosted demo extracts up to 900 frames and processes them at a working resolution with a 960-pixel long side. Larger uploads take longer to upload and extract; a longer clip also means more frames to track and export. A 10-second clip is a good first test.</p>
            <p className="mt-3">The hosted setup runs one CPU demo worker. Tracking and export jobs run on the service and do not survive an instance restart. SAM 2.1 is not enabled on that hosted CPU service. Treat the hosted service as a private, single-editor demo, not shared or archival storage.</p>
          </section>

          <section id="trouble" className="scroll-mt-8">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-500">07 / Troubleshooting</p>
            <h2 className="mt-2 text-xl font-medium tracking-[-0.025em] text-ink-100">When something goes wrong</h2>
            <dl className="mt-4 divide-y divide-ink-800 border-y border-ink-800">
              <div className="grid gap-1 py-3 sm:grid-cols-[210px_1fr]"><dt className="font-medium text-ink-100">Upload or frames keep loading</dt><dd>Check your connection, then use Retry if it appears. Large clips take longer. If the service restarted, reload the clip to see whether extraction completed.</dd></div>
              <div className="grid gap-1 py-3 sm:grid-cols-[210px_1fr]"><dt className="font-medium text-ink-100">Track is unavailable</dt><dd>Add a foreground click or a box around the subject. Background-only marks cannot start an object track.</dd></div>
              <div className="grid gap-1 py-3 sm:grid-cols-[210px_1fr]"><dt className="font-medium text-ink-100">The mask misses or includes something</dt><dd>Add a foreground or background mark on a clear frame near the error, then track again and inspect nearby frames.</dd></div>
              <div className="grid gap-1 py-3 sm:grid-cols-[210px_1fr]"><dt className="font-medium text-ink-100">The preview does not match my marks</dt><dd>Wait for the preview after each click. The preview is only for the current frame; the saved result changes after a new tracking run completes.</dd></div>
              <div className="grid gap-1 py-3 sm:grid-cols-[210px_1fr]"><dt className="font-medium text-ink-100">A job stopped or disappeared</dt><dd>Jobs are held in service memory, so an instance restart interrupts active work. Reload the clip and run tracking or export again.</dd></div>
            </dl>
          </section>

          <section id="remove" className="scroll-mt-8">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-500">08 / Your files</p>
            <h2 className="mt-2 text-xl font-medium tracking-[-0.025em] text-ink-100">Remove a clip and its work</h2>
            <p className="mt-3">Use the clip’s delete control in the project list to remove its uploaded video, extracted frames, tracking sessions, masks, and exports from this workspace. Active jobs must finish or be cancelled first. On the hosted demo, files are stored in the service’s Cloud Storage workspace until you delete them; this is a private demo workspace, not a backup or long-term archive.</p>
          </section>

          <footer className="border-t border-ink-800 pt-5 text-xs text-ink-500">
            <Link href="/" className="text-ink-300 hover:text-ink-100">Return to the editor <span aria-hidden="true">→</span></Link>
            <p className="mt-2">Project and deployment details are in the repository README.</p>
          </footer>
        </article>
      </div>
    </main>
  );
}
