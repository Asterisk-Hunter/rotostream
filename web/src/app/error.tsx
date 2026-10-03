"use client";

import { Button } from "@/components/ui";

export default function StudioError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 p-6 text-center">
      <h1 className="text-xl font-semibold text-ink-100">The studio could not render</h1>
      <p className="max-w-sm text-sm text-ink-300">Your clips and completed exports stay saved on the server. Reload the studio to continue.</p>
      <Button variant="primary" onClick={reset}>Reload studio</Button>
    </main>
  );
}
