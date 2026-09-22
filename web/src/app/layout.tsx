import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RotoStream — AI rotoscoping studio",
  description:
    "Click an object once and propagate a temporally consistent mask across the video, then export an alpha matte, a transparent video or a replaced background.",
};

// Explicit typing rather than Next's generated `LayoutProps` global, so
// `tsc --noEmit` works before the first `next build` has emitted types.
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans min-h-screen antialiased">{children}</body>
    </html>
  );
}
