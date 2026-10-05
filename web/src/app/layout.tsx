import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import "./globals.css";

const plexSans = IBM_Plex_Sans({ subsets: ["latin"], variable: "--font-plex-sans", display: "swap" });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: "400", variable: "--font-plex-mono", display: "swap" });

export const metadata: Metadata = {
  title: "RotoStream — AI rotoscoping studio",
  description:
    "Click an object once and propagate a temporally consistent mask across the video, then export an alpha matte, a transparent video or a replaced background.",
};

// Explicit typing rather than Next's generated `LayoutProps` global, so
// `tsc --noEmit` works before the first `next build` has emitted types.
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
      <body className="font-sans min-h-screen antialiased">{children}</body>
    </html>
  );
}
