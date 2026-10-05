import type { NextConfig } from "next";
import path from "node:path";

const backendValue = process.env.ROTOSTREAM_BACKEND_URL;
const backendOrigin = backendValue ? new URL(backendValue).origin : null;

if (backendValue && new URL(backendValue).protocol !== "https:") {
  throw new Error("ROTOSTREAM_BACKEND_URL must use HTTPS");
}

const nextConfig: NextConfig = {
  output: "standalone",
  outputFileTracingRoot: path.join(process.cwd(), ".."),
  poweredByHeader: false,
  async headers() {
    return [{ source: "/:path*", headers: [
      { key: "X-Content-Type-Options", value: "nosniff" },
      { key: "X-Frame-Options", value: "DENY" },
      { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
      { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
    ] }];
  },
  async rewrites() {
    return backendOrigin
      ? [{ source: "/api/:path*", destination: `${backendOrigin}/api/:path*` }]
      : [];
  },
};

export default nextConfig;
