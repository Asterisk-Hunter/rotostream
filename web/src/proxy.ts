import { NextRequest, NextResponse } from "next/server";
import { getSessionSigningSecret, SESSION_COOKIE, verifySessionToken } from "./lib/session";
import { DemoLimiter, mutationAllowed } from "./lib/demoLimits";

const demoLimiter = new DemoLimiter();

function isAuthenticated(request: NextRequest): boolean {
  const username = process.env.ROTOSTREAM_AUTH_USER;
  const secret = getSessionSigningSecret();
  return Boolean(username && secret && verifySessionToken(
    request.cookies.get(SESSION_COOKIE)?.value,
    secret,
    username,
  ));
}

export function proxy(request: NextRequest): NextResponse {
  if (process.env.NODE_ENV !== "production") return NextResponse.next();

  if (request.nextUrl.pathname.startsWith("/api/")) {
    // Next's internal URL can use localhost behind a reverse proxy. Compare
    // browser origins against the requested host and Vercel's trusted protocol.
    const protocol = process.env.VERCEL === "1"
      ? request.headers.get("x-forwarded-proto") || "https"
      : request.nextUrl.protocol.slice(0, -1);
    const expectedOrigin = `${protocol}://${request.headers.get("host") || request.nextUrl.host}`;
    if (!mutationAllowed(request.method, request.headers.get("origin"), expectedOrigin, isAuthenticated(request))) {
      return NextResponse.json({ detail: "Request not allowed in the public demo" }, {
        status: 403,
        headers: { "Cache-Control": "no-store" },
      });
    }

    // Vercel overwrites x-forwarded-for at its edge. Do not use a caller's
    // arbitrary header as identity when running behind other hosting providers.
    const client = process.env.VERCEL === "1"
      ? request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "unknown"
      : "shared";
    const retryAfter = demoLimiter.check(client, request.method, request.nextUrl.pathname);
    if (retryAfter) {
      return NextResponse.json({ detail: "Demo limit reached. Please try again later." }, {
        status: 429,
        headers: { "Retry-After": String(retryAfter), "Cache-Control": "no-store" },
      });
    }

    const username = process.env.ROTOSTREAM_AUTH_USER;
    const password = process.env.ROTOSTREAM_AUTH_PASSWORD;
    const backendUrl = process.env.ROTOSTREAM_BACKEND_URL;
    if (!username || !password || !backendUrl) {
      return NextResponse.json({ detail: "API connection is not configured" }, { status: 503 });
    }

    const upstream = new URL(`${request.nextUrl.pathname}${request.nextUrl.search}`, backendUrl);
    const headers = new Headers(request.headers);
    headers.delete("cookie");
    headers.set("Authorization", `Basic ${Buffer.from(`${username}:${password}`).toString("base64")}`);
    const response = NextResponse.rewrite(upstream, { request: { headers } });
    response.headers.set("Cache-Control", "private, no-store");
    return response;
  }

  if (request.nextUrl.pathname === "/login") return NextResponse.redirect(new URL("/", request.url));

  const response = NextResponse.next();
  response.headers.set("Cache-Control", "private, no-store");
  return response;
}

export const config = {
  matcher: ["/((?!auth(?:/|$)|_next/static|_next/image|favicon.ico|robots.txt).*)"],
};
