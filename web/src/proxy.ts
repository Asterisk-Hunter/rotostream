import { NextRequest, NextResponse } from "next/server";
import { getSessionSigningSecret, SESSION_COOKIE, verifySessionToken } from "./lib/session";
import { isPublicGuide } from "./lib/routeAccess";

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

  // The field guide helps people decide whether to sign in; it contains no
  // workspace data and should be reachable from the public login screen.
  if (isPublicGuide(request.nextUrl.pathname)) return NextResponse.next();

  const authenticated = isAuthenticated(request);

  if (request.nextUrl.pathname.startsWith("/api/")) {
    if (!authenticated) {
      return NextResponse.json({ detail: "Sign in to continue" }, {
        status: 401,
        headers: { "Cache-Control": "no-store" },
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
    headers.set("Authorization", `Basic ${Buffer.from(`${username}:${password}`).toString("base64")}`);
    const response = NextResponse.rewrite(upstream, { request: { headers } });
    response.headers.set("Cache-Control", "private, no-store");
    return response;
  }

  if (request.nextUrl.pathname === "/login") {
    if (authenticated) return NextResponse.redirect(new URL("/", request.url));
    return NextResponse.next();
  }

  if (!authenticated) {
    const login = new URL("/login", request.url);
    login.searchParams.set("next", `${request.nextUrl.pathname}${request.nextUrl.search}`);
    return NextResponse.redirect(login);
  }

  const response = NextResponse.next();
  response.headers.set("Cache-Control", "private, no-store");
  return response;
}

export const config = {
  matcher: ["/((?!auth(?:/|$)|_next/static|_next/image|favicon.ico|robots.txt).*)"],
};
