import { timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import { createSessionToken, getSessionSigningSecret, SESSION_COOKIE, SESSION_TTL_SECONDS } from "@/lib/session";

function matchesSecret(supplied: string, expected: string): boolean {
  const left = Buffer.from(supplied);
  const right = Buffer.from(expected);
  return left.length === right.length && timingSafeEqual(left, right);
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  const origin = request.headers.get("origin");
  if (!origin || origin !== request.nextUrl.origin) {
    return NextResponse.json({ detail: "Request origin not allowed" }, { status: 403 });
  }

  const expectedUsername = process.env.ROTOSTREAM_AUTH_USER;
  const expectedPassword = process.env.ROTOSTREAM_AUTH_PASSWORD;
  const sessionSecret = getSessionSigningSecret(expectedPassword);
  if (!expectedUsername || !expectedPassword || !sessionSecret) {
    return NextResponse.json({ detail: "Sign-in is not configured" }, { status: 503 });
  }

  let credentials: unknown;
  try {
    credentials = await request.json();
  } catch {
    return NextResponse.json({ detail: "Enter your username and password" }, { status: 400 });
  }

  if (!credentials || typeof credentials !== "object") {
    return NextResponse.json({ detail: "Enter your username and password" }, { status: 400 });
  }
  const { username, password } = credentials as Record<string, unknown>;
  if (typeof username !== "string" || typeof password !== "string"
    || !matchesSecret(username, expectedUsername)
    || !matchesSecret(password, expectedPassword)) {
    return NextResponse.json({ detail: "That username or password didn’t match" }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(SESSION_COOKIE, createSessionToken(username, sessionSecret), {
    httpOnly: true,
    secure: true,
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_TTL_SECONDS,
  });
  response.headers.set("Cache-Control", "no-store");
  return response;
}
