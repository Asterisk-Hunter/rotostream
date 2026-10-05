import { NextRequest, NextResponse } from "next/server";
import { matchesBasicCredentials } from "./lib/basic-auth";

function authenticationRequired(): Response {
  return new Response("Authentication required", {
    status: 401,
    headers: {
      "Cache-Control": "private, no-store",
      "Content-Type": "text/plain; charset=utf-8",
      "WWW-Authenticate": 'Basic realm="RotoStream", charset="UTF-8"',
    },
  });
}

export function proxy(request: NextRequest): Response {
  // Local development talks directly to the FastAPI process. Requiring
  // deployment credentials here makes `pnpm dev` return a 503 before the studio
  // can load, while adding no protection to a machine-local server.
  if (process.env.NODE_ENV !== "production") return NextResponse.next();

  const username = process.env.ROTOSTREAM_AUTH_USER;
  const password = process.env.ROTOSTREAM_AUTH_PASSWORD;
  if (!username || !password) {
    return new Response("RotoStream authentication is not configured", {
      status: 503,
      headers: { "Cache-Control": "private, no-store" },
    });
  }

  const authorization = request.headers.get("authorization");
  if (!authorization || !matchesBasicCredentials(authorization, username, password)) {
    return authenticationRequired();
  }

  const response = NextResponse.next();
  response.headers.set("Cache-Control", "private, no-store");
  return response;
}

export const config = {
  matcher: ["/((?!api(?:/|$)|_next/static|_next/image|favicon.ico|robots.txt).*)"],
};
