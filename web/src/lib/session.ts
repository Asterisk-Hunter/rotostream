import { createHmac, timingSafeEqual } from "node:crypto";

export const SESSION_COOKIE = "rotostream_session";
export const SESSION_TTL_SECONDS = 8 * 60 * 60;

/** Bind editor sessions to the existing Vercel password so no extra key is needed. */
export function getSessionSigningSecret(password = process.env.ROTOSTREAM_AUTH_PASSWORD): string | null {
  if (!password) return null;
  return createHmac("sha256", password).update("rotostream/session/v1").digest("hex");
}

interface SessionClaims {
  sub: string;
  iat: number;
  exp: number;
  iss: "rotostream";
}

function sign(encodedClaims: string, secret: string): Buffer {
  return createHmac("sha256", secret).update(encodedClaims).digest();
}

export function createSessionToken(
  username: string,
  secret: string,
  nowSeconds = Math.floor(Date.now() / 1000),
): string {
  const claims: SessionClaims = {
    sub: username,
    iat: nowSeconds,
    exp: nowSeconds + SESSION_TTL_SECONDS,
    iss: "rotostream",
  };
  const encodedClaims = Buffer.from(JSON.stringify(claims)).toString("base64url");
  return `${encodedClaims}.${sign(encodedClaims, secret).toString("base64url")}`;
}

export function verifySessionToken(
  token: string | undefined,
  secret: string,
  expectedUsername: string,
  nowSeconds = Math.floor(Date.now() / 1000),
): boolean {
  if (!token || !secret || !expectedUsername) return false;
  const [encodedClaims, encodedSignature, extra] = token.split(".");
  if (!encodedClaims || !encodedSignature || extra !== undefined) return false;

  try {
    const suppliedSignature = Buffer.from(encodedSignature, "base64url");
    const expectedSignature = sign(encodedClaims, secret);
    if (suppliedSignature.length !== expectedSignature.length
      || !timingSafeEqual(suppliedSignature, expectedSignature)) return false;

    const claims = JSON.parse(Buffer.from(encodedClaims, "base64url").toString("utf8")) as Partial<SessionClaims>;
    return claims.iss === "rotostream"
      && claims.sub === expectedUsername
      && Number.isSafeInteger(claims.iat)
      && Number.isSafeInteger(claims.exp)
      && (claims.iat as number) <= nowSeconds + 60
      && (claims.exp as number) > nowSeconds;
  } catch {
    return false;
  }
}
