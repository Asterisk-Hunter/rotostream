import { Studio } from "@/components/Studio";
import { cookies } from "next/headers";
import { getSessionSigningSecret, SESSION_COOKIE, verifySessionToken } from "@/lib/session";

export default async function Home() {
  const publicDemo = process.env.NODE_ENV === "production" && Boolean(process.env.ROTOSTREAM_BACKEND_URL);
  const username = process.env.ROTOSTREAM_AUTH_USER;
  const secret = getSessionSigningSecret();
  const token = (await cookies()).get(SESSION_COOKIE)?.value;
  const canDelete = !publicDemo || Boolean(username && secret && verifySessionToken(token, secret, username));
  return <Studio canDelete={canDelete} publicDemo={publicDemo} />;
}
