type Bucket = { count: number; resetAt: number };
type Rule = { key: string; limit: number; windowMs: number };

const HOUR = 60 * 60 * 1000;
const MINUTE = 60 * 1000;

/** Best-effort instance-local protection; not a durable billing cap. */
export class DemoLimiter {
  private readonly buckets = new Map<string, Bucket>();

  check(client: string, method: string, pathname: string, now = Date.now()): number {
    for (const [key, bucket] of this.buckets) {
      if (bucket.resetAt <= now) this.buckets.delete(key);
    }
    // Fail closed instead of evicting active quotas under a flood of new IPs.
    if (this.buckets.size >= 10_000 && !this.buckets.has(`requests:${client}`)) return 60;

    const rules: Rule[] = [{ key: `requests:${client}`, limit: 300, windowMs: MINUTE }];
    if (method === "POST") {
      const upload = pathname === "/api/videos" || pathname === "/api/videos/";
      const category = upload ? "uploads" : "processing";
      rules.push(
        { key: `${category}:${client}`, limit: upload ? 2 : 20, windowMs: HOUR },
        { key: `${category}:global`, limit: upload ? 20 : 100, windowMs: HOUR },
      );
    }

    const retryAfter = Math.max(0, ...rules.map(({ key, limit }) => {
      const bucket = this.buckets.get(key);
      return bucket && bucket.count >= limit ? Math.ceil((bucket.resetAt - now) / 1000) : 0;
    }));
    if (retryAfter) return retryAfter;
    for (const { key, windowMs } of rules) {
      const bucket = this.buckets.get(key);
      if (bucket) bucket.count += 1;
      else this.buckets.set(key, { count: 1, resetAt: now + windowMs });
    }
    return 0;
  }
}

export function mutationAllowed(method: string, origin: string | null, expectedOrigin: string, editor: boolean): boolean {
  if (["GET", "HEAD", "OPTIONS"].includes(method)) return true;
  if (origin !== expectedOrigin) return false;
  return method !== "DELETE" || editor;
}
