interface StartupConnection<T> {
  load: () => Promise<T>;
  onReady: (value: T) => void;
  onError: (cause: unknown) => void;
}

const RETRY_DELAYS_MS = [1000, 2000, 4000] as const;

function isTransient(cause: unknown): boolean {
  if (cause && typeof cause === "object" && "status" in cause) {
    return [0, 502, 503, 504].includes(Number(cause.status));
  }
  return cause instanceof Error && (
    cause.name === "TimeoutError" ||
    cause.name === "NetworkError" ||
    (cause instanceof TypeError && /fetch|network|load failed/i.test(cause.message))
  );
}

/** Retry startup reads only; disposal cancels timers and discards in-flight results. */
export function connectStartup<T>({ load, onReady, onError }: StartupConnection<T>): () => void {
  let disposed = false;
  let retries = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const attempt = async () => {
    let value: T;
    try {
      value = await load();
    } catch (cause) {
      if (disposed) return;
      if (isTransient(cause) && retries < RETRY_DELAYS_MS.length) {
        timer = setTimeout(() => void attempt(), RETRY_DELAYS_MS[retries++]);
      } else {
        onError(cause);
      }
      return;
    }
    if (!disposed) onReady(value);
  };

  void attempt();
  return () => {
    disposed = true;
    if (timer !== undefined) clearTimeout(timer);
  };
}

export function startupErrorMessage(cause: unknown): string {
  if (cause instanceof Error && cause.name === "TimeoutError") {
    return "The processing service took too long to respond. It may still be starting. Retry to reconnect.";
  }
  if (isTransient(cause)) {
    return "Could not connect to the processing service. Check your connection and retry to reconnect.";
  }
  const detail = cause instanceof Error ? cause.message : String(cause);
  return `Could not load the processing service: ${detail}. Retry to reconnect.`;
}
