export type BasicCredentials = Readonly<{ username: string; password: string }>;

function constantTimeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;

  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return difference === 0;
}

export function parseBasicCredentials(header: string | null): BasicCredentials | null {
  if (!header?.startsWith("Basic ")) return null;

  try {
    const decoded = atob(header.slice(6));
    const separator = decoded.indexOf(":");
    if (separator < 1) return null;
    return {
      username: decoded.slice(0, separator),
      password: decoded.slice(separator + 1),
    };
  } catch {
    return null;
  }
}

export function matchesBasicCredentials(
  header: string | null,
  expectedUsername: string,
  expectedPassword: string,
): boolean {
  const credentials = parseBasicCredentials(header);
  return credentials !== null
    && constantTimeEqual(credentials.username, expectedUsername)
    && constantTimeEqual(credentials.password, expectedPassword);
}
