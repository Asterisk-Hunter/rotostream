/** The workflow guide has no workspace data and is useful before sign-in. */
export function isPublicGuide(pathname: string): boolean {
  return pathname === "/docs";
}
