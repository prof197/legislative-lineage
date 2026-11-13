// Export files fetched in the browser, under the site's base path.
const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export async function fetchData<T>(name: string): Promise<T> {
  const response = await fetch(`${BASE}/data/${name}`);
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  return (await response.json()) as T;
}

export const href = (path: string): string => `${BASE}${path}`;
