export async function requestJson(baseUrl, path, options = {}) {
  const normalizedBase = baseUrl.replace(/\/$/, "");
  const response = await fetch(`${normalizedBase}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof payload === "string" ? payload : payload?.detail || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload;
}
