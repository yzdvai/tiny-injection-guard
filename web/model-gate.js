// The model is a 14 MB download, so it only starts when someone presses the button.
// If this browser already saved it, it loads straight away, since nothing new is downloaded.

// True when any Cache Storage entry ends with this path (transformers.js keeps its files in "transformers-cache")
export async function isSaved(path) {
  try {
    for (const name of await caches.keys()) {
      const keys = await (await caches.open(name)).keys();
      if (keys.some((r) => new URL(r.url).pathname.endsWith(path))) return true;
    }
  } catch { /* no Cache Storage (private mode, old browser): ask first */ }
  return false;
}

// Runs start() now if the file is saved, else shows the button and waits for a click
export async function gate(button, path, start) {
  if (await isSaved(path)) return start();
  button.hidden = false;
  button.addEventListener("click", () => { button.hidden = true; start(); }, { once: true });
  return false;
}

// Fetches a file with progress, keeping a copy in Cache Storage so the next visit needs no click.
// It shares transformers.js's cache, so a model the demo page saved is reused here and the other way round.
export async function fetchSaved(url, onProgress) {
  const cache = await caches.open("transformers-cache").catch(() => null);
  const hit = await caches.match(url).catch(() => null);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const total = +res.headers.get("content-length") || 0, parts = [];
  let loaded = 0;
  for (const reader = res.body.getReader(); ;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); loaded += value.length;
    onProgress?.(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const p of parts) { bytes.set(p, at); at += p.length; }
  await cache?.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
  return bytes;
}
