// Shared liquid-glass behaviour for every page: theme toggle, lens refraction (Chromium only) and the pointer highlight.

// ===== Theme (stored per viewer; works without storage) =====
const storage = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const saved = storage.get("ig-theme");
if (saved) document.documentElement.dataset.theme = saved;

export function initTheme(button) {
  button?.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    storage.set("ig-theme", document.documentElement.dataset.theme);
    document.dispatchEvent(new CustomEvent("themechange"));
  });
}

// ===== Liquid glass: per-panel lens refraction (Chromium only; others keep frosted blur) =====
const LENS = {
  strong: { bezel: 26, scale: 64, blur: 9, aberration: 0.08 },
  soft: { bezel: 16, scale: 36, blur: 5, aberration: 0.05 },
};
const canRefract = (() => {
  const brands = navigator.userAgentData?.brands?.map((b) => b.brand) || [];
  const reduce = matchMedia("(prefers-reduced-transparency: reduce)").matches;
  return !reduce && brands.some((b) => /Chromium|Google Chrome|Microsoft Edge|Opera|Brave/.test(b));
})();

// Displacement map for a rounded rectangle: pixels near the edge point inward, strongest at the rim,
// so the backdrop bends like light through a curved glass edge. R = x offset, G = y offset (128 = none).
function lensMap(w, h, r, bezel) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(w, h);
  const d = img.data;
  const hw = w / 2, hh = h / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const px = x + 0.5 - hw, py = y + 0.5 - hh;
      const qx = Math.abs(px) - (hw - r), qy = Math.abs(py) - (hh - r);
      const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0));
      const depth = r - outside - Math.min(Math.max(qx, qy), 0); // distance from the edge, inside > 0
      let dx = 0, dy = 0;
      if (depth > 0 && depth < bezel) {
        let nx, ny;
        if (qx > 0 && qy > 0) { const l = Math.hypot(qx, qy) || 1; nx = qx / l; ny = qy / l; }
        else if (qx > qy) { nx = 1; ny = 0; } else { nx = 0; ny = 1; }
        nx *= Math.sign(px) || 1; ny *= Math.sign(py) || 1;
        const t = 1 - depth / bezel;
        const m = t * t * (3 - 2 * t); // smooth falloff toward the centre
        dx = -nx * m; dy = -ny * m;
      }
      const i = (y * w + x) * 4;
      d[i] = 128 + dx * 127; d[i + 1] = 128 + dy * 127; d[i + 2] = 128; d[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL();
}

let defs = null;
const lensState = new Map();
let lensId = 0;
function buildLens(el) {
  const w = Math.round(el.offsetWidth), h = Math.round(el.offsetHeight);
  if (!w || !h) return;
  const key = `${w}x${h}`;
  const prev = lensState.get(el);
  if (prev && prev.key === key) return;
  const p = LENS[el.dataset.glass] || LENS.soft;
  const radius = Math.min(parseFloat(getComputedStyle(el).borderTopLeftRadius) || 24, w / 2, h / 2);
  const id = prev?.id || `lg-${++lensId}`;
  const map = lensMap(w, h, radius, Math.min(p.bezel, radius + 6, w / 3, h / 3));
  const channel = (scale, keep, name) => `
    <feDisplacementMap in="pre" in2="map" scale="${scale}" xChannelSelector="R" yChannelSelector="G" result="d${name}"/>
    <feColorMatrix in="d${name}" type="matrix" values="${keep}" result="${name}"/>`;
  const html = `
    <filter id="${id}" x="0" y="0" width="${w}" height="${h}" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
      <feImage href="${map}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none" result="map"/>
      <feGaussianBlur in="SourceGraphic" stdDeviation="1.2" result="pre"/>
      ${channel(p.scale, "1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0", "r")}
      ${channel(p.scale * (1 + p.aberration), "0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0", "g")}
      ${channel(p.scale * (1 + 2 * p.aberration), "0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0", "b")}
      <feBlend in="r" in2="g" mode="screen" result="rg"/>
      <feBlend in="rg" in2="b" mode="screen" result="rgb"/>
      <feGaussianBlur in="rgb" stdDeviation="${p.blur}" result="soft"/>
      <feColorMatrix in="soft" type="saturate" values="1.75"/>
    </filter>`;
  const old = document.getElementById(id);
  if (old) old.outerHTML = html; else defs.insertAdjacentHTML("beforeend", html);
  el.style.backdropFilter = `url(#${id})`;
  el.style.webkitBackdropFilter = `url(#${id})`;
  lensState.set(el, { key, id });
}
export function refreshLenses() {
  if (!canRefract) return;
  document.querySelectorAll(".glass[data-glass]").forEach(buildLens);
}
export function initGlass() {
  defs = document.querySelector("#lg-defs");
  if (canRefract) {
    document.documentElement.classList.add("refract");
    let t;
    const ro = new ResizeObserver(() => { clearTimeout(t); t = setTimeout(refreshLenses, 120); });
    document.querySelectorAll(".glass[data-glass]").forEach((el) => ro.observe(el));
  }
  // Specular highlight follows the pointer
  document.addEventListener("pointermove", (e) => {
    const el = e.target.closest?.(".glass");
    if (!el) return;
    const b = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${((e.clientX - b.left) / b.width) * 100}%`);
    el.style.setProperty("--my", `${((e.clientY - b.top) / b.height) * 100}%`);
  });
  addEventListener("load", refreshLenses);
  document.fonts?.ready.then(refreshLenses);
}
