// Shared by every page: English/Arabic and the theme toggle.

// ===== English / Arabic (shares the "lang" choice with the rest of yazeed.quest) =====
// The HTML carries both languages: [data-en] / [data-ar] elements, which CSS shows by <html lang>, and
// data-ar-<attr> for attributes. The head script sets lang and dir before first paint. T('English', 'العربية')
// picks the string for scripts; switching saves the choice and reloads so every script redraws its text.
export const AR = document.documentElement.lang === "ar";
export const T = (en, ar) => (AR ? ar : en);
window.T = T;
// An Arabic count with the right noun form: 1 → one, 2 → two, 3–10 → "n few", 11–99 → "n many" (by the last two digits)
export const arN = (n, one, two, few, many) => { const r = Math.abs(n) % 100; return n === 1 ? one : n === 2 ? two : r >= 3 && r <= 10 ? `${n} ${few}` : r >= 11 ? `${n} ${many}` : `${n} ${one}`; };
if (AR) {
  for (const el of document.querySelectorAll("*"))
    for (const a of [...el.attributes]) if (a.name.startsWith("data-ar-")) el.setAttribute(a.name.slice(8), a.value);
  const title = document.querySelector("title[data-ar-text]");
  if (title) document.title = title.getAttribute("data-ar-text");
}
export function initLang(button) {
  if (!button) return;
  button.textContent = AR ? "EN" : "ع";
  button.lang = AR ? "en" : "ar";
  button.setAttribute("aria-label", AR ? "Switch to English" : "التبديل إلى العربية");
  button.addEventListener("click", () => {
    try { localStorage.setItem("lang", AR ? "en" : "ar"); } catch { /* private mode */ }
    const u = new URL(location.href);
    u.searchParams.delete("lang");
    if (u.href === location.href) location.reload(); else location.replace(u.href);
  });
}

// ===== Theme (shares the "mode" choice with the rest of yazeed.quest; works without storage) =====
const storage = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const saved = storage.get("mode");
if (saved) document.documentElement.dataset.theme = saved;

export function initTheme(button) {
  button?.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    storage.set("mode", document.documentElement.dataset.theme);
    document.dispatchEvent(new CustomEvent("themechange"));
  });
}
