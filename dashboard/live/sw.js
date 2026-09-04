// Service worker for the dashboard PWA, 2026-09-04.
//
// Scope is /dashboard/live/ (this file's own directory), which covers the whole app.
//
// DELIBERATELY NOT A CACHING SERVICE WORKER. The usual PWA move is to precache the shell for
// offline use; that is wrong here. Every number on this dashboard is live market state, and a
// cached shell paired with a failed fetch shows a confident-looking page full of stale prices --
// the exact failure mode that is worse than an obvious error. So this worker does one job:
// receive pushes and show notifications. Navigation always hits the network.
//
// The notifier daemon (scripts/live_push_notifier_20260904.py) decides WHAT to send; this file
// only decides how it looks.

self.addEventListener("install", (event) => {
  // Take over immediately instead of waiting for every tab to close -- otherwise a fixed
  // notification bug keeps shipping the old worker until the user quits the installed app.
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

// Tier -> presentation. `requireInteraction` keeps a notification on screen until dismissed,
// which is the whole point for T1 (something actually happened to real money); T2 and the digest
// are context and must NOT sit there demanding attention.
const TIER_STYLE = {
  t1: { requireInteraction: true, silent: false, renotify: true },
  t2: { requireInteraction: false, silent: true, renotify: false },
  digest: { requireInteraction: false, silent: true, renotify: false },
  test: { requireInteraction: false, silent: false, renotify: true },
};

self.addEventListener("push", (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (err) {
    // A push whose body is not JSON should still surface rather than vanish silently -- an
    // invisible failure here is indistinguishable from "notifications are broken".
    data = { title: "알림", body: event.data ? event.data.text() : "" };
  }

  const style = TIER_STYLE[data.tier] || TIER_STYLE.t2;
  const options = {
    body: data.body || "",
    icon: "/dashboard/live/icons/icon-192.png",
    badge: "/dashboard/live/icons/badge-96.png",
    // `tag` collapses repeats: a second digest replaces the first instead of stacking 40 deep.
    // The notifier sets a per-event tag for T1/T2 so distinct events still show separately.
    tag: data.tag || "dashboard",
    timestamp: data.ts ? Date.parse(data.ts) : Date.now(),
    data: { url: data.url || "/dashboard/live/" },
    ...style,
  };
  event.waitUntil(self.registration.showNotification(data.title || "트레이딩 대시보드", options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const target = (event.notification.data && event.notification.data.url) || "/dashboard/live/";
  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((windows) => {
      // Reuse an already-open dashboard window rather than opening a second one every click.
      for (const client of windows) {
        if (client.url.includes("/dashboard/live") && "focus" in client) {
          client.navigate(target).catch(() => {});
          return client.focus();
        }
      }
      return self.clients.openWindow(target);
    })
  );
});
