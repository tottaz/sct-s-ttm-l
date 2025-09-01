self.addEventListener("install", event => {
  event.waitUntil(
    caches.open("sattmal-cache").then(cache => {
      return cache.addAll([
        "/",
        "/docdashboard",
        "/docupload",
        "/static/js/signature_pad.umd.min.js"
      ]);
    })
  );
});

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});
