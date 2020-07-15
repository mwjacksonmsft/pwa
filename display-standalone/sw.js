
this.addEventListener('install', async (event) => {
  return;
});

this.addEventListener('fetch', (event) => {
  if (event.request.method != 'GET') return;

  event.respondWith(async function() {
    // Try to get the response from a cache.
    const cache = await caches.open('dynamic-v1');
    const cachedResponse = await cache.match(event.request);

    if (cachedResponse) {
      // If we found a match in the cache, return it, but also
      // update the entry in the cache in the background.
      event.waitUntil(cache.add(event.request));
      return cachedResponse;
    }

    // If we didn't find a match in the cache, use the network.
    fetch(event.request).catch((error) => {
      console.log(error);
    });




  }());
});
