
this.addEventListener('install', async (event) => {
  return;
});

self.addEventListener('fetch', e => {

  if (e.request.url.indexOf('fencedframe.html') >= 0) {
    const newResponse = fetch(e.request).then(response => {
      const newHeaders = new Headers(response.headers);
      newHeaders.append('Supports-Loading-Mode', 'fenced-frame');

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders
      });
    }).catch(() => {
      return new Response('Hello offline page');
    });

    e.respondWith(newResponse);
  } else {
    e.respondWith(
      fetch(e.request).catch(() => {
        return new Response('Hello offline page');
      })
    );
  }
});