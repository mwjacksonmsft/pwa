
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/display-override-browser/sw.js', { scope: '/pwa/display-override-browser/'});
      console.log('Service worker registered');
  } catch (e) {
      console.log(`Registration failed: ${e}`);
  }
}

if (navigator.serviceWorker) {
  registerServiceWorker();
}
