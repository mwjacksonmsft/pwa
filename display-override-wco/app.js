
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/display-override-wco/sw.js', { scope: '/pwa/display-override-wco/'});
      console.log('Service worker registered');
  } catch (e) {
      console.log(`Registration failed: ${e}`);
  }
}

if (navigator.serviceWorker) {
  registerServiceWorker();
}
