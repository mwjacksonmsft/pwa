
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/display-modifiers-wco/sw.js', { scope: '/pwa/display-modifiers-wco/'});
      console.log('Service worker registered');
  } catch (e) {
      console.log(`Registration failed: ${e}`);
  }
}

if (navigator.serviceWorker) {
  registerServiceWorker();
}
