
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/display-override-custom/sw.js', { scope: '/pwa/display-override-custom/'});
      console.log('Service worker registered');
  } catch (e) {
      console.log(`Registration failed: ${e}`);
  }
}

if (navigator.serviceWorker) {
  registerServiceWorker();
}
