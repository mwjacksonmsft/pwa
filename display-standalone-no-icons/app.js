
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/display-standalone-no-icons/sw.js', { scope: '/pwa/display-standalone-no-icons/'});
      console.log('Service worker registered');
  } catch (e) {
      console.log(`Registration failed: ${e}`);
  }
}

if (navigator.serviceWorker) {
  registerServiceWorker();
}
