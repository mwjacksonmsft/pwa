
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/display-standalone/sw.js', { scope: '/pwa/display-standalone/'});
      console.log('Service worker registered');
  } catch (e) {
      console.log(`Registration failed: ${e}`);
  }
}

if (navigator.serviceWorker) {
  registerServiceWorker();
}




window.addEventListener('DOMContentLoaded', (event) => {
  console.log('DOM fully loaded and parsed');
  document.getElementsByTagName('body')[0].requestFullscreen();
});





