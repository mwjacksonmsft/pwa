
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/file-handlers/sw.js', { scope: '/pwa/file-handlers/'});
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

  if ('launchQueue' in window) {
    launchQueue.setConsumer((launchParams) => {
      // Nothing to do when the queue is empty.
      console.log('launchParams.files.length = ' + launchParams.files.length);
      if (!launchParams.files.length) {
        return;
      }
      for (const fileHandle of launchParams.files) {
        // Handle the file.
        console.log('fileHandle = ' + fileHandle);
      }
    });
  }


});





