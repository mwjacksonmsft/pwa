
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


async function verifyPermission(fileHandle, withWrite) {
  const opts = {};
  if (withWrite) {
    opts.mode = 'readwrite';
  }

  // Check if we already have permission, if so, return true.
  if (await fileHandle.queryPermission(opts) === 'granted') {
    return true;
  }

  // Request permission to the file, if the user grants permission, return true.
  if (await fileHandle.requestPermission(opts) === 'granted') {
    return true;
  }

  // The user did not grant permission, return false.
  return false;
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
        console.log('fileHandle.kind = ' + fileHandle.kind);
        console.log('fileHandle.name = ' + fileHandle.name);
        console.log('can read/write = ' + verifyPermission(fileHandle, true));

      }
    });
  }


});





