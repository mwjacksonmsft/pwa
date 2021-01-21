
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/get-user-media/sw.js', { scope: '/pwa/get-user-media/'});
      console.log('Service worker registered');
  } catch (e) {
      console.log(`Registration failed: ${e}`);
  }
}

if (navigator.serviceWorker) {
  registerServiceWorker();
}


function startVideoCapture() {
  // Prefer camera resolution nearest to 1280x720.
  var constraints = { audio: true, video: { width: 1280, height: 720 } };

  navigator.mediaDevices.getUserMedia(constraints)
    .then(function (mediaStream) {
      var video = document.querySelector('video');
      video.srcObject = mediaStream;
      video.onloadedmetadata = function (e) {
        video.play();
      };
    })
    .catch(function (err) { console.log(err.name + ": " + err.message); }); // always check for errors at the end.
}

window.addEventListener('DOMContentLoaded', (event) => {
  console.log('DOM fully loaded and parsed');
  document.getElementsByTagName('body')[0].requestFullscreen();
});





