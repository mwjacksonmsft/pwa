
const registerServiceWorker = async () => {
  try {
    await navigator.serviceWorker.register('/pwa/get-user-media/sw.js', { scope: '/pwa/get-user-media/' });
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

function getlocation() {
  navigator.geolocation.getCurrentPosition(showLoc);
}
function showLoc(pos) {
  document.getElementById("location").innerHTML = "Latitude: " + pos.coords.latitude +
    "<br>Longitude: " + pos.coords.longitude;
}


window.addEventListener('DOMContentLoaded', (event) => {
  document.querySelector('#showVideo').addEventListener('click', e => startVideoCapture());
  getlocation();
});

