
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
  var constraints = { audio: true, video: { width: 640, height: 480 } };

  navigator.mediaDevices.getUserMedia(constraints)
    .then(function (mediaStream) {
      var video = document.getElementById('gum-local');
      video.srcObject = mediaStream;
      video.onloadedmetadata = function (e) {
        video.play();
      };
    })
    .catch(function (err) { document.getElementById('video-error').innerHTML = (err.name + ": " + err.message); }); // always check for errors at the end.
}

function getLocation(id) {
  if (navigator.geolocation) {

    navigator.geolocation.getCurrentPosition(function (position) {

      var latitude = position.coords.latitude,
        longitude = position.coords.longitude;

      document.getElementById(id).innerHTML = "Latitude: " + latitude + ", Longitude: " + longitude;

    }, handleError);

    function handleError(error) {
      //Handle Errors
      switch (error.code) {
        case error.PERMISSION_DENIED:
          document.getElementById(id).innerHTML = "User denied the request for Geolocation.";
          break;
        case error.POSITION_UNAVAILABLE:
          document.getElementById(id).innerHTML = "Location information is unavailable.";
          break;
        case error.TIMEOUT:
          document.getElementById(id).innerHTML = "The request to get user location timed out.";
          break;
        case error.UNKNOWN_ERROR:
          document.getElementById(id).innerHTML = "An unknown error occurred.";
          break;
      }
    }
  } else {
    document.getElementById(id).innerHTML = "Geolocation is not Supported for this browser/OS";
  }
}

window.addEventListener('DOMContentLoaded', (event) => {
  document.getElementById('showVideo').addEventListener('click', e => startVideoCapture());
  document.getElementById('show-location').addEventListener('click', e => getLocation('location-click'));
});

window.onload = function () {
  getLocation('location-startup');
}
