
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

function getLocation() {
  if (navigator.geolocation) {

    navigator.geolocation.getCurrentPosition(function (position) {

      var latitude = position.coords.latitude,
        longitude = position.coords.longitude;

      document.getElementById("location").innerHTML = "Latitude: " + latitude + ", Longitude: " + longitude;

    }, handleError);

    function handleError(error) {
      //Handle Errors
      switch (error.code) {
        case error.PERMISSION_DENIED:
          document.getElementById("location").innerHTML = "User denied the request for Geolocation.";
          break;
        case error.POSITION_UNAVAILABLE:
          document.getElementById("location").innerHTML = "Location information is unavailable.";
          break;
        case error.TIMEOUT:
          document.getElementById("location").innerHTML = "The request to get user location timed out.";
          break;
        case error.UNKNOWN_ERROR:
          document.getElementById("location").innerHTML = "An unknown error occurred.";
          break;
      }
    }
  } else {
    document.getElementById("location").innerHTML = "Geolocation is not Supported for this browser/OS";
  }
}

window.addEventListener('DOMContentLoaded', (event) => {
  document.querySelector('#showVideo').addEventListener('click', e => startVideoCapture());
});

window.onload = function () {
  getLocation();
}
