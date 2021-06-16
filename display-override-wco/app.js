const registerServiceWorker = async () => {
  try {
    await navigator.serviceWorker.register('/pwa/display-override-wco/sw.js', { scope: '/pwa/display-override-wco/' });
    console.log('Service worker registered');
  } catch (e) {
    console.log(`Registration failed: ${e}`);
  }
}

if (navigator.serviceWorker) {
  registerServiceWorker();
}

// Button handler for changing theme-color via meta tag
const updateThemeColor = (newColor) => {
  var themeColorElement = document.querySelector("meta[name=theme-color]");

  if (!themeColorElement) {
    themeColorElement = document.createElement('meta');
    themeColorElement.name = "theme-color";
    document.head.appendChild(themeColorElement);
  }
  themeColorElement.setAttribute("content", newColor);
}

const removeThemeColorTag = () => {
  var themeColorElement = document.querySelector("meta[name=theme-color]");
  if (themeColorElement) {
    themeColorElement.remove();
  }
}

const handleApplyThemeColorClick = () => {
  const newThemeColorString = document.getElementById("themeColor").value;
  if (newThemeColorString) {
    updateThemeColor(newThemeColorString);
  } else {
    removeThemeColorTag();
  }
}

const handleThemeColorKeyDown = (e) => {
  if (e.keyCode == 13) {
    handleApplyThemeColorClick();
  }
}

const updatewindowControlsOverlayInfo = () => {

  const windowControlsOverlayJSDiv = document.getElementById('windowControlsOverlayJS');
  const windowControlsOverlayCSSDiv = document.getElementById('windowControlsOverlayCSS');
  const geometryChangeCountDiv = document.getElementById('windowControlsOverlayGeometryChange');
  const resizeCountDiv = document.getElementById('resizeCount');
  const errorDiv = document.getElementById("error");

  geometryChangeCountDiv.textContent = `geometrychange count: ${geometryChangeCount}`;
  resizeCountDiv.textContent = `resize count: ${resizeCount}`;

  if (!navigator.windowControlsOverlay) {
    errorDiv.innerText = 'navigator.windowControlsOverlay not defined';
    errorDiv.style.visibility = 'visible';
    windowControlsOverlayJSDiv.style.visibility = 'hidden';
    windowControlsOverlayCSSDiv.style.visibility = 'hidden';
    console.error(errorDiv.innerText);
    return;
  }

  errorDiv.style.visibility = 'hidden';
  windowControlsOverlayJSDiv.style.visibility = 'visible';
  windowControlsOverlayCSSDiv.style.visibility = 'visible';

  const boundingClientRect = navigator.windowControlsOverlay.getBoundingClientRect();
  windowControlsOverlayJSDiv.innerText = `navigator.windowControlsOverlay.visible = ${navigator.windowControlsOverlay.visible}
navigator.windowControlsOverlay.getBoundingClientRect() = {
x: ${boundingClientRect.x},
y: ${boundingClientRect.y},
width: ${boundingClientRect.width},
height: ${boundingClientRect.height}
}`;

  const windowControlsOverlayElementStyle = document.getElementById('windowControlsOverlayElementStyle');
  const x = getComputedStyle(windowControlsOverlayElementStyle).getPropertyValue('padding-left');
  const w = getComputedStyle(windowControlsOverlayElementStyle).getPropertyValue('padding-right');
  const y = getComputedStyle(windowControlsOverlayElementStyle).getPropertyValue('padding-top');
  const h = getComputedStyle(windowControlsOverlayElementStyle).getPropertyValue('padding-bottom');

  windowControlsOverlayCSSDiv.innerText = `titlebar-area-x: ${x},
titlebar-area-width: ${w},
titlebar-area-y: ${y},
titlebar-area-height: ${h}`;
}

let geometryChangeCount = 0;
let resizeCount = 0;

const handleGeometryChange = () => {
  geometryChangeCount++;
  updatewindowControlsOverlayInfo();
}

const handleResize = () => {
  resizeCount++;
  updatewindowControlsOverlayInfo();
}

window.addEventListener('DOMContentLoaded', (event) => {
  console.log('DOM fully loaded and parsed');

  if (navigator.windowControlsOverlay) {
    navigator.windowControlsOverlay.addEventListener('geometrychange', handleGeometryChange);
  } else {
    window.addEventListener('resize', handleResize);
  }

  if (document.getElementById('themeColor')) {
    document.getElementById('themeColor').addEventListener('keydown', handleThemeColorKeyDown, false);
    document.getElementById('applyThemeColor').addEventListener('click', handleApplyThemeColorClick, false);
  }

  if (document.getElementById('overflowMainContent')) {
    panzoom(document.getElementById('overflowMainContent'), {
      maxZoom: 2.0,
      minZoom: 1.0,
      bounds: true,
      boundsPadding: 0.1,
      onTouch: function (e) {
        return false;
      },
      beforeWheel: function (e) {
        return true;
      },
      beforeMouseDown: function (e) {
        return true;
      },
      filterKey: function () {
        return true;
      }

    });
  }

  updatewindowControlsOverlayInfo();
});
