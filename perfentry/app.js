
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/perfentry/sw.js', { scope: '/pwa/perfentry/'});
      console.log('Service worker registered');
  } catch (e) {
      console.log(`Registration failed: ${e}`);
  }
}

const very_long_frame_duration = 360;
function busy_wait(ms_delay = very_long_frame_duration) {
  const deadline = performance.now() + ms_delay;
  while (performance.now() < deadline) {}
}

function removeElementById(id) {
  let element = document.getElementById(id);
  if (element) {
    element.parentNode.removeChild(element);
  }
}

// Grid Options are properties passed to the grid
const gridOptions = {

  // each entry here represents one column
  columnDefs: [
    { field: 'order', headerName: 'Order', width: 100 },
    { field: 'entryType', headerName: 'Entry Type', width: 170 },
    { field: 'name', headerName: 'Name', width: 300 },
    { field: 'startTime', headerName: 'Start Time' },
    { field: 'duration', headerName: 'Duration' },
  ],

  // default col def properties get applied to all columns
  defaultColDef: {sortable: true, filter: true, resizable: true},
};

if (navigator.serviceWorker) {
  registerServiceWorker();
}

let gridOrder = 0;
function renderDataInTheTable(list, observer) {
  let newItems = [];

  for (let entry of list.getEntries()) {
    const clone = JSON.parse(JSON.stringify(entry));
    gridOrder++;
    clone['order'] = gridOrder;
    newItems.push(clone);
  }

  const res = gridOptions.api.applyTransaction({
    add: newItems,
  });
}

const [navigationEntries] = window.performance.getEntriesByType('navigation');
if (navigationEntries.confidence) {
  const confidence = document.createElement('div');
  confidence.id = 'confidence';
  confidence.innerText = "value == " + navigationEntries.confidence.value + ", randomizedTriggerRate == " + navigationEntries.confidence.randomizedTriggerRate;
  document.getElementById('insertItem').appendChild(confidence);
}

window.addEventListener('DOMContentLoaded', (event) => {
  console.log('DOM fully loaded and parsed');

  // get div to host the grid
  const eGridDiv = document.getElementById('perfGrid');
  // new grid instance, passing in the hosting DIV and Grid Options
  new agGrid.Grid(eGridDiv, gridOptions);

  if (window.PerformanceLongAnimationFrameTiming) {
    document.getElementById('requirementsMet').innerText = "You have experimental web platform features enabled.";
  }

  const observer = new PerformanceObserver(renderDataInTheTable);

  //'script' << not supported
  [
    'back-forward-cache-restoration', 'element', 'event', 'first-input',
    'largest-contentful-paint', 'layout-shift', 'long-animation-frame',
    'longtask', 'mark', 'measure', 'navigation', 'paint', 'resource',
    'soft-navigation', 'taskattribution', 'visibility-state'
  ].forEach((type) => {
    console.log('Observing: ' + type);
    observer.observe({ type, buffered: true, includeSoftNavigationObservations: true })
  });


  const navigationEntries = window.performance.getEntriesByType('navigation');
  let systemEntropy = 'none';
  if (navigationEntries.length > 0) {
    const navigationEntry = navigationEntries[0];
    if (navigationEntry.systemEntropy) {
      systemEntropy = navigationEntry.systemEntropy
    }
  }
  let systemEntropyState = document.getElementById('systemEntropy');
  systemEntropyState.innerText = 'System Entropy: ' + systemEntropy;
});


// BFCache Handlers
window.addEventListener('pageshow', (event) => {
  let bfCacheState = document.getElementById('bfCacheState');
  if (event.persisted) {
    bfCacheState.innerText = 'This page was restored from the bfcache.';
  } else {
    bfCacheState.innerText =  'This page was loaded normally.';
  }
  console.log(bfCacheState.innerText);
});

window.addEventListener('pagehide', (event) => {
  let bfCacheState = document.getElementById('bfCacheState');
  if (event.persisted) {
    bfCacheState.innerText = 'This page *might* be entering the bfcache.';
  } else {
    bfCacheState.innerText = 'This page will unload normally and be discarded.';
  }
  console.log(bfCacheState.innerText);
});

// Element and Layout Shift
generateElementButton.addEventListener('click', (event) => {
  removeElementById('elementTimingTarget');

  const backgroundDiv = document.createElement('div');
  backgroundDiv.id = 'elementTimingTarget';
  backgroundDiv.setAttribute('elementtiming', 'my_div');
  document.getElementById('insertItem').appendChild(backgroundDiv);
});

// Long Animation Frame and Long Task
generateLongAnimationFrameButton.addEventListener('click', event => {
  removeElementById('animationFrameImage');

  const img = document.createElement('img');
  img.src = '/pwa/perfentry/green.png';
  img.addEventListener('load', () => {
      busy_wait();
  });
  img.id = 'animationFrameImage';
  document.getElementById('insertItem').appendChild(img);
});

// Mark
generateMarkButton.addEventListener('click', (event) => {
  window.performance.mark('mark_clicked');
});

// Measure
var reqCnt = 0;
generateMeasureButton.addEventListener('click', (event) => {
  reqCnt++;
  window.performance.mark('measure_clicked');
  window.performance.measure('measure_load_from_dom' + reqCnt, 'domComplete', 'measure_clicked');
});

// Resource
generateResourceButton.addEventListener('click', (event) => {
  let resourceState = document.getElementById('resourceState');

  var xhr = new XMLHttpRequest();
  xhr.open('GET', 'https://mwjacksonmsft.github.io/pwa/perfentry/xmlhttprequestpayload.txt', true);

  xhr.addEventListener('progress', (e) => {
    resourceState.innerText = 'request in progress';
    console.log(resourceState.innerText);
  });

  xhr.addEventListener('load', (e) => {
    resourceState.innerText = xhr.responseText;
    console.log(resourceState.innerText);
  });

  xhr.addEventListener('error', (e) => {
    resourceState.innerText = 'error';
    console.log(resourceState.innerText);
  });

  xhr.addEventListener('abort', (e) => {
    resourceState.innerText = 'abort';
    console.log(resourceState.innerText);
  });

  xhr.send();
});

// Soft Navigation
let softNavigates = 0;
generateSoftNavigationButton.addEventListener('click', (event) => {
  removeElementById('elementTimingTarget');
  removeElementById('animationFrameImage');

  let resourceState = document.getElementById('resourceState');
  resourceState.innerText = 'Not Set';

  softNavigates++;
  let path = window.location.href.split('#')[0];
  window.location.href = path + '#' + softNavigates;

});

generateFencedFrameButton.addEventListener('click', (event) => {
  if (!window.HTMLFencedFrameElement) {
    let resourceState = document.getElementById('resourceState');

    resourceState.innerText = 'HTMLFencedFrameElement does not exist';
    console.log(resourceState.innerText);
    return;
  }

  var f = document.createElement('fencedframe');
  const url = new URL(
    '/pwa/perfentry/fencedframe.html',
    location.origin
  );
  f.config = new FencedFrameConfig(url);
  document.body.appendChild(f);
  document.getElementById('insertItem').appendChild(f);
});
