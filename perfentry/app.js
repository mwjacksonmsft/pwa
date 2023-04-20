
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

// Grid Options are properties passed to the grid
const gridOptions = {

  // each entry here represents one column
  columnDefs: [
    { field: 'order', headerName: 'Order', width: 50 },
    { field: 'entryType', headerName: 'Entry Type', width: 150 },
    { field: 'name', headerName: 'Name' },
    { field: 'startTime', headerName: 'Start Time' },
    { field: 'duration', headerName: 'Duration' },
  ],

  // default col def properties get applied to all columns
  defaultColDef: {sortable: true, filter: true, resizable: true},
};

if (navigator.serviceWorker) {
  registerServiceWorker();
}

let allPerfEntries = [];

function renderDataInTheTable(list, observer) {
  for (let entry of list.getEntries()) {
    entry['order'] = allPerfEntries.length;
    allPerfEntries.push(entry);
  }
  gridOptions.api.setRowData(allPerfEntries);
}

window.addEventListener('DOMContentLoaded', (event) => {
  console.log('DOM fully loaded and parsed');

  // get div to host the grid
  const eGridDiv = document.getElementById('perfGrid');
  // new grid instance, passing in the hosting DIV and Grid Options
  new agGrid.Grid(eGridDiv, gridOptions);

  const observer = new PerformanceObserver(renderDataInTheTable);

  // 'script', << not supported
  [
    'back-forward-cache-restoration', 'element', 'event', 'first-input',
    'largest-contentful-paint', 'layout-shift', 'long-animation-frame',
    'longtask', 'mark', 'measure', 'navigation', 'paint', 'resource',
    'soft-navigation', 'taskattribution', 'visibility-state'
  ].forEach((type) => {
    console.log('Observing: ' + type);
    observer.observe({ type, buffered: true })
  });

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

// Element
generateElementButton.addEventListener('click', event => {

  if (document.getElementById('elementTimingTarget')) {
    var element = document.getElementById('elementTimingTarget');
    element.parentNode.removeChild(element);
  }

  const backgroundDiv = document.createElement("div");
  backgroundDiv.id = "elementTimingTarget";
  backgroundDiv.elementtiming = "my_div"
  document.body.appendChild(backgroundDiv);
});

// Long Animation Frame
generateLongAnimationFrameButton.addEventListener('click', event => {
  const img = document.createElement("img");
  img.src = "/pwa/perfentry/green.png";
  img.addEventListener("load", () => {
      busy_wait();
  });
  img.id = "image";
  document.body.appendChild(img);
});

// Long Task
generateLongTaskButton.addEventListener('click', event => {
});

// Mark
generateMarkButton.addEventListener('click', event => {
  window.performance.mark('mark_clicked');
});

// Measure
var reqCnt = 0;
generateMeasureButton.addEventListener('click', event => {
  reqCnt++;
  window.performance.mark('measure_clicked');
  window.performance.measure('measure_load_from_dom' + reqCnt, 'domComplete', 'measure_clicked');
});

// Navigation
generateNavigationButton.addEventListener('click', event => {
});

// Paint
generatePaintButton.addEventListener('click', event => {
});

// Resource
generateResourceButton.addEventListener('click', event => {
  let resourceState = document.getElementById('resourceState');

  var xhr = new XMLHttpRequest();
  xhr.open('GET', 'https://mwjacksonmsft.github.io/pwa/perfentry/xmlhttprequestpayload.txt', true);

  xhr.addEventListener("progress", e => {
    resourceState.innerText = 'request in progress';
    console.log(resourceState.innerText);
  });

  xhr.addEventListener("load", e => {
    resourceState.innerText = xhr.responseText;
    console.log(resourceState.innerText);
  });

  xhr.addEventListener("error", e => {
    resourceState.innerText = 'error';
    console.log(resourceState.innerText);
  });

  xhr.addEventListener("abort", e => {
    resourceState.innerText = 'abort';
    console.log(resourceState.innerText);
  });

  xhr.send();
});

// Soft Navigation
generateSoftNavigationButton.addEventListener('click', event => {
});

// Task Attribution
generateTaskAttributionButton.addEventListener('click', event => {
});

// Visibility State
generateVisibilityStateButton.addEventListener('click', event => {
});
