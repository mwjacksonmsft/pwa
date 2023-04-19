
const registerServiceWorker = async () => {
  try {
      await navigator.serviceWorker.register('/pwa/perfentry/sw.js', { scope: '/pwa/perfentry/'});
      console.log('Service worker registered');
  } catch (e) {
      console.log(`Registration failed: ${e}`);
  }
}

// Grid Options are properties passed to the grid
const gridOptions = {

  // each entry here represents one column
  columnDefs: [
    { field: "name" },
    { field: "entryType" },
    { field: "startTime" },
    { field: "duration" },
  ],

  // default col def properties get applied to all columns
  defaultColDef: {sortable: true, filter: true},
};

if (navigator.serviceWorker) {
  registerServiceWorker();
}

let allPerfEntries = [];

function renderDataInTheTable(list, observer) {
  /*
  const perfTable = document.getElementById("perfGrid");
  perfEntries.forEach(entry => {
      let newRow = document.createElement("tr");
      Object.values(entry).forEach((value) => {
          let cell = document.createElement("td");
          cell.innerText = value;
          newRow.appendChild(cell);
      })
      perfTable.appendChild(newRow);
  });
  */

  allPerfEntries.push(list.getEntries());
  gridOptions.api.setRowData(allPerfEntries);
}

window.addEventListener('DOMContentLoaded', (event) => {
  console.log('DOM fully loaded and parsed');

  // get div to host the grid
  const eGridDiv = document.getElementById("perfGrid");
  // new grid instance, passing in the hosting DIV and Grid Options
  new agGrid.Grid(eGridDiv, gridOptions);

  const observer = new PerformanceObserver(renderDataInTheTable);

  // "script", << not supported
  [
    "back-forward-cache-restoration", "element", "event", "first-input",
    "largest-contentful-paint", "layout-shift", "long-animation-frame",
    "longtask", "navigation", "paint", "resource", "soft-navigation",
    "taskattribution", "visibility-state"
  ].forEach((type) =>
    observer.observe({ type, buffered: true })
  );

  observer.observe({ entryTypes: ["measure", "mark"] });
});

