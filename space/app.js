const state = {
  payload: null,
  tasks: new Map(),
  solutions: new Map(),
};

const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const formatScore = (value) => Number(value).toFixed(3);

function sourceLink(record) {
  return `<a class="source" href="${escapeHtml(record.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(record.source_commit.slice(0, 7))} ↗</a>`;
}

function renderTable(headers, rows) {
  const head = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function primaryRecords() {
  return state.payload.results.filter((record) => record.primary);
}

function taskDatasets(taskId) {
  const seen = new Set();
  return state.payload.results.flatMap((record) => {
    if (record.task_id !== taskId) return [];
    const value = `${record.dataset_id}|${record.dataset_version}`;
    if (seen.has(value)) return [];
    seen.add(value);
    return [{ value, label: `${record.dataset_id} (${record.dataset_version})` }];
  });
}

function populateDatasetSelector() {
  const selector = document.querySelector("#dataset-selector");
  selector.replaceChildren();
  taskDatasets(document.querySelector("#task-selector").value).forEach((option) =>
    selector.add(new Option(option.label, option.value)),
  );
}

function renderTask() {
  const taskId = document.querySelector("#task-selector").value;
  const [datasetId, datasetVersion] = document
    .querySelector("#dataset-selector")
    .value.split("|");
  const query = document.querySelector("#task-search").value.trim().toLowerCase();
  const rowLimit = Number(document.querySelector("#task-rows").value);
  const task = state.tasks.get(taskId);
  const selected = state.payload.results.filter(
    (record) =>
      record.task_id === taskId &&
      record.dataset_id === datasetId &&
      record.dataset_version === datasetVersion,
  );
  const metricOrder = [...new Set(selected.map((record) => record.metric_id))].sort(
    (left, right) => Number(right === task.primary_metric) - Number(left === task.primary_metric),
  );
  const labels = new Map(selected.map((record) => [record.metric_id, record.metric_label]));
  const grouped = new Map();
  selected.forEach((record) => {
    if (!grouped.has(record.solution_id)) grouped.set(record.solution_id, new Map());
    grouped.get(record.solution_id).set(record.metric_id, record);
  });
  const matching = [...grouped.keys()]
    .filter((solutionId) => {
      const solution = state.solutions.get(solutionId);
      const searchable = `${solutionId} ${solution.title} ${solution.description} ${metricOrder.join(" ")}`;
      return searchable.toLowerCase().includes(query);
    })
    .sort(
      (left, right) =>
        grouped.get(right).get(task.primary_metric).value -
        grouped.get(left).get(task.primary_metric).value,
    );
  const shown = matching.slice(0, rowLimit);
  const rows = shown.map((solutionId) => {
    const solution = state.solutions.get(solutionId);
    const records = grouped.get(solutionId);
    const rank = matching.indexOf(solutionId) + 1;
    const metricCells = metricOrder.map((metricId) => {
      const record = records.get(metricId);
      if (!record) return '<span class="muted">—</span>';
      const primaryClass = metricId === task.primary_metric ? " primary" : "";
      return `<span class="score${primaryClass}">${formatScore(record.value)}</span>`;
    });
    return [
      `<span class="rank">${rank}</span>`,
      `<strong>${escapeHtml(solution.title)}</strong><br><span class="muted">${escapeHtml(solution.id)}</span>`,
      ...metricCells,
      '<span class="pill">Demo</span>',
      sourceLink(records.values().next().value),
    ];
  });

  document.querySelector("#task-context").innerHTML = `
    <strong>${escapeHtml(task.title)}</strong>
    <p>${escapeHtml(task.description)}</p>
    <p>Primary signal: <code>${escapeHtml(labels.get(task.primary_metric))}</code> · Dataset: <code>${escapeHtml(datasetId)}</code> · Version: <code>${escapeHtml(datasetVersion)}</code></p>`;
  document.querySelector("#task-status").innerHTML = matching.length
    ? `Showing <strong>${shown.length}</strong> of <strong>${matching.length}</strong> matching rows.`
    : "No rows match the current search.";
  document.querySelector("#task-table").innerHTML = renderTable(
    ["Rank", "Solution", ...metricOrder.map((id) => labels.get(id)), "Status", "Source"],
    rows,
  );
}

function renderSolution() {
  const solutionId = document.querySelector("#solution-selector").value;
  const query = document.querySelector("#solution-search").value.trim().toLowerCase();
  const rowLimit = Number(document.querySelector("#solution-rows").value);
  const solution = state.solutions.get(solutionId);
  const matching = primaryRecords()
    .filter((record) => record.solution_id === solutionId)
    .filter((record) => {
      const task = state.tasks.get(record.task_id);
      const searchable = `${task.title} ${record.task_id} ${record.dataset_id} ${record.metric_label}`;
      return searchable.toLowerCase().includes(query);
    })
    .sort((left, right) => left.task_id.localeCompare(right.task_id));
  const shown = matching.slice(0, rowLimit);
  const rows = shown.map((record) => {
    const peers = primaryRecords()
      .filter(
        (item) =>
          item.task_id === record.task_id &&
          item.dataset_id === record.dataset_id &&
          item.dataset_version === record.dataset_version,
      )
      .sort((left, right) => right.value - left.value);
    const rank = peers.findIndex((item) => item.solution_id === solutionId) + 1;
    return [
      `<strong>${escapeHtml(state.tasks.get(record.task_id).title)}</strong><br><span class="muted">${escapeHtml(record.task_id)}</span>`,
      escapeHtml(record.metric_label),
      `<span class="score primary">${formatScore(record.value)}</span>`,
      `<span class="rank">${rank} / ${peers.length}</span>`,
      `${escapeHtml(record.dataset_id)}<br><span class="muted">${escapeHtml(record.dataset_version)}</span>`,
      sourceLink(record),
    ];
  });

  document.querySelector("#solution-context").innerHTML = `
    <strong>${escapeHtml(solution.title)}</strong>
    <p>${escapeHtml(solution.description)}</p>
    <p>Solution ID: <code>${escapeHtml(solution.id)}</code></p>`;
  document.querySelector("#solution-status").innerHTML = matching.length
    ? `Showing <strong>${shown.length}</strong> of <strong>${matching.length}</strong> matching rows.`
    : "No rows match the current search.";
  document.querySelector("#solution-table").innerHTML = renderTable(
    ["Task", "Primary metric", "Score", "Rank", "Dataset release", "Source"],
    rows,
  );
}

function renderCoverage() {
  const taskIds = [...state.tasks.keys()].sort();
  const primary = primaryRecords();
  const rows = [...state.solutions.entries()].map(([solutionId, solution]) => {
    const cells = [
      `<strong>${escapeHtml(solution.title)}</strong><br><span class="muted">${escapeHtml(solutionId)}</span>`,
    ];
    taskIds.forEach((taskId) => {
      const record = primary.find(
        (item) => item.solution_id === solutionId && item.task_id === taskId,
      );
      if (!record) {
        cells.push('<span class="muted">Not evaluated</span>');
        return;
      }
      const peers = primary
        .filter((item) => item.task_id === taskId)
        .sort((left, right) => right.value - left.value);
      const rank = peers.findIndex((item) => item.solution_id === solutionId) + 1;
      cells.push(
        `<span class="score">${formatScore(record.value)}</span> ` +
          `<span class="rank">#${rank}</span><br>` +
          `<span class="muted">${escapeHtml(record.metric_label)}</span>`,
      );
    });
    return cells;
  });
  document.querySelector("#coverage-table").innerHTML = renderTable(
    ["Solution", ...taskIds.map((taskId) => state.tasks.get(taskId).title)],
    rows,
  );
}

function renderCatalogs() {
  const taskRows = [...state.tasks.values()].map((task) => {
    const releases = taskDatasets(task.id).map((item) => item.label).join(", ");
    return [
      `<strong>${escapeHtml(task.title)}</strong><br><span class="muted">${escapeHtml(task.id)}</span>`,
      escapeHtml(task.description),
      escapeHtml(task.primary_metric),
      escapeHtml(task.version),
      escapeHtml(releases),
    ];
  });
  document.querySelector("#task-catalog").innerHTML = renderTable(
    ["Task", "Description", "Primary metric", "Task version", "Dataset releases"],
    taskRows,
  );

  const solutionRows = [...state.solutions.values()].map((solution) => {
    const taskCount = new Set(
      state.payload.results
        .filter((record) => record.solution_id === solution.id)
        .map((record) => record.task_id),
    ).size;
    return [
      `<strong>${escapeHtml(solution.title)}</strong>`,
      escapeHtml(solution.id),
      escapeHtml(solution.description),
      String(taskCount),
    ];
  });
  document.querySelector("#solution-catalog").innerHTML = renderTable(
    ["Solution", "ID", "Description", "Evaluated tasks"],
    solutionRows,
  );
}

function installTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      document.querySelector(`#${button.dataset.panel}`).classList.add("active");
    });
  });
}

function installControls() {
  const taskSelector = document.querySelector("#task-selector");
  [...state.tasks.values()].forEach((task) => taskSelector.add(new Option(task.title, task.id)));
  taskSelector.addEventListener("change", () => {
    populateDatasetSelector();
    renderTask();
  });
  document.querySelector("#dataset-selector").addEventListener("change", renderTask);
  document.querySelector("#task-search").addEventListener("input", renderTask);
  document.querySelector("#task-rows").addEventListener("change", renderTask);

  const solutionSelector = document.querySelector("#solution-selector");
  [...state.solutions.values()].forEach((solution) =>
    solutionSelector.add(new Option(solution.title, solution.id)),
  );
  solutionSelector.addEventListener("change", renderSolution);
  document.querySelector("#solution-search").addEventListener("input", renderSolution);
  document.querySelector("#solution-rows").addEventListener("change", renderSolution);
}

async function main() {
  const response = await fetch("data/results.json");
  if (!response.ok) throw new Error(`Unable to load benchmark data: ${response.status}`);
  state.payload = await response.json();
  state.tasks = new Map(state.payload.tasks.map((task) => [task.id, task]));
  state.solutions = new Map(
    state.payload.solutions.map((solution) => [solution.id, solution]),
  );

  const pairCount = new Set(
    primaryRecords().map((record) => `${record.task_id}:${record.solution_id}`),
  ).size;
  const releaseCount = new Set(
    state.payload.results.map(
      (record) => `${record.task_id}:${record.dataset_id}:${record.dataset_version}`,
    ),
  ).size;
  document.querySelector("#summary").innerHTML = `
    Rows: <strong>${state.payload.results.length}</strong> | Tasks: <strong>${state.tasks.size}</strong> | Solutions: <strong>${state.solutions.size}</strong><br>
    Task/solution pairs: <strong>${pairCount}</strong> | Dataset releases: <strong>${releaseCount}</strong> | Release: <code>${escapeHtml(state.payload.benchmark.release)}</code>`;
  document.querySelector("#preview-notice").innerHTML =
    `<strong>Preview data:</strong> ${escapeHtml(state.payload.benchmark.notice)}`;

  installTabs();
  installControls();
  populateDatasetSelector();
  renderTask();
  renderSolution();
  renderCoverage();
  renderCatalogs();
}

main().catch((error) => {
  document.querySelector("#preview-notice").textContent = error.message;
  console.error(error);
});
