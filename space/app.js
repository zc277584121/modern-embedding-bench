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

function taskDatasetOptions() {
  const seen = new Set();
  return state.payload.results.flatMap((record) => {
    const value = `${record.task_id}|${record.dataset_id}|${record.dataset_version}`;
    if (seen.has(value)) return [];
    seen.add(value);
    const task = state.tasks.get(record.task_id);
    return [{
      value,
      label: `${task.title} · ${record.dataset_id} (${record.dataset_version})`,
    }];
  });
}

function renderTask(selection) {
  const [taskId, datasetId, datasetVersion] = selection.split("|");
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
  const ordered = [...grouped.keys()].sort(
    (left, right) =>
      grouped.get(right).get(task.primary_metric).value -
      grouped.get(left).get(task.primary_metric).value,
  );
  const rows = ordered.map((solutionId, index) => {
    const solution = state.solutions.get(solutionId);
    const records = grouped.get(solutionId);
    const metricCells = metricOrder.map((metricId) => {
      const record = records.get(metricId);
      if (!record) return '<span class="muted">—</span>';
      const primaryClass = metricId === task.primary_metric ? " primary" : "";
      return `<span class="score${primaryClass}">${formatScore(record.value)}</span>`;
    });
    return [
      `<span class="rank">#${index + 1}</span>`,
      `<strong>${escapeHtml(solution.title)}</strong><br><span class="muted">${escapeHtml(solution.description)}</span>`,
      ...metricCells,
      '<span class="pill">Demo</span>',
      sourceLink(records.values().next().value),
    ];
  });
  document.querySelector("#task-context").innerHTML = `
    <div class="context"><h2>${escapeHtml(task.title)}</h2>
    <p>${escapeHtml(task.description)} Dataset: ${escapeHtml(datasetId)} · ${escapeHtml(datasetVersion)}</p></div>`;
  document.querySelector("#task-table").innerHTML = renderTable(
    ["Rank", "Solution", ...metricOrder.map((id) => labels.get(id)), "Status", "Source"],
    rows,
  );
}

function renderSolution(solutionId) {
  const solution = state.solutions.get(solutionId);
  const records = primaryRecords()
    .filter((record) => record.solution_id === solutionId)
    .sort((left, right) => left.task_id.localeCompare(right.task_id));
  const rows = records.map((record) => {
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
      `<strong>${escapeHtml(state.tasks.get(record.task_id).title)}</strong>`,
      escapeHtml(record.metric_label),
      `<span class="score primary">${formatScore(record.value)}</span>`,
      `<span class="rank">#${rank} / ${peers.length}</span>`,
      `${escapeHtml(record.dataset_id)}<br><span class="muted">${escapeHtml(record.dataset_version)}</span>`,
      sourceLink(record),
    ];
  });
  document.querySelector("#solution-context").innerHTML = `
    <div class="context"><h2>${escapeHtml(solution.title)}</h2>
    <p>${escapeHtml(solution.description)}</p></div>`;
  document.querySelector("#solution-table").innerHTML = renderTable(
    ["Task", "Primary metric", "Score", "Rank", "Dataset", "Source"],
    rows,
  );
}

function renderMatrix() {
  const taskIds = [...state.tasks.keys()].sort();
  const primary = primaryRecords();
  const rows = [...state.solutions.entries()].map(([solutionId, solution]) => {
    const cells = [`<strong>${escapeHtml(solution.title)}</strong>`];
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
  document.querySelector("#matrix-table").innerHTML = renderTable(
    ["Solution", ...taskIds.map((taskId) => state.tasks.get(taskId).title)],
    rows,
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

async function main() {
  const response = await fetch("data/results.json");
  if (!response.ok) throw new Error(`Unable to load benchmark data: ${response.status}`);
  state.payload = await response.json();
  state.tasks = new Map(state.payload.tasks.map((task) => [task.id, task]));
  state.solutions = new Map(
    state.payload.solutions.map((solution) => [solution.id, solution]),
  );

  document.querySelector("#preview-notice").innerHTML =
    `<strong>Framework preview.</strong> ${escapeHtml(state.payload.benchmark.notice)}`;
  document.querySelector("#stats").innerHTML = [
    [state.tasks.size, "Modern IR tasks"],
    [state.solutions.size, "Solutions"],
    [state.payload.results.length, "Measurements"],
    [state.payload.benchmark.release, "Active release"],
  ]
    .map(([value, label]) => `<div class="stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`)
    .join("");

  const taskSelector = document.querySelector("#task-selector");
  taskDatasetOptions().forEach((option) => taskSelector.add(new Option(option.label, option.value)));
  taskSelector.addEventListener("change", () => renderTask(taskSelector.value));

  const solutionSelector = document.querySelector("#solution-selector");
  [...state.solutions.entries()].forEach(([id, solution]) =>
    solutionSelector.add(new Option(solution.title, id)),
  );
  solutionSelector.addEventListener("change", () => renderSolution(solutionSelector.value));

  installTabs();
  renderTask(taskSelector.value);
  renderSolution(solutionSelector.value);
  renderMatrix();
}

main().catch((error) => {
  document.querySelector("#preview-notice").textContent = error.message;
  console.error(error);
});
