/* Pure selection/summary/export logic shared by the console and Node regression tests. */
(function (root) {
  function select(tasks, filter = 'all', query = '') {
    const needle = query.trim().toLocaleLowerCase();
    return tasks.filter(task => {
      const matches = filter === 'all' || (filter === 'unconfirmed'
        ? task.stop_confirmation === 'unconfirmed' : task.phase === filter);
      const text = [task.run_id, task.instruction, task.message, task.stop_error,
        task.failure?.title, task.failure?.detail, task.failure?.recovery,
        ...(task.events || []).map(event => event.text)].filter(Boolean).join('\n').toLocaleLowerCase();
      return matches && (!needle || text.includes(needle));
    });
  }
  function summarize(tasks) {
    const outcomes = {success: 0, failed: 0, stopped: 0, other: 0};
    const durations = [], failures = new Map();
    let unconfirmed = 0, pending = 0;
    for (const task of tasks) {
      const phase = ['success', 'failed', 'stopped'].includes(task.phase) ? task.phase : 'other';
      outcomes[phase]++;
      if (task.stop_confirmation === 'unconfirmed') unconfirmed++;
      if (task.stop_confirmation === 'pending') pending++;
      const duration = task.finished_at - task.started_at;
      if (Number.isFinite(task.started_at) && Number.isFinite(task.finished_at) &&
          Number.isFinite(duration) && duration >= 0) durations.push(duration);
      if (phase === 'failed') {
        const title = task.failure?.title?.trim() || '未分类失败';
        failures.set(title, (failures.get(title) || 0) + 1);
      }
    }
    durations.sort((a, b) => a - b);
    const middle = Math.floor(durations.length / 2);
    const median = durations.length ? (durations.length % 2 ? durations[middle]
      : durations[middle - 1] / 2 + durations[middle] / 2) : null;
    return {total: tasks.length, outcomes,
      completion_ratio: tasks.length ? outcomes.success / tasks.length : null,
      stop_confirmation: {unconfirmed, pending},
      duration_seconds: {count: durations.length, excluded: tasks.length - durations.length,
        median, max: durations.length ? durations[durations.length - 1] : null},
      failure_groups: [...failures].map(([title, count]) => ({title, count}))
        .sort((a, b) => b.count - a.count || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0))};
  }
  function exportData(data, tasks, filter, query, stale) {
    return {scope: data.scope, limit: data.limit, storage_warning: data.storage_warning || '', exported_at: new Date().toISOString(),
      selection: {filter, query}, snapshot_stale: stale, summary: summarize(tasks), tasks};
  }
  const api = {select, summarize, exportData};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TaskHistory = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
