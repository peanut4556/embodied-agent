/* Pure selection/export logic shared by the console and Node regression tests. */
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
  function exportData(data, tasks, filter, query, stale) {
    return {scope: data.scope, limit: data.limit, exported_at: new Date().toISOString(),
      selection: {filter, query}, snapshot_stale: stale, tasks};
  }
  const api = {select, exportData};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TaskHistory = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
