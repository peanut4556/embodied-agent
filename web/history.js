/* Pure selection/summary/export logic shared by the console and Node regression tests. */
(function (root) {
  function dateRange(start = '', end = '') {
    function parse(value) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw Error('请输入有效日期。');
      const [year, month, day] = value.split('-').map(Number);
      const date = new Date(0);
      date.setFullYear(year, month - 1, day);date.setHours(0, 0, 0, 0);
      if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day)
        throw Error('请输入有效日期。');
      return date;
    }
    const from = start ? parse(start) : null, through = end ? parse(end) : null;
    if (from && through && from > through) throw Error('开始日期不能晚于结束日期。');
    // Advance the calendar day, not 24 hours: DST days can be shorter or longer.
    if (through) through.setDate(through.getDate() + 1);
    return {start_date: start, end_date: end,
      from_inclusive: from ? from.getTime() / 1000 : null,
      until_exclusive: through ? through.getTime() / 1000 : null};
  }
  function select(tasks, filter = 'all', query = '', range = null) {
    const needle = query.trim().toLocaleLowerCase();
    return tasks.filter(task => {
      const matches = filter === 'all' || (filter === 'unconfirmed'
        ? task.stop_confirmation === 'unconfirmed' : task.phase === filter);
      const text = [task.run_id, task.instruction, task.message, task.stop_error,
        task.failure?.title, task.failure?.detail, task.failure?.recovery,
        ...(task.events || []).map(event => event.text)].filter(Boolean).join('\n').toLocaleLowerCase();
      const bounded = range && (range.from_inclusive !== null || range.until_exclusive !== null);
      const inRange = !bounded || (Number.isFinite(task.finished_at) &&
        (range.from_inclusive === null || task.finished_at >= range.from_inclusive) &&
        (range.until_exclusive === null || task.finished_at < range.until_exclusive));
      return matches && inRange && (!needle || text.includes(needle));
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
  function exportData(data, tasks, filter, query, stale, range = null) {
    return {scope: data.scope, limit: data.limit, storage_warning: data.storage_warning || '', exported_at: new Date().toISOString(),
      selection: {filter, query, time_range: range}, snapshot_stale: stale, summary: summarize(tasks), tasks};
  }
  const api = {select, summarize, exportData, dateRange};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TaskHistory = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
