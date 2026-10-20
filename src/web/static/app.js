const form = document.querySelector('#jobForm');
const fileInput = document.querySelector('#video');
const state = document.querySelector('#jobStatus');
const errorBox = document.querySelector('#error');
const progressArea = document.querySelector('#progress');
const progressText = document.querySelector('#progressText');
const progressFill = document.querySelector('#progressFill');
const resultsArea = document.querySelector('#results');
const blankArea = document.querySelector('#empty');
const byId = id => document.getElementById(id);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function escapeText(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
fileInput.addEventListener('change', () => {
  byId('upload-label').textContent = fileInput.files[0]?.name || 'Choose your traffic video';
});
function fileUrl(id, name) { return `/api/jobs/${encodeURIComponent(id)}/files/${encodeURIComponent(name)}`; }
function showResults(id, summary) {
  blankArea.hidden = true; progressArea.hidden = true; resultsArea.hidden = false;
  byId('crossings').textContent = summary.cumulative_crossings;
  byId('congestions').textContent = summary.congestion_events;
  byId('duration').textContent = `${summary.duration_seconds.toFixed(1)}s`;
  const player = byId('player'); player.src = fileUrl(id, summary.video); player.load();
  byId('preview').src = fileUrl(id, 'preview.jpg');
  byId('breakdown').innerHTML = Object.entries(summary.flow_by_class).map(([label,count]) => `<span>${escapeText(label)}: ${count} crossings</span>`).join('');
  const history = summary.events.length ? summary.events.map(ev => {
    const alert = ev.alert ? `<small>SMS: ${escapeText(ev.alert.status)} (${escapeText(ev.alert.mode)})</small>` : '';
    const message = ev.alert ? `<details><summary>View SMS content</summary><pre>${escapeText(ev.alert.message)}</pre></details>` : '';
    return `<div><strong>${escapeText(ev.type.replaceAll('_',' '))}</strong><br><small>At ${ev.timestamp_seconds}s · ${ev.occupancy} vehicles</small><br>${alert}${message}<br><a href="${fileUrl(id, ev.screenshot)}" target="_blank" rel="noopener">View evidence frame ↗</a></div>`;
  }).join('') : '<div>No sustained congestion detected with these settings.</div>';
  byId('events').innerHTML = history;
  const files = [['annotated.mp4','Annotated Video'],['preview.jpg','Preview'],['summary.json','Summary JSON'],['counts.csv','Counts CSV'],['events.json','Events JSON'],['alerts.json','SMS Audit JSON']];
  byId('downloads').innerHTML = files.map(([name,label]) => `<a href="${fileUrl(id,name)}" download>${label} ↘</a>`).join('');
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = byId('run'); button.disabled = true;
  errorBox.hidden = true; resultsArea.hidden = true; blankArea.hidden = true; progressArea.hidden = false;
  progressFill.style.width = '2%'; progressText.textContent = 'Uploading video…'; state.textContent = 'UPLOADING';
  try {
    const response = await fetch('/api/jobs', {method:'POST', body:new FormData(form)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    const job = data.job_id;
    while (true) {
      await sleep(1200);
      const statusResponse = await fetch(`/api/jobs/${encodeURIComponent(job)}`);
      if (!statusResponse.ok) throw new Error('Unable to read job status');
      const status = await statusResponse.json();
      state.textContent = status.status.toUpperCase();
      const percent = status.total_frames ? Math.min(97, Math.round(status.frames/status.total_frames*97)) : 15;
      progressFill.style.width = `${percent}%`;
      progressText.textContent = `${status.status} · ${status.frames} / ${status.total_frames || '?'} frames processed`;
      if (status.status === 'completed') { progressFill.style.width = '100%'; showResults(job, status.summary); break; }
      if (status.status === 'failed') throw new Error(status.error || 'Video processing failed');
    }
  } catch (error) {
    state.textContent = 'ERROR'; errorBox.hidden = false; errorBox.textContent = error.message;
    progressArea.hidden = true;
  } finally { button.disabled = false; }
});
