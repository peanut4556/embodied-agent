const {test}=require('node:test');
const assert=require('node:assert/strict');
const {select,summarize,exportData}=require('../../web/history.js');
const tasks=[
 {run_id:'one',phase:'failed',instruction:'抓取红块',failure:{detail:'Camera OFFLINE'},stop_confirmation:'confirmed'},
 {run_id:'two',phase:'stopped',instruction:'放进盒子',stop_confirmation:'unconfirmed'},
 {run_id:'three',phase:'success',instruction:'红块入盒',events:[{text:'验收通过'}]}
];
test('filters status and stop confirmation independently',()=>{
 assert.deepEqual(select(tasks,'failed').map(t=>t.run_id),['one']);
 assert.deepEqual(select(tasks,'unconfirmed').map(t=>t.run_id),['two']);
 assert.equal(select(tasks,'success').length,1);
});
test('combines case-insensitive search and status',()=>{
 assert.deepEqual(select(tasks,'failed',' offline ').map(t=>t.run_id),['one']);
 assert.deepEqual(select(tasks,'success','验收').map(t=>t.run_id),['three']);
 assert.equal(select(tasks,'success','offline').length,0);
 assert.equal(select(tasks,'all','two')[0].run_id,'two');
});
test('preserves input, order and treats markup as literal text',()=>{
 const input=JSON.stringify(tasks);assert.equal(select(tasks).length,3);
 assert.equal(JSON.stringify(tasks),input);
 assert.equal(select(tasks,'all','<script>').length,0);
 assert.equal(select([{instruction:'<script>',phase:'failed'}],'all','<script>').length,1);
});
test('exports precisely selected rows and snapshot freshness',()=>{
 const rows=select(tasks,'failed');
 const result=exportData({scope:'session',limit:20},rows,'failed','',true);
 const decoded=JSON.parse(JSON.stringify(result));
 assert.equal(decoded.tasks.length,1);assert.equal(decoded.tasks[0].run_id,'one');
 assert.equal(decoded.snapshot_stale,true);assert.equal(decoded.selection.filter,'failed');
 assert.equal(decoded.scope,'session');assert.ok(Number.isFinite(Date.parse(decoded.exported_at)));
});
test('empty and single exports never include hidden rows',()=>{
 assert.equal(exportData({},[],'failed','missing',false).tasks.length,0);
 assert.deepEqual(exportData({},[tasks[1]],'single','two',false).tasks,[tasks[1]]);
});

const fs=require('node:fs');
const vm=require('node:vm');
async function page(){
 const html=fs.readFileSync(require('node:path').join(__dirname,'../../web/simulation.html'),'utf8');
 const script=html.slice(html.indexOf('let historyData='),html.indexOf('\nfunction point'));
 class Element{
  constructor(){this.children=[];this.value='';this.textContent='';this.style={};this.isConnected=true;}
  append(...children){this.children.push(...children);}
  replaceChildren(){for(const child of this.children)child.isConnected=false;this.children=[];}
  click(){this.onclick?.();}
 }
 const elements={};const $=id=>elements[id]??=(new Element());
 $('history-filter').value='all';
 let rows=tasks.map(t=>({...t,started_at:1,finished_at:2,plan:[],events:t.events||[],message:'result'}));
 let fail=false, blob;
 const context=vm.createContext({$,TaskHistory:{select,summarize,exportData},
  document:{createElement:()=>new Element()},Date,Set,JSON,Blob,
  AbortSignal:{timeout:()=>undefined},setTimeout:()=>{},setInterval:()=>{},
  URL:{createObjectURL:value=>{blob=value;return 'blob:test';},revokeObjectURL:()=>{}},
  fetch:async()=>({ok:!fail,json:async()=>({tasks:rows,limit:20,scope:'session'})})});
 vm.runInContext(script,context);
 await new Promise(resolve=>setImmediate(resolve));
 return {$,context,setRows:value=>{rows=value;},setFail:value=>{fail=value;},getExport:async()=>JSON.parse(await blob.text())};
}
test('rendered filtering and batch download use the same records',async()=>{
 const p=await page();assert.equal(p.$('history').children.length,3);
 p.$('history-filter').value='failed';p.$('history-filter').onchange();
 assert.equal(p.$('history').children.length,1);
 assert.match(p.$('history-state').textContent,/1 \/ 3/);
 p.$('export-history').click();assert.equal((await p.getExport()).tasks[0].run_id,'one');
 p.$('history-search').value='missing';p.$('history-search').oninput();
 assert.equal(p.$('history').children.length,0);assert.equal(p.$('export-history').disabled,true);
});
test('single download includes only its task and failure marks stale data',async()=>{
 const p=await page();p.setFail(true);await vm.runInContext('refreshHistory()',p.context);
 assert.match(p.$('history-state').textContent,/旧快照/);
 p.$('history').children[1].children[2].click();
 const result=await p.getExport();assert.equal(result.tasks.length,1);
 assert.equal(result.tasks[0].run_id,'two');assert.equal(result.snapshot_stale,true);
});
test('expanded record survives refresh and filter changes',async()=>{
 const p=await page(),detail=p.$('history').children[0];detail.open=true;detail.ontoggle();
 p.$('history-filter').value='success';p.$('history-filter').onchange();
 p.$('history-filter').value='all';p.$('history-filter').onchange();
 assert.equal(p.$('history').children[0].open,true);
});
test('persistent storage warning is retained in exports',()=>{
 const result=exportData({scope:'local',limit:20,storage_warning:'disk full'},tasks,'all','',false);
 assert.equal(result.storage_warning,'disk full');assert.equal(result.scope,'local');
 assert.equal(result.snapshot_stale,false);
});
test('summary uses all selected outcomes as denominator and keeps stop counts separate',()=>{
 const rows=[...tasks,{phase:'success',stop_confirmation:'pending'},
  {phase:'failed',stop_confirmation:'unconfirmed'},{phase:'__proto__'}];
 const before=JSON.stringify(rows),stats=summarize(rows);
 assert.deepEqual(stats.outcomes,{success:2,failed:2,stopped:1,other:1});
 assert.equal(stats.total,6);assert.equal(stats.completion_ratio,2/6);
 assert.deepEqual(stats.stop_confirmation,{unconfirmed:2,pending:1});
 assert.equal(JSON.stringify(rows),before);
});
test('durations exclude missing, nonnumeric, reversed and nonfinite times, retaining zero',()=>{
 const rows=[{started_at:0,finished_at:0},{started_at:1,finished_at:5},
  {started_at:10,finished_at:20},{started_at:0,finished_at:30},
  {},{started_at:null,finished_at:2},{started_at:'1',finished_at:2},
  {started_at:3,finished_at:2},{started_at:Infinity,finished_at:Infinity},
  {started_at:-Number.MAX_VALUE,finished_at:Number.MAX_VALUE}];
 assert.deepEqual(summarize(rows).duration_seconds,{count:4,excluded:6,median:7,max:30});
 assert.equal(summarize(rows.slice(0,3)).duration_seconds.median,4);
});
test('empty summary and unavailable durations are null rather than zero performance',()=>{
 assert.equal(summarize([]).completion_ratio,null);
 assert.deepEqual(summarize([]).duration_seconds,{count:0,excluded:0,median:null,max:null});
 assert.equal(summarize(tasks).duration_seconds.median,null);
});
test('failure grouping counts only failures, keeps literal titles and sorts ties deterministically',()=>{
 const rows=[{phase:'failed',failure:{title:' B '}},{phase:'failed',failure:{title:'B'}},
  {phase:'failed',failure:{title:'A'}},{phase:'failed',failure:{title:'__proto__'}},
  {phase:'failed',failure:{title:'<img onerror=alert(1)>'}},
  {phase:'failed',failure:{title:' '}},{phase:'stopped',failure:{title:'B'}}];
 assert.deepEqual(summarize(rows).failure_groups,[{title:'B',count:2},
  {title:'<img onerror=alert(1)>',count:1},{title:'A',count:1},
  {title:'__proto__',count:1},{title:'未分类失败',count:1}]);
});
test('overview follows search and filter, and exports exactly the displayed summary',async()=>{
 const p=await page();
 assert.match(p.$('history-summary').textContent,/33.3%（1\/3）/);
 assert.match(p.$('history-summary').textContent,/停止未确认 1/);
 p.$('history-filter').value='failed';p.$('history-filter').onchange();
 assert.match(p.$('history-summary').textContent,/0.0%（0\/1）/);
 assert.match(p.$('history-failures').textContent,/未分类失败 × 1/);
 p.$('export-history').click();
 const exported=await p.getExport();
 assert.deepEqual(exported.summary,summarize(exported.tasks));
 p.$('history-search').value='absent';p.$('history-search').oninput();
 assert.match(p.$('history-summary').textContent,/暂无统计/);
});
test('failed refresh preserves overview and recovery replaces it, rendering titles as text',async()=>{
 const p=await page(),original=p.$('history-summary').textContent;
 p.setFail(true);await vm.runInContext('refreshHistory()',p.context);
 assert.equal(p.$('history-summary').textContent,original);
 p.$('history').children[1].children[2].click();
 const single=await p.getExport();assert.equal(single.summary.total,1);
 assert.equal(single.summary.outcomes.stopped,1);assert.equal(single.snapshot_stale,true);
 p.setFail(false);p.setRows([{run_id:'new',phase:'failed',plan:[],events:[],
  started_at:1,finished_at:3,failure:{title:'<script>bad()</script>',detail:'error'}}]);
 await vm.runInContext('refreshHistory()',p.context);
 assert.doesNotMatch(p.$('history-state').textContent,/旧快照/);
 assert.match(p.$('history-summary').textContent,/0.0%（0\/1）/);
 assert.match(p.$('history-failures').textContent,/<script>bad\(\)<\/script> × 1/);
 assert.equal(p.$('history-failures').children.length,0);
 p.$('export-history').click();assert.equal((await p.getExport()).snapshot_stale,false);
});
