'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const strategies={vector:'Basic vector',bm25:'Keyword · BM25',hybrid:'Hybrid search',hybrid_rerank:'Hybrid + re-ranking',parent_child:'Parent-child'};
const chunkers={recursive:'Recursive',section:'Section-based',parent_child:'Parent-child'};
let pendingFiles=[],docs=[],chunks=[],traces=[],queryFilters={},health=null,busy=false,catalogReady=false,lastAnswerScope=null;
const pages={help:['Help & guide','Understand the workflow, compare the techniques, and see what powers your workspace.'],playground:['A little curiosity. Better answers.','Explore your knowledge, with every source in sight.'],documents:['Your knowledge, connected.','A home for every document behind your answers.'],compare:['Better retrieval starts with comparison.','Compare real answers, evidence and latency.'],chunks:['The details make the difference.','Inspect indexed passages and their metadata.']};
function options(map,all){return (all!==undefined?`<option value="">${esc(all)}</option>`:'')+Object.entries(map).map(([v,n])=>`<option value="${esc(v)}">${esc(n)}</option>`).join('')}
function toast(t){$('toast').textContent=t;$('toast').classList.remove('hidden');clearTimeout(window.tt);window.tt=setTimeout(()=>$('toast').classList.add('hidden'),6500)}
function friendlyError(detail){
  if(typeof detail==='string')return detail;
  if(Array.isArray(detail)){
    return [...new Set(detail.map(issue=>{
      const field=(issue.loc||[]).filter(x=>x!=='body').pop();
      if(field==='top_k')return 'Final top-K chunks must be a whole number between 1 and 100.';
      const labels={query:'Question',semantic_weight:'Semantic search weight',bm25_weight:'Keyword search weight',retrieval_strategy:'Retrieval strategy',chunk_size:'Chunk size',chunk_overlap:'Chunk overlap',parent_chunk_size:'Parent chunk size',child_chunk_size:'Child chunk size'};
      const label=labels[field]||String(field||'Input').replace(/_/g,' ');
      return label+': '+(issue.msg||'Please check this value.')+'.';
    }))].join(' ');
  }
  return detail?.message||'Something went wrong. Please try again.';
}
function validateTopK(focus=false){
  const input=$('topk'),value=Number(input.value);
  const valid=input.value.trim()!==''&&Number.isInteger(value)&&value>=1&&value<=100;
  $('topkerror').textContent=valid?'':'Enter a whole number between 1 and 100 for Final top-K chunks.';
  input.setAttribute('aria-invalid',String(!valid));
  if(!valid&&focus)input.focus();
  return valid;
}
async function api(path,body,method){const r=await fetch(path,{method:method||(body?'POST':'GET'),cache:'no-store',...(body?body instanceof FormData?{body}:{headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})});if(r.status===401){location.replace('/login');throw Error('Your session has expired. Please sign in.')}const data=await r.json();if(!r.ok){const d=data.detail??data;const e=new Error(friendlyError(d));e.traceId=d.trace_id;throw e}return data}
function navigate(p){document.body.classList.toggle('help-mode',p==='help');document.querySelectorAll('.page').forEach(x=>x.classList.toggle('hidden',x.id!==p));document.querySelectorAll('[data-page]').forEach(x=>x.classList.toggle('active',x.dataset.page===p));$('pagetitle').textContent=pages[p][0];$('subtitle').textContent=pages[p][1];if(p==='help')window.scrollTo({top:0,behavior:'instant'});if(p==='chunks')loadChunks().catch(report)}
function report(e){toast(e.message)}
function modal(html){$('modal').innerHTML=html+'<div style="margin-top:22px;text-align:right"><button onclick="document.getElementById(\'modal\').close()">Close</button></div>';if(!$('modal').open)$('modal').showModal()}
function suggest(b){$('query').value=b.textContent.replace(' ↗','');$('query').focus()}
function syncTheme(){const dark=document.documentElement.dataset.theme==='dark';$('themeLabel').textContent=dark?'Light mode':'Dark mode';$('themeToggle').setAttribute('aria-pressed',String(dark));$('themeToggle').setAttribute('aria-label',dark?'Switch to light theme':'Switch to dark theme');document.querySelector('.theme-icon').textContent=dark?'☀':'☾'}
function toggleTheme(){document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('lattice-theme',document.documentElement.dataset.theme)}catch{}syncTheme()}
async function refresh({documentsOnly=false}={}){
  const d=await api('/api/documents');
  docs=d;
  $('totaldocs').textContent=d.length;$('doccount').textContent=d.length;document.querySelector('.sectionhead .badge.green').textContent=d.length+' sources connected';$('docrows').innerHTML=d.map((x,i)=>`<tr><td>${esc(x.filename)}</td><td>${x.total_pages}</td><td>${esc(chunkers[x.strategy]||x.strategy)}</td><td>Indexed</td><td><button onclick="removeDoc(${i})">Remove</button></td></tr>`).join('')||'<tr><td colspan="5">Upload your first document.</td></tr>';for(const id of ['abdoc','chunksource']){const prev=$(id).value;$(id).innerHTML=`<option value="">${id==='abdoc'?'Entire corpus':'All sources'}</option>`+d.map(x=>`<option value="${esc(id==='abdoc'?x.doc_id:x.filename)}">${esc(x.filename)}</option>`).join('');$(id).value=prev}catalogReady=true;refreshDocumentScope();syncAB()
  if(documentsOnly)return d;
  try{
    const response=await fetch('/health',{cache:'no-store'});
    health=await response.json();
    $('connection').textContent=health.status==='healthy'?'Backend connected':'Backend not ready';
    $('chunkcount').textContent=health.vector_chunks??'—';
    $('metricchunks').textContent=health.vector_chunks??'—';
    $('parents').textContent=health.parent_chunks??'—';
  }catch(e){$('connection').textContent='Health check unavailable'}
  return d;
}
async function showHealth(){await refresh().catch(report);modal('<h2>System health</h2><pre>'+esc(JSON.stringify(health,null,2))+'</pre><a href="/docs" target="_blank">Explore API documentation</a>')}
function openUpload(){navigate('documents');$('fileinput').click()}
function renderPending(){
  $('pendingupload').classList.toggle('hidden',!pendingFiles.length);
  $('pendingfiles').innerHTML=pendingFiles.map(f=>`<li>${esc(f.name)} · ${(f.size/1024).toFixed(1)} KB</li>`).join('');
}
function addFiles(files){
  if(busy)return toast('Wait for the current operation.');
  const added=Array.from(files).filter(f=>/\.(pdf|txt|docx|md|markdown|csv|json)$/i.test(f.name));
  if(!added.length)return toast('Choose PDF, TXT, DOCX, Markdown, CSV or JSON files.');
  pendingFiles.push(...added);$('fileinput').value='';setUploadProgress('idle','Choose a chunking technique to continue.');
  renderPending();navigate('documents');
}
function cancelUpload(){if(busy)return;pendingFiles=[];$('fileinput').value='';setUploadProgress('idle','');renderPending()}
function setUploadProgress(state,message,percent){
  const panel=$('uploadstatus'),bar=$('uploadbar');
  panel.hidden=!message;panel.dataset.state=state;
  panel.setAttribute('aria-busy',String(state==='uploading'||state==='processing'));
  $('uploadprogress').textContent=message;
  $('uploadtrack').hidden=state==='idle'||state==='error';
  bar.setAttribute('aria-valuetext',message);
  if(state==='processing'){bar.removeAttribute('aria-valuenow');bar.style.width='35%'}
  else{const value=state==='success'?100:(percent??0);bar.setAttribute('aria-valuenow',String(value));bar.style.width=value+'%'}
}
function uploadDocument(data,onProgress){
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();
    xhr.open('POST','/api/documents/upload');
    xhr.timeout=15*60*1000;
    xhr.upload.onprogress=e=>onProgress('uploading',e.lengthComputable?Math.min(100,Math.round(e.loaded/e.total*100)):null);
    xhr.upload.onload=()=>onProgress('processing');
    xhr.onload=()=>{
      if(xhr.status===401){location.replace('/login');reject(Error('Your session has expired. Please sign in.'));return}
      let result;try{result=JSON.parse(xhr.responseText)}catch{reject(Error('Unexpected server response. Refresh documents before retrying.'));return}
      if(xhr.status>=200&&xhr.status<300)resolve(result);
      else{const detail=result.detail??result;reject(Error(friendlyError(detail)))}
    };
    xhr.onerror=()=>reject(Error('Connection lost. Check the document list before retrying.'));
    xhr.ontimeout=()=>reject(Error('Processing is taking too long. It may still finish on the server; refresh documents before retrying.'));
    xhr.onabort=()=>reject(Error('Upload interrupted. Check the document list before retrying.'));
    xhr.send(data);
  });
}
async function indexFiles(){
  if(busy||!pendingFiles.length)return;
  busy=true;const strategy=$('uploadstrategy').value,total=pendingFiles.length;let completed=0;
  for(const id of ['browsefiles','indexfiles','cancelupload','uploadstrategy'])$(id).disabled=true;
  $('drop').setAttribute('aria-busy','true');
  try{
    while(pendingFiles.length){
      const f=pendingFiles[0],suffix=f.name+' · '+(completed+1)+' of '+total;
      setUploadProgress('uploading','Uploading document… '+suffix,0);
      const data=new FormData();data.append('file',f,f.name);data.append('strategy',strategy);
      const r=await uploadDocument(data,(state,percent)=>setUploadProgress(state,(state==='processing'?'Chunking and indexing document… ':'Uploading document… ')+suffix+(state==='uploading'&&percent!==null?' · '+percent+'%':''),percent));
      pendingFiles.shift();completed++;renderPending();
      toast(`${r.filename}: ${r.chunk_count} chunks · ${r.total_tokens} tokens · ${r.total_pages} pages`);
    }
    setUploadProgress('success',total===1?'Document ready — searchable in Playground.':total+' documents ready — searchable in Playground.');
  }catch(e){setUploadProgress('error','Could not finish: '+e.message);report(e)}
  finally{
    busy=false;$('drop').setAttribute('aria-busy','false');
    for(const id of ['browsefiles','indexfiles','cancelupload','uploadstrategy'])$(id).disabled=false;
    await refresh().catch(report);
  }
}
async function removeDoc(i){const name=docs[i].filename;if(!confirm(`Delete all indexed uploads named "${name}" and their chunks?`))return;try{await api('/api/documents/'+encodeURIComponent(name),null,'DELETE');await refresh();await loadChunks();toast('Document removed')}catch(e){report(e)}}
function refreshDocumentScope(){
  const names=new Map();for(const d of docs)names.set(d.filename,(names.get(d.filename)||0)+1);
  const selected=queryFilters.source||'';
  $('querysource').innerHTML='<option value="">All documents</option>'+Array.from(names,([name,count])=>`<option value="${esc(name)}">${esc(name)}${count>1?' ('+count+' uploads)':''}</option>`).join('');
  if(selected&&!names.has(selected))$('querysource').insertAdjacentHTML('beforeend',`<option value="${esc(selected)}" disabled>${esc(selected)} — no longer indexed</option>`);
  $('querysource').value=selected;$('querysource').disabled=busy;syncScopeHint();
}
function syncScopeHint(){
  const source=queryFilters.source||'',count=docs.filter(d=>d.filename===source).length;
  $('scopehint').textContent=!catalogReady?'Loading your indexed documents…':source?(count?'Searching: '+source+(count>1?' · Includes '+count+' indexed uploads with this filename.':''):'This document is no longer indexed. Choose another document before running a query.'):(docs.length?'Searching: All documents':'No indexed documents yet. Add a document in Documents to get started.');
  $('scopechanged').hidden=lastAnswerScope===null||lastAnswerScope===source;
}
function changeDocumentScope(){
  if($('querysource').value)queryFilters.source=$('querysource').value;else delete queryFilters.source;
  syncScopeHint();
}
function filters(){modal(`<h2>Metadata filters</h2><p>Document scope is selected above the question. All additional conditions must match.</p><div class="two"><div class="field"><label for="fmin">Page from</label><input id="fmin" type="number" min="1"></div><div class="field"><label for="fmax">Page to</label><input id="fmax" type="number" min="1"></div></div><div class="field"><label for="fsection">Exact section</label><input id="fsection"></div><div class="field"><label for="fstrategy">Chunk strategy</label><select id="fstrategy">${options(chunkers,'All strategies')}</select></div><button class="primary" onclick="applyFilters()">Apply filters</button> <button onclick="resetFilters()">Reset filters</button>`);for(const [id,key] of [['fmin','page_min'],['fmax','page_max'],['fsection','section'],['fstrategy','strategy']])$(id).value=queryFilters[key]??''}
function applyFilters(){if(!$('fmin').checkValidity()||!$('fmax').checkValidity())return toast('Use positive page numbers.');if($('fmin').value&&$('fmax').value&&+$('fmin').value>+$('fmax').value)return toast('End page must follow start page.');const source=queryFilters.source;queryFilters=source?{source}:{};for(const [id,key] of [['fmin','page_min'],['fmax','page_max'],['fsection','section'],['fstrategy','strategy']])if($(id).value)queryFilters[key]=id==='fmin'||id==='fmax'?+$(id).value:$(id).value;$('filterbadge').textContent=Object.keys(queryFilters).some(k=>k!=='source')?'●':'';$('modal').close()}
function resetFilters(){const source=queryFilters.source;queryFilters=source?{source}:{};$('filterbadge').textContent='';$('modal').close()}
function syncRecommendation(){$('strategyrecommended').hidden=$('strategy').value!=='hybrid_rerank'}
function config(){return {retrieval_strategy:$('strategy').value,top_k:parseInt($('topk').value),semantic_weight:+$('weight').value/100,bm25_weight:1-+$('weight').value/100}}
function evidence(c){return `<div style="padding:15px 0;border-bottom:1px solid var(--line)"><span class="badge">${esc(c.source)} · p. ${esc(c.page)}</span><p>${esc(c.text)}</p><details><summary>Scores and metadata</summary><pre>${esc(JSON.stringify(Object.fromEntries(Object.entries(c).filter(([k])=>k!=='text')),null,2))}</pre></details></div>`}
async function generationStatus(r){
  try{const t=await api('/api/query/trace/'+encodeURIComponent(r.trace_id));r.trace=t;r.fallback=t.steps.some(s=>s.details.api_fallback)||r.model_name==='offline-synthesizer'}
  catch{r.traceUnavailable=true;r.fallback=r.model_name==='offline-synthesizer'}
  return r;
}
function responseHTML(r){return AnswerView.render(r)}
async function runQuery(){
  if(busy)return;if(!validateTopK(true))return;const query=$('query').value.trim();if(!query)return toast('Enter a question.');
  if(!catalogReady)return toast('Document list is not ready. Please reload the page and try again.');
  if(!docs.length)return toast('Add a document in Documents before asking a question.');
  const requestFilters={...queryFilters},requestScope=requestFilters.source||'';
  if(requestScope&&!docs.some(d=>d.filename===requestScope))return toast('The selected document is no longer indexed. Choose another document.');
  busy=true;$('querysource').disabled=true;$('run').disabled=true;$('run').textContent='Finding your answer…';$('pipestatus').textContent='Working';$('answer').setAttribute('aria-busy','true');
  $('flow').innerHTML='';$('pipeline-details').hidden=true;$('pipeline-details').open=false;
  $('answer').innerHTML='<div class="answer-state"><span class="eyebrow">Searching your knowledge</span><h3>Finding the context that matters.</h3><p>Retrieving relevant passages and preparing your answer. References will appear alongside the response.</p><div class="loading-lines" aria-hidden="true"><i></i><i></i><i></i></div></div>';
  try{
    const r=await api('/api/query',{query,...config(),filters:requestFilters});
    r.query_scope=requestScope;await generationStatus(r);$('answer').innerHTML=responseHTML(r);lastAnswerScope=requestScope;syncScopeHint();
    $('latency').textContent=(r.total_latency_ms/1000).toFixed(2)+'s';$('pipestatus').textContent=!r.retrieved_chunks.length?'No matching evidence':r.fallback?'Extraction fallback':'Complete';
    if(r.trace)renderPipeline(r.trace);await loadHistory();
  }catch(e){
    $('answer').innerHTML=`<div class="answer-state"><span class="eyebrow">Query could not complete</span><h3>We couldn’t prepare this answer.</h3><p role="alert">${esc(e.message)}</p><button onclick="runQuery()">Try again</button></div>`;
    $('pipestatus').textContent='Query failed';if(e.traceId)await loadTrace(e.traceId).catch(report);
  }finally{busy=false;$('querysource').disabled=false;$('run').disabled=false;$('run').textContent='Run query ↗';$('answer').setAttribute('aria-busy','false')}
}
function renderPipeline(t){$('pipeline-details').hidden=false;$('flow').innerHTML=t.steps.map(s=>`<div class="node"><b>${esc(s.name)}</b><small>${s.duration_ms.toFixed(1)}ms · ${esc(s.status)}${s.details.api_fallback?' · Extraction fallback':''}</small></div>`).join('')}
async function loadTrace(id){const t=await api('/api/query/trace/'+encodeURIComponent(id));renderPipeline(t);return t}

async function showTrace(id){try{const t=await api('/api/query/trace/'+encodeURIComponent(id));modal(`<h2>Execution trace</h2><p>${esc(t.query)}</p><p>${t.total_latency_ms}ms · ${t.total_tokens} estimated tokens</p>${t.steps.map(s=>`<details open><summary>${esc(s.name)} · ${s.duration_ms}ms · ${esc(s.status)}</summary><pre>${esc(JSON.stringify(s.details,null,2))}</pre></details>`).join('')}`)}catch(e){report(e)}}
async function refreshTraces(){
  const button=$('refreshtraces');if(button.disabled)return;
  button.disabled=true;button.textContent='Refreshing…';button.setAttribute('aria-busy','true');
  try{if(await loadHistory())toast('Recent explorations refreshed.')}
  finally{button.disabled=false;button.textContent='Refresh traces';button.setAttribute('aria-busy','false')}
}
async function loadHistory(){try{traces=await api('/api/query/traces?limit=100');$('history').innerHTML=traces.map((t,i)=>`<tr><td>${esc(t.query)}</td><td>${esc(strategies[t.retrieval_strategy]||t.retrieval_strategy)}</td><td><button onclick="showTrace('${esc(t.trace_id)}')">Inspect</button>${t.saved_answer?` <button onclick="openSavedAnswer(${i})">Answer</button>`:''}</td><td>${t.total_latency_ms.toFixed(1)}ms</td><td><button aria-label="Reuse query ${i+1}" onclick="reuseQuery(${i})">↗</button></td></tr>`).join('')||'<tr><td colspan="5">No saved research yet.</td></tr>';return true}catch(e){report(e);return false}}
function reuseQuery(i){navigate('playground');$('query').value=traces[i].query;$('strategy').value=traces[i].retrieval_strategy;syncRecommendation();$('query').focus()}
async function loadChunks(){try{const p=new URLSearchParams({limit:$('chunklimit').value});if($('chunksource').value)p.set('source',$('chunksource').value);if($('chunkstrategy').value)p.set('strategy',$('chunkstrategy').value);chunks=await api('/api/chunks?'+p);renderChunks(document.querySelector('[aria-label="Search chunks"]').value)}catch(e){report(e)}}
function renderChunks(q=''){$('chunkrows').innerHTML=chunks.filter(c=>JSON.stringify(c).toLowerCase().includes(q.toLowerCase())).map(evidence).join('')||'<p>No matching chunks in the loaded subset.</p>'}
function abFields(p){return `<div class="card" style="padding:15px"><h3>Pipeline ${p.toUpperCase()}</h3><div class="field"><label for="${p}name">Name</label><input id="${p}name" value="Pipeline ${p.toUpperCase()}"></div>${[['chunking','Chunking strategy','select'],['size','Chunk size','500'],['overlap','Chunk overlap','50'],['parent','Parent size','1000'],['child','Child size','200'],['top','Top K','5'],['weight','Semantic weight (0–1)','0.5']].map(([k,label,v])=>`<div class="field"><label for="${p+k}">${label}</label>${v==='select'?`<select id="${p+k}" data-chunk-setting>${options(chunkers)}</select>`:`<input id="${p+k}" type="number" value="${v}" min="0" step="${k==='weight'?'0.01':'1'}" ${['size','overlap','parent','child'].includes(k)?'data-chunk-setting':''}>`}</div>`).join('')}</div>`}
function syncAB(){document.querySelectorAll('[data-chunk-setting]').forEach(x=>x.disabled=!$('abdoc').value)}
function pipeline(p){const weight=+$(p+'weight').value;return {name:$(p+'name').value,retrieval_strategy:$(p==='a'?'strategyA':'strategyB').value,chunking_strategy:$(p+'chunking').value,chunk_size:+$(p+'size').value,chunk_overlap:+$(p+'overlap').value,parent_chunk_size:+$(p+'parent').value,child_chunk_size:+$(p+'child').value,top_k:+$(p+'top').value,semantic_weight:weight,bm25_weight:1-weight}}
async function compareRun(){if(busy)return;if(!$('comparequery').value.trim())return toast('Enter a comparison question.');busy=true;$('comparebtn').disabled=true;$('compareresult').textContent='Running both pipelines…';try{const r=await api('/api/query/ab-test',{query:$('comparequery').value,document_id:$('abdoc').value||null,pipeline_a:pipeline('a'),pipeline_b:pipeline('b')});await Promise.all([generationStatus(r.pipeline_a.response),generationStatus(r.pipeline_b.response)]);$('compareresult').innerHTML=[r.pipeline_a,r.pipeline_b].map(p=>`<div class="card comparecard"><h2>${esc(p.name)}</h2>${responseHTML(p.response)}<details><summary>Pipeline configuration</summary><pre>${esc(JSON.stringify(p.config,null,2))}</pre></details></div>`).join('')+`<p>${esc(r.faster_pipeline)} faster by ${r.latency_delta_ms}ms (execution only).</p>`;await loadHistory()}catch(e){$('compareresult').textContent=e.message}finally{busy=false;$('comparebtn').disabled=false}}
$('querysource').addEventListener('change',changeDocumentScope);
document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>navigate(b.dataset.page));$('strategy').innerHTML=options(strategies);$('strategy').value='hybrid_rerank';$('strategy').addEventListener('change',syncRecommendation);syncRecommendation();for(const id of ['strategyA','strategyB'])$(id).innerHTML=options(strategies);$('strategyB').value='hybrid_rerank';$('uploadstrategy').innerHTML=options(chunkers);$('chunkstrategy').innerHTML=options({...chunkers,parent_child_parent:'Parent context'},'All strategies');$('absettings').innerHTML=abFields('a')+abFields('b');syncAB();$('fileinput').onchange=e=>addFiles(e.target.files);
for(const event of ['dragover','drop'])$('drop').addEventListener(event,e=>{e.preventDefault();if(event==='drop'){navigate('documents');addFiles(e.dataTransfer.files);$('drop').classList.remove('drag')}else $('drop').classList.add('drag')});$('drop').ondragleave=()=>$('drop').classList.remove('drag');
document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='k'){e.preventDefault();navigate('playground');$('query').focus()}if((e.metaKey||e.ctrlKey)&&e.key==='Enter'&&!$('modal').open)runQuery()});syncTheme();refresh().catch(e=>{$('connection').textContent='Connection failed';report(e)});loadHistory();

async function signOut(){try{await api('/api/auth/session',null,'DELETE');location.replace('/login')}catch(e){report(e)}}
let greetingName='';
function updateGreeting(now=new Date()){
  const hour=now.getHours();
  const greeting=hour<5?'Good evening':hour<12?'Good morning':hour<17?'Good afternoon':'Good evening';
  $('userGreeting').textContent=greeting+(greetingName?', '+greetingName:'');
}
updateGreeting();
api('/api/auth/me').then(u=>{greetingName=typeof u.name==='string'?u.name.trim():'';updateGreeting()}).catch(report);
window.setInterval(()=>updateGreeting(),60000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)updateGreeting()});
window.addEventListener('pageshow',e=>{if(e.persisted)location.reload()});

function openSavedAnswer(i){const t=traces[i];if(!t.saved_answer)return;reuseQuery(i);const r=t.saved_answer;$('answer').innerHTML=responseHTML(r);lastAnswerScope=r.query_scope||'';syncScopeHint();}

// Guide links scroll within the guide without changing document scope or query state.
document.querySelectorAll('.guide-toc a').forEach(link=>link.addEventListener('click',event=>{event.preventDefault();const target=document.getElementById(link.hash.slice(1));if(target){target.setAttribute('tabindex','-1');target.focus({preventScroll:true});target.scrollIntoView({block:'start',behavior:'instant'})}}));

$('topk').addEventListener('input',()=>validateTopK());
