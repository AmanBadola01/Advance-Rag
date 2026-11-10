/* Readable, sanitized answers with scoped citations and independent comparison tabs. */
const AnswerView=(()=>{
  const responses=new Map();let nextId=0;
  function format(text,citations){
    const root=document.createElement('div');
    root.innerHTML=DOMPurify.sanitize(marked.parse(String(text||''),{gfm:true,breaks:false}),{FORBID_TAGS:['img','style','svg','math','form','input','button'],FORBID_ATTR:['style']});
    root.querySelectorAll('a').forEach(a=>{a.target='_blank';a.rel='noopener noreferrer'});
    const valid=new Set(citations.map(c=>Number(c.index)));
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);const nodes=[];
    while(walker.nextNode())if(!walker.currentNode.parentElement.closest('pre,code,a'))nodes.push(walker.currentNode);
    for(const node of nodes){
      const text=node.textContent,re=/\[(\d+(?:\s*,\s*\d+)*)\]/g;let match,last=0,changed=false;const fragment=document.createDocumentFragment();
      while((match=re.exec(text))){
        const indexes=match[1].split(',').map(Number);if(!indexes.every(i=>valid.has(i)))continue;
        fragment.append(document.createTextNode(text.slice(last,match.index)));
        for(const i of indexes){const button=document.createElement('button');button.className='inline-cite';button.type='button';button.dataset.answerAction='citation';button.dataset.citation=String(i);button.textContent=String(i);button.setAttribute('aria-label',`View source ${i}`);fragment.append(button)}
        last=re.lastIndex;changed=true;
      }
      if(changed){fragment.append(document.createTextNode(text.slice(last)));node.replaceWith(fragment)}
    }
    root.querySelectorAll('table').forEach(table=>{const wrap=document.createElement('div');wrap.className='answer-table';table.replaceWith(wrap);wrap.append(table)});
    return root.innerHTML;
  }
  function render(r){
    const id=++nextId;responses.set(String(id),r);
    // Keep recent rendered answers for copy/trace actions without unbounded query history.
    if(responses.size>30)responses.delete(responses.keys().next().value);
    const sourceGroups=new Map();for(const c of r.citations){if(!sourceGroups.has(c.source))sourceGroups.set(c.source,{index:c.index,pages:new Set()});if(c.page!=null)sourceGroups.get(c.source).pages.add(c.page)}
    const sources=sourceGroups.size,noEvidence=r.retrieved_chunks.length===0;
    const scope=r.query_scope||'All documents';
    return `<article class="response-view" data-response="${id}">
      <div class="response-heading"><div><div class="eyebrow">${r.fallback?'Extractive response':'Response'}</div><h3>Your answer</h3></div><button class="answer-copy" data-answer-action="copy">Copy answer</button></div>
      <div class="response-meta"><span>${esc(strategies[r.retrieval_strategy]||r.retrieval_strategy)}</span><span>${(r.total_latency_ms/1000).toFixed(2)}s</span><span title="Generation model">${esc(r.model_name)}</span><span>${sources} source${sources===1?'':'s'} · ${r.citations.length} citation${r.citations.length===1?'':'s'}</span></div>
      ${Object.hasOwn(r,'query_scope')?`<div class="answer-scope"><span>Answer scope</span><strong>${esc(scope)}</strong></div>`:''}
      ${sourceGroups.size?`<div class="sources-used"><span class="eyebrow">Sources used</span>${Array.from(sourceGroups,([name,data])=>`<button class="source-chip" data-answer-action="citation" data-citation="${Number(data.index)}"><span>${esc(name)}</span><small>${data.pages.size?'Pages '+Array.from(data.pages).sort((a,b)=>a-b).map(esc).join(', '):'View passages'}</small></button>`).join('')}</div>`:''}
      ${r.fallback?'<div class="response-notice" role="status">Showing passages from your documents. The language model was unavailable; this answer uses local extraction.</div>':''}
      ${r.traceUnavailable?'<div class="response-notice">Execution details are temporarily unavailable.</div>':''}
      <div class="response-tabs" role="tablist" aria-label="Response views">
        ${[['answer','Answer'],['sources',`Sources · ${sources}`],['evidence',`Evidence · ${r.retrieved_chunks.length}`]].map(([key,label],i)=>`<button id="response-${id}-${key}-tab" role="tab" aria-selected="${i===0}" aria-controls="response-${id}-${key}" tabindex="${i===0?'0':'-1'}" data-answer-action="tab" data-view="${key}">${label}</button>`).join('')}
      </div>
      <section class="response-panel" id="response-${id}-answer" role="tabpanel" aria-labelledby="response-${id}-answer-tab" tabindex="0"><div class="answer-prose">${noEvidence?`<div class="no-evidence"><h3>${r.query_scope?'No matching evidence in this document':'No matching evidence found'}</h3><p>${r.query_scope?'No passages were returned from <strong>'+esc(r.query_scope)+'</strong> with these settings.':'No passages were returned from your indexed documents with these settings.'} Try another question, adjust metadata filters, or change the document scope.</p></div>`:format(r.answer,r.citations)}</div>${r.citations.length?'<p class="answer-tip">Select a numbered reference to read the supporting passage.</p>':''}</section>
      <section class="response-panel" id="response-${id}-sources" role="tabpanel" aria-labelledby="response-${id}-sources-tab" tabindex="0" hidden><div class="source-list">${r.citations.map(c=>`<article class="source-card" data-source-index="${Number(c.index)}" tabindex="-1"><div class="source-heading"><span class="source-number">${Number(c.index)}</span><div><h4>${esc(c.source)}</h4><span class="sub">Page ${esc(c.page??'—')}${c.section?' · '+esc(c.section):''}</span></div></div><p>${esc(c.snippet)}</p></article>`).join('')||'<p class="answer-empty">No sources were returned for this query.</p>'}</div></section>
      <section class="response-panel" id="response-${id}-evidence" role="tabpanel" aria-labelledby="response-${id}-evidence-tab" tabindex="0" hidden><p class="answer-tip">Passages returned by retrieval, before answer generation. Scores depend on the selected strategy.</p>${r.retrieved_chunks.map((c,i)=>`<details class="evidence-card"><summary><span class="source-number">${esc(c.rank??i+1)}</span><span>${esc(c.source)}<small>Page ${esc(c.page??'—')} · ${esc(c.strategy)}</small></span><span class="evidence-score">Score ${Number(c.score).toFixed(3)}</span></summary><p>${esc(c.text)}</p><details><summary>Technical metadata</summary><pre>${esc(JSON.stringify(Object.fromEntries(Object.entries(c).filter(([k])=>k!=='text')),null,2))}</pre></details></details>`).join('')||'<p class="answer-empty">No matching evidence was found. Try another question or adjust the filters.</p>'}</section>
      <div class="response-footer"><span>Answers can be checked against the linked passages.</span><button data-answer-action="trace">Execution details ↗</button></div>
    </article>`;
  }
  function selectTab(root,key){
    root.querySelectorAll('[role="tab"]').forEach(b=>{const selected=b.dataset.view===key;b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1});
    root.querySelectorAll('[role="tabpanel"]').forEach(panel=>panel.hidden=!panel.id.endsWith('-'+key));
  }
  async function copy(root,button){
    const body=root.querySelector('.answer-prose');const holder=body.cloneNode(true);holder.querySelectorAll('.inline-cite').forEach(b=>b.replaceWith(document.createTextNode('['+b.textContent+']')));
    // Rendered body text only: no source list, hidden metadata or controls.
    holder.style.cssText='position:fixed;left:-9999px;top:0';document.body.append(holder);const text=holder.innerText;holder.remove();
    try{
      if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(text);
      else{const input=document.createElement('textarea');input.value=text;input.style.cssText='position:fixed;left:-9999px;top:0';document.body.append(input);input.select();const ok=document.execCommand('copy');input.remove();button.focus();if(!ok)throw Error('Clipboard unavailable')}
      button.textContent='Copied';setTimeout(()=>{button.textContent='Copy answer'},1800);
    }catch{toast('Clipboard unavailable. Select the answer text to copy.')}
  }
  document.addEventListener('click',e=>{
    const button=e.target.closest('[data-answer-action]');if(!button)return;const root=button.closest('.response-view');if(!root)return;
    const r=responses.get(root.dataset.response);const action=button.dataset.answerAction;
    if(action==='tab')selectTab(root,button.dataset.view);
    if(action==='copy')copy(root,button);
    if(action==='trace'&&r)showTrace(r.trace_id);
    if(action==='citation'){
      selectTab(root,'sources');const source=root.querySelector(`[data-source-index="${Number(button.dataset.citation)}"]`);
      if(source){source.focus({preventScroll:true});source.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'nearest'})}
    }
  });
  document.addEventListener('keydown',e=>{
    const tab=e.target.closest('.response-tabs [role="tab"]');if(!tab||!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
    e.preventDefault();const root=tab.closest('.response-view'),tabs=Array.from(root.querySelectorAll('[role="tab"]'));let i=tabs.indexOf(tab);
    i=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
    selectTab(root,tabs[i].dataset.view);tabs[i].focus();
  });
  return {render,format};
})();
