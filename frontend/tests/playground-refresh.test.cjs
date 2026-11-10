const {test}=require('node:test');const assert=require('node:assert/strict');const {JSDOM}=require('jsdom');const vm=require('node:vm');const fs=require('node:fs');const path=require('node:path');const root=path.join(__dirname,'..');
async function setup(){const dom=new JSDOM(fs.readFileSync(path.join(root,'index.html'),'utf8'),{url:'http://localhost/workspace',runScripts:'outside-only'});const w=dom.window,context=dom.getInternalVMContext();let catalog=[{doc_id:'a',filename:'alpha.pdf',strategy:'recursive',total_pages:3},{doc_id:'b',filename:'beta.pdf',strategy:'recursive',total_pages:5}];const requests=[];
w.HTMLDialogElement.prototype.showModal=function(){this.open=true};w.HTMLDialogElement.prototype.close=function(){this.open=false};
w.fetch=async(url,opts={})=>{let data;if(url==='/api/documents')data=catalog;else if(url==='/health')data={status:'healthy',vector_chunks:2,parent_chunks:0};else if(url.startsWith('/api/query/traces'))data=[];else if(url.startsWith('/api/query/trace/'))data={steps:[]};else if(url==='/api/query'){const body=JSON.parse(opts.body);requests.push(body);const names=body.filters.source?[body.filters.source]:['alpha.pdf','beta.pdf'];const empty=body.filters.page_min===999;data={answer:'**Answer text** [1]',retrieval_strategy:'hybrid_rerank',total_latency_ms:100,model_name:'test',trace_id:'test-trace',citations:empty?[]:names.map((name,i)=>({index:i+1,source:name,page:3,snippet:'Supporting passage'})),retrieved_chunks:empty?[]:names.map(name=>({source:name,page:3,score:.8,text:'Evidence'}))}}else throw Error('Unexpected '+url);return {ok:true,json:async()=>data}};
for(const file of ['vendor/marked.umd.js','vendor/purify.min.js','answer-view.js','app.js'])vm.runInContext(fs.readFileSync(path.join(root,file),'utf8'),context);
await new Promise(resolve=>setImmediate(resolve));const get=id=>w.document.getElementById(id);const choose=value=>{get('querysource').value=value;get('querysource').dispatchEvent(new w.Event('change'))};return {w,get,choose,requests,run:()=>vm.runInContext('runQuery()',context),exec:code=>vm.runInContext(code,context),setCatalog:value=>catalog=value,close:()=>w.close()}}

test('document scope has no manual refresh button or status',async()=>{
 const a=await setup();try{
  assert.equal(a.get('refreshscope'),null);
  assert.equal(a.get('scoperefreshstatus'),null);
  assert.ok(a.get('querysource'));
 }finally{a.close()}
});
test('document catalog still updates when health request fails',async()=>{
 const a=await setup();try{
  a.setCatalog([{doc_id:'c',filename:'new.pdf'}]);
  const original=a.w.fetch;a.w.fetch=(url,opts)=>url==='/health'?Promise.reject(Error('Unavailable')):original(url,opts);
  await a.exec('refresh()');
  assert.match(a.get('querysource').textContent,/new.pdf/);
  assert.equal(a.get('connection').textContent,'Health check unavailable');
 }finally{a.close()}
});
test('trace refresh reports completion and restores button on failure',async()=>{
 const a=await setup();try{
  await a.exec('refreshTraces()');
  assert.match(a.get('toast').textContent,/Recent explorations refreshed/);
  a.w.fetch=async()=>{throw Error('Offline')};
  await a.exec('refreshTraces()');
  assert.match(a.get('toast').textContent,/Offline/);
  assert.equal(a.get('refreshtraces').disabled,false);
 }finally{a.close()}
});
