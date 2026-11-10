const {test}=require('node:test');const assert=require('node:assert/strict');const {JSDOM}=require('jsdom');const vm=require('node:vm');const fs=require('node:fs');const path=require('node:path');const root=path.join(__dirname,'..');
async function setup(){const dom=new JSDOM(fs.readFileSync(path.join(root,'index.html'),'utf8'),{url:'http://localhost/workspace',runScripts:'outside-only'});const w=dom.window,context=dom.getInternalVMContext();let catalog=[{doc_id:'a',filename:'alpha.pdf',strategy:'recursive',total_pages:3},{doc_id:'b',filename:'beta.pdf',strategy:'recursive',total_pages:5}];const requests=[];
w.HTMLDialogElement.prototype.showModal=function(){this.open=true};w.HTMLDialogElement.prototype.close=function(){this.open=false};
w.fetch=async(url,opts={})=>{let data;if(url==='/api/documents')data=catalog;else if(url==='/health')data={status:'healthy',vector_chunks:2,parent_chunks:0};else if(url.startsWith('/api/query/traces'))data=[];else if(url.startsWith('/api/query/trace/'))data={steps:[]};else if(url==='/api/query'){const body=JSON.parse(opts.body);requests.push(body);const names=body.filters.source?[body.filters.source]:['alpha.pdf','beta.pdf'];const empty=body.filters.page_min===999;data={answer:'**Answer text** [1]',retrieval_strategy:'hybrid_rerank',total_latency_ms:100,model_name:'test',trace_id:'test-trace',citations:empty?[]:names.map((name,i)=>({index:i+1,source:name,page:3,snippet:'Supporting passage'})),retrieved_chunks:empty?[]:names.map(name=>({source:name,page:3,score:.8,text:'Evidence'}))}}else throw Error('Unexpected '+url);return {ok:true,json:async()=>data}};
for(const file of ['vendor/marked.umd.js','vendor/purify.min.js','answer-view.js','app.js'])vm.runInContext(fs.readFileSync(path.join(root,file),'utf8'),context);
await new Promise(resolve=>setImmediate(resolve));const get=id=>w.document.getElementById(id);const choose=value=>{get('querysource').value=value;get('querysource').dispatchEvent(new w.Event('change'))};return {w,get,choose,requests,run:()=>vm.runInContext('runQuery()',context),exec:code=>vm.runInContext(code,context),setCatalog:value=>catalog=value,close:()=>w.close()}}

function mockUpload(a){
  const requests=[];
  a.w.XMLHttpRequest=class{
    constructor(){this.upload={};requests.push(this)}
    open(method,url){this.method=method;this.url=url}
    send(data){this.data=data}
    finish(status=201,result={filename:'alpha.txt',chunk_count:3,total_tokens:50,total_pages:1}){this.status=status;this.responseText=JSON.stringify(result);this.onload()}
  };
  return requests;
}
test('single bar uses transfer events then indeterminate processing, survives queue disappearing',async()=>{
 const a=await setup();try{
  const requests=mockUpload(a);
  a.w.files=[new a.w.File(['Document evidence'],'alpha.txt',{type:'text/plain'})];
  a.exec('addFiles(window.files)');const run=a.exec('indexFiles()');
  assert.match(a.get('uploadprogress').textContent,/Uploading document/);
  assert.equal(a.get('indexfiles').disabled,true);
  assert.equal(requests[0].url,'/api/documents/upload');
  assert.equal(requests[0].data.get('strategy'),'recursive');
  requests[0].upload.onprogress({lengthComputable:true,loaded:50,total:100});
  assert.equal(a.get('uploadbar').getAttribute('aria-valuenow'),'50');
  requests[0].upload.onload();
  assert.match(a.get('uploadprogress').textContent,/Chunking and indexing/);
  assert.equal(a.get('uploadbar').hasAttribute('aria-valuenow'),false);
  // Double submit must not create a second upload.
  await a.exec('indexFiles()');assert.equal(requests.length,1);
  requests[0].finish();await run;
  assert.equal(a.get('pendingupload').classList.contains('hidden'),true);
  assert.equal(a.get('uploadstatus').hidden,false);
  assert.match(a.get('uploadprogress').textContent,/Document ready/);
  assert.equal(a.get('uploadbar').getAttribute('aria-valuenow'),'100');
  assert.equal(a.get('indexfiles').disabled,false);
 }finally{a.close()}
});
test('multiple documents run sequentially; errors retain failed files and allow retry',async()=>{
 const a=await setup();try{
  const requests=mockUpload(a);
  a.w.files=[new a.w.File(['one'],'one.txt'),new a.w.File(['two'],'two.txt')];
  a.exec('addFiles(window.files)');const run=a.exec('indexFiles()');
  assert.match(a.get('uploadprogress').textContent,/1 of 2/);
  requests[0].finish();await new Promise(setImmediate);
  assert.equal(requests.length,2);
  assert.match(a.get('uploadprogress').textContent,/2 of 2/);
  requests[1].finish(400,{detail:'No text found'});await run;
  assert.match(a.get('uploadprogress').textContent,/No text found/);
  assert.match(a.get('pendingfiles').textContent,/two.txt/);
  assert.doesNotMatch(a.get('pendingfiles').textContent,/one.txt/);
  assert.equal(a.get('uploadstatus').dataset.state,'error');
  assert.equal(a.get('indexfiles').disabled,false);
  const retry=a.exec('indexFiles()');requests[2].finish();await retry;
  assert.equal(a.get('uploadstatus').dataset.state,'success');
 }finally{a.close()}
});
test('network and timeout errors restore controls and do not claim success',async()=>{
 for(const event of ['onerror','ontimeout','onabort']){
  const a=await setup();try{
   const requests=mockUpload(a);
   a.w.files=[new a.w.File(['one'],'one.txt')];
   a.exec('addFiles(window.files)');const run=a.exec('indexFiles()');
   requests[0][event]();await run;
   assert.equal(a.get('uploadstatus').dataset.state,'error');
   assert.equal(a.get('indexfiles').disabled,false);
   assert.match(a.get('pendingfiles').textContent,/one.txt/);
  }finally{a.close()}
 }
});
