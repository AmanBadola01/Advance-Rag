const {test}=require('node:test');const assert=require('node:assert/strict');const {JSDOM}=require('jsdom');const vm=require('node:vm');const fs=require('node:fs');const path=require('node:path');const root=path.join(__dirname,'..');
async function setup(){const dom=new JSDOM(fs.readFileSync(path.join(root,'index.html'),'utf8'),{url:'http://localhost/workspace',runScripts:'outside-only'});const w=dom.window,context=dom.getInternalVMContext();let catalog=[{doc_id:'a',filename:'alpha.pdf',strategy:'recursive',total_pages:3},{doc_id:'b',filename:'beta.pdf',strategy:'recursive',total_pages:5}];const requests=[];
w.HTMLDialogElement.prototype.showModal=function(){this.open=true};w.HTMLDialogElement.prototype.close=function(){this.open=false};
w.fetch=async(url,opts={})=>{let data;if(url==='/api/documents')data=catalog;else if(url==='/health')data={status:'healthy',vector_chunks:2,parent_chunks:0};else if(url.startsWith('/api/query/traces'))data=[];else if(url.startsWith('/api/query/trace/'))data={steps:[]};else if(url==='/api/query'){const body=JSON.parse(opts.body);requests.push(body);const names=body.filters.source?[body.filters.source]:['alpha.pdf','beta.pdf'];const empty=body.filters.page_min===999;data={answer:'**Answer text** [1]',retrieval_strategy:'hybrid_rerank',total_latency_ms:100,model_name:'test',trace_id:'test-trace',citations:empty?[]:names.map((name,i)=>({index:i+1,source:name,page:3,snippet:'Supporting passage'})),retrieved_chunks:empty?[]:names.map(name=>({source:name,page:3,score:.8,text:'Evidence'}))}}else throw Error('Unexpected '+url);return {ok:true,json:async()=>data}};
for(const file of ['vendor/marked.umd.js','vendor/purify.min.js','answer-view.js','app.js'])vm.runInContext(fs.readFileSync(path.join(root,file),'utf8'),context);
await new Promise(resolve=>setImmediate(resolve));const get=id=>w.document.getElementById(id);const choose=value=>{get('querysource').value=value;get('querysource').dispatchEvent(new w.Event('change'))};return {w,get,choose,requests,run:()=>vm.runInContext('runQuery()',context),exec:code=>vm.runInContext(code,context),setCatalog:value=>catalog=value,close:()=>w.close()}}

test('invalid top K is explained beside the field and never sent to the API',async()=>{
 const a=await setup();try{
  for(const value of ['0','500','-1','1.5','','101']){
   a.get('topk').value=value;await a.run();
   assert.equal(a.requests.length,0);
   assert.match(a.get('topkerror').textContent,/whole number between 1 and 100/);
   assert.equal(a.get('topk').getAttribute('aria-invalid'),'true');
   assert.equal(a.w.document.activeElement,a.get('topk'));
  }
  for(const value of ['1','100']){
   a.get('topk').value=value;a.get('topk').dispatchEvent(new a.w.Event('input'));
   assert.equal(a.get('topkerror').textContent,'');
   await a.run();
   assert.equal(a.requests.at(-1).top_k,Number(value));
  }
 }finally{a.close()}
});
test('server validation responses appear as readable text, not JSON',async()=>{
 const a=await setup();try{
  const original=a.w.fetch;
  a.w.fetch=(url,opts)=>url==='/api/query'?Promise.resolve({ok:false,status:422,json:async()=>({detail:[{type:'less_than_equal',loc:['body','top_k'],msg:'Input should be less than or equal to 100',input:500,ctx:{le:100}}]})}):original(url,opts);
  await a.run();
  assert.match(a.get('answer').textContent,/Final top-K chunks must be a whole number between 1 and 100/);
  assert.doesNotMatch(a.get('answer').textContent,/less_than_equal|ctx|loc|Error details/);
  assert.equal(a.get('pipestatus').textContent,'Query failed');
 }finally{a.close()}
});
