const {test}=require('node:test');const assert=require('node:assert/strict');const {JSDOM}=require('jsdom');const vm=require('node:vm');const fs=require('node:fs');const path=require('node:path');const root=path.join(__dirname,'..');
async function setup(){const dom=new JSDOM(fs.readFileSync(path.join(root,'index.html'),'utf8'),{url:'http://localhost/workspace',runScripts:'outside-only'});const w=dom.window,context=dom.getInternalVMContext();let catalog=[{doc_id:'a',filename:'alpha.pdf',strategy:'recursive',total_pages:3},{doc_id:'b',filename:'beta.pdf',strategy:'recursive',total_pages:5}];const requests=[];
w.HTMLDialogElement.prototype.showModal=function(){this.open=true};w.HTMLDialogElement.prototype.close=function(){this.open=false};
w.fetch=async(url,opts={})=>{let data;if(url==='/api/documents')data=catalog;else if(url==='/health')data={status:'healthy',vector_chunks:2,parent_chunks:0};else if(url.startsWith('/api/query/traces'))data=[];else if(url.startsWith('/api/query/trace/'))data={steps:[]};else if(url==='/api/query'){const body=JSON.parse(opts.body);requests.push(body);const names=body.filters.source?[body.filters.source]:['alpha.pdf','beta.pdf'];const empty=body.filters.page_min===999;data={answer:'**Answer text** [1]',retrieval_strategy:'hybrid_rerank',total_latency_ms:100,model_name:'test',trace_id:'test-trace',citations:empty?[]:names.map((name,i)=>({index:i+1,source:name,page:3,snippet:'Supporting passage'})),retrieved_chunks:empty?[]:names.map(name=>({source:name,page:3,score:.8,text:'Evidence'}))}}else throw Error('Unexpected '+url);return {ok:true,json:async()=>data}};
for(const file of ['vendor/marked.umd.js','vendor/purify.min.js','answer-view.js','app.js'])vm.runInContext(fs.readFileSync(path.join(root,file),'utf8'),context);
await new Promise(resolve=>setImmediate(resolve));const get=id=>w.document.getElementById(id);const choose=value=>{get('querysource').value=value;get('querysource').dispatchEvent(new w.Event('change'))};return {w,get,choose,requests,run:()=>vm.runInContext('runQuery()',context),exec:code=>vm.runInContext(code,context),setCatalog:value=>catalog=value,close:()=>w.close()}}

test('header greeting follows local time and displays the signed-in name safely',async()=>{
 const a=await setup();try{
  const cases=[[0,'Good evening'],[4,'Good evening'],[5,'Good morning'],[10,'Good morning'],[12,'Good afternoon'],[16,'Good afternoon'],[17,'Good evening'],[23,'Good evening']];
  a.exec("greetingName='XYZ'");
  for(const [hour,label] of cases){
   a.exec('updateGreeting(new Date(2026,10,6,'+hour+',0))');
   assert.equal(a.get('userGreeting').textContent,label+', XYZ');
  }
  a.exec("greetingName='<img src=x onerror=alert(1)>';updateGreeting(new Date(2026,10,6,10))");
  assert.equal(a.get('userGreeting').querySelector('img'),null);
  assert.match(a.get('userGreeting').textContent,/<img/);
  assert.equal(a.get('accountName'),null);
  a.exec("greetingName='';updateGreeting(new Date(2026,10,6,10))");
  assert.equal(a.get('userGreeting').textContent,'Good morning');
 }finally{a.close()}
});
test('greeting uses the existing authenticated profile',async()=>{
 const a=await setup();try{
  const original=a.w.fetch;
  a.w.fetch=(url,opts)=>url==='/api/auth/me'?Promise.resolve({ok:true,json:async()=>({name:'Example User'})}):original(url,opts);
  await a.exec("api('/api/auth/me').then(u=>{greetingName=u.name;updateGreeting()})");
  assert.match(a.get('userGreeting').textContent,/, Example User$/);
 }finally{a.close()}
});
