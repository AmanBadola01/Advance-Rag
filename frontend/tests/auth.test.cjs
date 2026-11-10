const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const path=require('node:path');const {JSDOM}=require('jsdom');
const base=path.join(__dirname,'..');
test('signup shows five live requirements, strength and password reveal',()=>{const dom=new JSDOM(fs.readFileSync(path.join(base,'auth.html'),'utf8'),{url:'http://localhost/signup',runScripts:'outside-only'});const w=dom.window;w.eval(fs.readFileSync(path.join(base,'auth.js'),'utf8'));const d=w.document;assert.equal(d.getElementById('requirements').hidden,false);const p=d.getElementById('password');p.value='a';p.dispatchEvent(new w.Event('input'));assert.equal(d.getElementById('strength').textContent,'Weak');p.value='LongEnough!99';p.dispatchEvent(new w.Event('input'));assert.equal(d.querySelectorAll('li.met').length,5);assert.equal(d.getElementById('strength').textContent,'Strong');d.getElementById('reveal').click();assert.equal(p.type,'text');dom.window.close()});
test('weak signup is blocked before sending credentials',()=>{const dom=new JSDOM(fs.readFileSync(path.join(base,'auth.html'),'utf8'),{url:'http://localhost/signup',runScripts:'outside-only'});const w=dom.window;w.fetch=()=>{throw Error('must not send')};w.eval(fs.readFileSync(path.join(base,'auth.js'),'utf8'));w.document.getElementById('password').value='short';w.document.getElementById('authForm').dispatchEvent(new w.Event('submit',{cancelable:true}));assert.match(w.document.getElementById('error').textContent,/five password/);dom.window.close()});
for(const route of ['login','signup']){
  test(route+' rejects typed/pasted whitespace before any request',async()=>{
    const dom=new JSDOM(fs.readFileSync(path.join(base,'auth.html'),'utf8'),{url:'http://localhost/'+route,runScripts:'outside-only'});
    const w=dom.window,d=w.document;let requests=0;
    w.fetch=async()=>{requests++;throw Error('must not send')};
    w.eval(fs.readFileSync(path.join(base,'auth.js'),'utf8'));
    for(const field of route==='signup'?['email','password','confirm']:['email','password']){
      for(const space of [' ','\t','\u00a0']){
        for(const position of ['start','middle','end']){
          d.getElementById('email').value='user@example.com';
          d.getElementById('password').value='Valid!Pass99';
          d.getElementById('confirm').value='Valid!Pass99';
          const input=d.getElementById(field),value=input.value;
          const i=position==='start'?0:position==='end'?value.length:2;
          // Text/password inputs remove ASCII tabs/newlines natively; test spaces/NBSP
          // and verify no sanitizer can silently strip boundary spaces from email.
          input.value=value.slice(0,i)+space+value.slice(i);
          if(!/\s/.test(input.value))continue;
          input.dispatchEvent(new w.Event('input'));
          assert.match(input.validationMessage,/whitespace/);
          await d.getElementById('authForm').onsubmit({preventDefault(){}});
          assert.match(d.getElementById('error').textContent,/whitespace/);
        }
      }
    }
    assert.equal(requests,0);
    const email=d.getElementById('email');email.value=' user@example.com ';
    assert.equal(email.value,' user@example.com ');
    email.value='user@example.com';email.dispatchEvent(new w.Event('input'));
    assert.equal(email.validationMessage,'');
    dom.window.close();
  });
}
