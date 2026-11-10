import pytest
from fastapi.testclient import TestClient
from app.integrated import app
from app import auth
from app.core.account_context import account_id
from app.core.container import container
from app.services.pipeline_tracer import pipeline_tracer

def test_account_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(auth,'ROOT',tmp_path/'accounts')
    auth._attempts.clear()
    a,b=TestClient(app),TestClient(app)
    credentials={'email':'a@example.com','password':'Valid!Pass99'}
    assert a.get('/api/documents').status_code==401
    assert a.get('/workspace',follow_redirects=False).status_code==303
    assert a.post('/api/auth/signup',json={**credentials,'password':'short'}).status_code==400
    response=a.post('/api/auth/signup',json=credentials)
    assert response.status_code==200
    assert 'httponly' in response.headers['set-cookie'].lower()
    uid=response.json()['id']
    token=account_id.set(uid)
    try:
        container.register_raw_document('private-doc','private.txt',[{'text':'evidence'}],'recursive')
        trace=pipeline_tracer.start_trace('private question','hybrid')
        pipeline_tracer.end_trace(trace)
        pipeline_tracer.save_answer(trace, {"answer": "Account-private answer"})
    finally:
        account_id.reset(token)
    assert len(a.get('/api/documents').json())==1
    assert b.post('/api/auth/signup',json={**credentials,'email':'b@example.com'}).status_code==200
    assert b.get('/api/documents').json()==[]
    assert b.get('/api/query/traces').json()==[]
    assert b.get('/api/query/trace/'+trace).status_code==404
    b.delete('/api/documents/private.txt')
    assert len(a.get('/api/documents').json())==1
    assert a.delete('/api/auth/session').status_code==200
    assert a.get('/api/documents').status_code==401
    assert a.post('/api/auth/login',json={**credentials,'password':'wrong'}).status_code==401
    assert a.post('/api/auth/login',json=credentials).status_code==200
    container._containers.pop(uid)
    pipeline_tracer._tracers.pop(uid)
    assert len(a.get('/api/documents').json())==1
    assert a.get('/api/query/trace/'+trace).json()['saved_answer']['answer']=='Account-private answer'
    assert a.delete('/api/documents/private.txt',headers={'Origin':'https://untrusted.example'}).status_code==403
