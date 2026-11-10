"""Exercise real auth routes with isolated storage, without loading RAG models."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app import auth


@pytest.mark.parametrize('action', ['signup', 'login'])
@pytest.mark.parametrize('field', ['email', 'password'])
@pytest.mark.parametrize('whitespace', [' ', '\t', '\n', '\u00a0'])
@pytest.mark.parametrize('position', ['start', 'end', 'middle'])
def test_reject_whitespace(tmp_path, monkeypatch, action, field, whitespace, position):
    monkeypatch.setattr(auth, 'ROOT', tmp_path / 'accounts')
    app = FastAPI()
    app.include_router(auth.router)
    credentials = {'email': 'a@example.com', 'password': 'Valid!Pass99'}
    value = credentials[field]
    offset = {'start': 0, 'end': len(value), 'middle': 2}[position]
    credentials[field] = value[:offset] + whitespace + value[offset:]
    response = TestClient(app).post('/api/auth/' + action, json=credentials)
    assert response.status_code == 400
    assert 'whitespace' in response.json()['detail']
    assert 'set-cookie' not in response.headers
    assert not (auth.ROOT / 'accounts.json').exists()


def test_valid_signup_login_and_exact_password(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, 'ROOT', tmp_path / 'accounts')
    monkeypatch.setattr(auth, '_attempts', {})
    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)
    credentials = {'email': 'User@example.com', 'password': 'Valid!Pass99'}
    assert client.post('/api/auth/signup', json=credentials).status_code == 200
    assert client.post('/api/auth/login', json={**credentials, 'email': 'user@example.com'}).status_code == 200
    assert client.post('/api/auth/login', json={**credentials, 'password': 'Valid!Pass98'}).status_code == 401
