"""Same-origin, cookie-based authentication with persistent salted password hashes."""
import hashlib
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

router = APIRouter(prefix='/api/auth')
ROOT = Path(os.getenv('RAG_ACCOUNT_DIRECTORY', './backend/data/accounts'))
_lock = threading.RLock()
_attempts = {}
COOKIE = 'retrieval_session'
TTL = 60 * 60 * 24 * 7

def read_store():
    path = ROOT / 'accounts.json'
    return json.loads(path.read_text()) if path.exists() else {'users': {}, 'sessions': {}}

def save_store(data):
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = ROOT / 'accounts.json'
    temp = path.with_suffix('.new')
    temp.write_text(json.dumps(data))
    temp.chmod(0o600)
    temp.replace(path)

def password_valid(p):
    return (8 <= len(p) <= 128 and re.search('[A-Z]', p) and re.search('[a-z]', p)
            and re.search('[0-9]', p) and re.search(r'[^\w\s]', p))

def digest(password, salt):
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()

def current_user(request):
    token = request.cookies.get(COOKIE, '')
    with _lock:
        data = read_store()
        session = data['sessions'].get(hashlib.sha256(token.encode()).hexdigest())
        if session and session['expires'] > time.time():
            return data['users'].get(session['uid'])
    return None

def public(user):
    return {k: user[k] for k in ('id', 'email', 'name')}

class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)
    name: str = Field(default='', max_length=80)

@router.post('/{action}')
def authenticate(action: str, request: Request, response: Response, body: Credentials):
    if action not in ('login', 'signup'):
        raise HTTPException(404)
    # Reject rather than normalize whitespace, especially in passwords.
    if re.search(r'\s', body.email):
        raise HTTPException(400, 'Email address must not contain spaces or other whitespace.')
    if re.search(r'\s', body.password):
        raise HTTPException(400, 'Password must not contain spaces or other whitespace.')
    email = body.email.lower()
    if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        raise HTTPException(400, 'Enter a valid email address.')
    with _lock:
        key = request.client.host if request.client else 'unknown'
        now = time.time()
        _attempts[key] = [t for t in _attempts.get(key, []) if t > now - 900]
        if len(_attempts[key]) >= 30:
            raise HTTPException(429, 'Too many attempts. Please try again in 15 minutes.')
        _attempts[key].append(now)
        data = read_store()
        user = next((u for u in data['users'].values() if u['email'] == email), None)
        if action == 'signup':
            if not password_valid(body.password):
                raise HTTPException(400, 'Use 8–128 characters with uppercase, lowercase, a number and a special character.')
            if user:
                raise HTTPException(409, 'An account with this email already exists. Sign in instead.')
            salt = secrets.token_hex(16)
            user = dict(id=secrets.token_hex(16), email=email, name=body.name.strip() or email.split('@')[0], salt=salt, password=digest(body.password, salt))
            data['users'][user['id']] = user
        else:
            candidate = digest(body.password, user['salt'] if user else '00' * 16)
            if not user or not secrets.compare_digest(candidate, user['password']):
                raise HTTPException(401, 'Email or password is incorrect.')
        token = secrets.token_urlsafe(32)
        data['sessions'] = {k:v for k,v in data['sessions'].items() if v['expires'] > now}
        data['sessions'][hashlib.sha256(token.encode()).hexdigest()] = {'uid': user['id'], 'expires': now + TTL}
        save_store(data)
    response.set_cookie(COOKIE, token, httponly=True, samesite='strict', secure=request.url.scheme == 'https', max_age=TTL)
    response.headers['Cache-Control'] = 'no-store'
    return public(user)

@router.get('/me')
def me(request: Request):
    user = current_user(request)
    if not user:
        raise HTTPException(401, 'Please sign in.')
    return public(user)

@router.delete('/session')
def logout(request: Request, response: Response):
    with _lock:
        data = read_store()
        data['sessions'].pop(hashlib.sha256(request.cookies.get(COOKIE, '').encode()).hexdigest(), None)
        save_store(data)
    response.delete_cookie(COOKIE)
    return {'ok': True}
