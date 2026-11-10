"""Serve the integrated workspace and existing API from a single origin."""
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.main import app

frontend = Path(__file__).resolve().parents[2] / 'frontend'
app.mount('/ui', StaticFiles(directory=frontend), name='workspace-assets')

@app.get('/workspace', include_in_schema=False)
def workspace():
    return FileResponse(frontend / 'index.html')

@app.get('/login', include_in_schema=False)
@app.get('/signup', include_in_schema=False)
def login_page():
    return FileResponse(frontend / 'auth.html', headers={'Cache-Control': 'no-store'})
