"""Launch the workspace without storing credentials in files."""
import os
import sys
from pathlib import Path
os.environ.setdefault('RAYON_NUM_THREADS', '2')
os.environ.setdefault('TOKIO_WORKER_THREADS', '2')
import uvicorn

root = Path(__file__).resolve().parent
os.chdir(root)
sys.path.insert(0, str(root / "backend"))
if os.getenv('SECRET_GEMINI_API_KEY') and not os.getenv('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = os.environ['SECRET_GEMINI_API_KEY']
os.environ.setdefault('GEMINI_MODEL', 'gemini-3.5-flash-lite')
uvicorn.run('app.integrated:app', host='0.0.0.0', port=int(os.environ.get('PORT', '20318')), workers=1)
