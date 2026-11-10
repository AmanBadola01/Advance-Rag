const path = require('path');
const root = path.resolve(__dirname, '..');
module.exports = { apps: [{
  name: 'retrieval-studio', cwd: root,
  script: path.join(root, 'run_workspace.py'),
  interpreter: path.join(root, 'backend/venv/bin/python'),
  instances: 1, exec_mode: 'fork', autorestart: true,
  restart_delay: 3000, max_restarts: 10, min_uptime: 10000,
  env: { PORT: '20304', RAYON_NUM_THREADS: '2', TOKIO_WORKER_THREADS: '2', OMP_NUM_THREADS: '2', OPENBLAS_NUM_THREADS: '2', HF_HOME: path.resolve(root, '../../outputs/backend-qa/hf-cache'), TIKTOKEN_CACHE_DIR: path.resolve(root, '../../outputs/backend-qa/token-cache'), RAG_MODEL_LOCAL_ONLY: 'true' }
}] };
