# Retrieval Studio deployment

Live address: http://116.202.210.102:20304/login

Frontend and API are served together by one Uvicorn worker on 0.0.0.0:20304.
Port 20304 was selected by testing bind availability inside the authorized 20300–20330 range.
Existing service on 20318 was preserved.

PM2 runs detached in foreground mode because ordinary daemon startup hangs in this restricted runtime. Its CPU/memory statistics cannot read /proc; application serving works. Automatic process restart is enabled. No server-boot service is configured; a full server reboot requires relaunch. This is a stable IP/port address while this server and port remain assigned, not a guaranteed permanent domain.

From the project root, with the Vault Gemini secret injected in the environment:

```sh
setsid nohup code/Advance-Rag/deploy/start.sh > code/Advance-Rag/deploy/supervisor.log 2>&1 < /dev/null &
```

Do not launch a duplicate when PM2 is already running. Management commands use:

```sh
export PM2_HOME="$PWD/code/Advance-Rag/deploy/.pm2"
pm2 list
pm2 restart retrieval-studio
pm2 delete retrieval-studio
```

Do not use pm2 save: PM2 dumps can persist injected secret environment variables. Configuration contains no credentials. Cached models are reused from outputs/backend-qa. Preserve backend/data/accounts and the other existing data directories. The old 20318 process was not stopped; use the new link for current access.

Verification: login/signup/assets return 200; workspace redirects unauthenticated users to login; auth/me returns 401 without a session. Browser rendered Sign in · Retrieval Studio via the public IP. These checks originate from server infrastructure and do not independently establish external firewall reachability. All 25 frontend tests passed. Authenticated ingestion/generation was not exercised in this deployment check.

Anyone who can reach the address can open the login/signup pages. Account content requires authentication. The service shares the owner's 2 CPU/8 GB resource limit. HTTPS and a custom domain are not configured.
