# Frontend/backend integration

The integrated workspace is served at `/workspace` by `app.integrated:app`. The original preview at `code/rag-preview` is preserved. The original API root, health, OpenAPI and Swagger documentation remain available. Existing backend QA fixes are preserved.

## Removed from the frontend

- Evaluations navigation, benchmark runner, simulated dataset, and evaluation CSV export: no evaluation API exists.
- Fabricated faithfulness, relevancy, precision, recall, dashboard trends, and sample query history.
- Query transformation controls: HyDE, multi-query expansion, decomposition and step-back prompting are not implemented.
- Search-only query tab: the backend query endpoint performs retrieval plus generation; indexed content remains searchable in Chunk explorer.
- Semantic chunking, OR filter combination, upload date and tag filtering. Tags are accepted by the API but ignored; the UI exposes only functional filters.
- Independent re-ranking toggle, replaced by the explicit Hybrid + re-ranking retrieval strategy; other retrievers do not support that toggle.
- Clear server history button: the backend has no trace deletion API. Refresh traces replaces it.
- Sample content, simulated pipeline animation, placeholder statistics and preview upload/remove behavior.

## Exposed backend features

- All five retrieval strategies, top K, normalized semantic/BM25 weights, optional generation model.
- Multipart PDF, DOCX, TXT, MD/Markdown, CSV, JSON upload with recursive, section or parent-child ingestion; real catalog and deletion by filename. Upload completion reports actual chunk/token/page counts.
- Query source, page range, exact section and chunk strategy filters (AND).
- Actual answers, citations, retrieved evidence, raw scores, ranks and metadata.
- Server execution traces, step status, duration and details, total estimated tokens, recent trace history, query reuse.
- Corpus or document-scoped A/B comparisons; per-pipeline name, retriever, chunker, chunk size/overlap, parent/child size, top K and weights; answers, evidence, execution latency and delta. Chunk controls disabled for corpus mode because the API ignores them there.
- Chunk inspector with source/strategy filters, parent-context inspection, result limit, loaded-subset text search, token/character counts and metadata.
- Backend health/readiness, index/parent/document counts and API documentation link.

## Run

From the repository root, use the existing environment with requirements installed:

```sh
PYTHONPATH=backend PORT=20318 GEMINI_MODEL=gemini-3.5-flash-lite backend/venv/bin/python run_workspace.py
```

The launcher maps the injected `SECRET_GEMINI_API_KEY` to `GOOGLE_API_KEY` in process memory only. It stores no secret. `GEMINI_MODEL` can select another provider-supported model. Set `HF_HOME` and `TIKTOKEN_CACHE_DIR` to provisioned caches, and `RAG_MODEL_LOCAL_ONLY=true` for offline model loading. Use one worker and preserve `CHROMA_PERSIST_DIRECTORY`.

The deployed process uses the project cache under `outputs/backend-qa`. The app has no authentication or per-user isolation. It shares the owner's resource limit, survives this chat, and requires restart after a server reboot.

## Backend limits reflected in the UI

Document listing has page counts but no aggregate chunk/token counts. Same-name uploads delete together. Chunk inspection returns at most 200 rows with no pagination; search explicitly applies to loaded rows. Traces are in memory and reset on restart. Token totals in traces are estimates. A/B latency excludes index preparation; no quality metrics exist. Model selection is free text because there is no model catalog endpoint. LLM generation may fall back to extraction; trace details expose fallback errors.

Provider calls are bounded to 30 seconds with automatic retries disabled; provider failures remain visible in trace details and a frontend extraction notice.

## Document-first upload flow (2026-11-05)

Navigation order is Documents, Playground, Compare strategies, Chunk explorer; Documents opens by default. Upload controls live in Documents. Files are added to a browser-held queue first; only then is the chunking selector shown. “Chunk and index documents” sends those files with the chosen technique to the existing atomic ingestion endpoint. No separate server-side staging endpoint is implied. Cancel clears the queue; successful files are removed from it so a retry only processes remaining files.

## Verified model replacement (2026-11-05)

Default generation is now `gemini-3.1-flash-lite`. Verified after service restart with two real document queries without model overrides; both returned answers and citations without extraction fallback. Evidence: `outputs/default-model-verification.json` at the Crew root. Earlier Gemini availability failures in QA are historical results.

## Answer workspace redesign

Answers render Markdown through locally bundled Marked and DOMPurify (versions and licenses in `frontend/vendor/`). The UI separates Answer, Sources and Evidence, links numbered inline citations to their supporting passages, and keeps pipeline timing collapsed below the answer. Each comparison result has independent tabs and copy controls. Copy exports only answer text, preserving reference numbers; HTTP clipboard fallback is supported. Loading, empty, query failure and extraction fallback states are explicit. A trace lookup failure no longer discards an otherwise successful query response.

UI checks: from `frontend`, run `npm ci` then `npm test`. Node is needed for development checks only; the served workspace has no build step or runtime CDN dependency.

## Retrieval controls (2026-11-06)

Only the Hybrid + re-ranking dropdown option is labeled Recommended in Playground. The generation-model field has been removed; Playground requests use the backend default.

## Visible document scope (2026-11-06)

Playground has an All documents / individual document selector above the question. This is the single source of truth for `filters.source`; the duplicate source control was removed from Metadata filters. Additional metadata filters and their reset preserve the selected scope. The active search scope is visible before querying; answers retain the scope used at request time, with contributing document names and pages shown under Sources used. Changing scope marks the old answer as needing a new query. Empty retrieval results show a scope-specific no-evidence message.

Source filtering is by filename in the existing backend. Duplicate filenames are grouped and labeled with their upload count. If a selected filename disappears on refresh, the UI preserves the selection as unavailable and blocks the query until another scope is chosen; it never silently broadens retrieval.

## Accounts (2026-11-06)

`/login` and `/signup` provide same-origin email/password authentication. Passwords require 8–128 characters, upper/lowercase, a number and a special character; signup shows a live strength estimate, requirement checklist and confirmation. Password hashes use salted scrypt. Opaque sessions are stored hashed on the server and issued as HttpOnly, SameSite=Strict cookies (Secure on HTTPS), expire after seven days and are revoked on logout. Cross-origin writes are rejected; login/signup attempts are rate limited.

Authenticated request context selects a separate persistent Chroma index, raw document registry and saved query/answer history per server-generated account ID. Query history retains up to 1,000 traces per account and offers an Answer button to reopen completed answers. Unsubmitted files and unsent form drafts remain browser-only. Existing legacy shared data is preserved without automatically assigning it to any new account. Account records, sessions and tenant data live under `backend/data/accounts` (override with `RAG_ACCOUNT_DIRECTORY`); back up this entire directory. Run one worker: the registry and storage locks are process-local. Recovery emails and email verification are not configured.

Verification: `RAYON_NUM_THREADS=2 TOKIO_WORKER_THREADS=2 PYTHONPATH=backend backend/venv/bin/python -m pytest backend/tests/test_accounts.py -q`; `npm --prefix frontend test`. The thread limits avoid Rust database worker initialization hangs in this runtime and are defaults in the launcher.

## Help & guide (2026-11-06)

The former Research team placeholder is now a Help & guide navigation button. The guide covers purpose, quick start, three implemented chunkers with benefits/trade-offs, five retrievers with a comparison table, pipeline concepts, the actual integrated tech stack and troubleshooting. It also explains document scope, accounts, provider fallback, absent OCR and unavailable techniques without claiming assignment requirements are implemented. Guide navigation preserves current query state; the entry remains available on mobile. Existing knowledge statistics and account/sign-out controls are preserved.

Frontend verification: 13 tests pass, including guide navigation, scope preservation and contents targets. Desktop/mobile visual checks used a temporary static fixture of the same markup/styles because the managed browser had no signed-in app session. The fixture was removed after verification; no account was created and no authentication was bypassed.


## 2026-11-06 model availability switch

The current launcher and backend default is `gemini-3.5-flash-lite`. The earlier `gemini-3.1-flash-lite` verification above is historical. A live provider probe for the replacement succeeded before changing the defaults. Runtime verification is recorded in Crew `outputs/model-change-2026-11-06-verification.json`.

## 2026-11-06 — Credential whitespace validation
- Login and signup reject whitespace anywhere in email/password, including boundary spaces, tabs, line breaks and non-breaking spaces. Passwords are never trimmed or changed.
- Browser input validation catches typed/pasted whitespace and prevents requests; backend rejects invalid credentials before lookup, hashing or storage so API calls cannot bypass the rule.
- Email uses text + email inputmode + an email pattern, because native email inputs silently strip leading/trailing ASCII spaces before JavaScript can validate them. Case-insensitive email lookup remains unchanged.
- Existing account data is not rewritten. Existing passwords containing whitespace will be rejected under this requested policy.
- Verification: 49 isolated backend cases and all 15 frontend tests passed; evidence in outputs/auth-whitespace-backend.log and outputs/auth-whitespace-frontend.log.

## 2026-11-06 — Document upload/processing progress
- One status bar outside the pending-file card stays visible after the queue empties. Status changes from Uploading document (real transfer percentage) to Chunking and indexing document (indeterminate animation) to Document ready.
- XMLHttpRequest upload progress/load events distinguish browser transfer completion from waiting for the existing combined ingestion endpoint; no backend stage events or fake chunking percentages. Processing label covers parsing, chunking, embeddings and persistence.
- Multiple files process sequentially with filename + file position. Successful files leave the queue; failed/unattempted files remain. Disabled controls guard duplicate submissions. Network/timeout messages warn to check indexed documents before retrying, since server processing may finish despite a connection loss.
- Uses existing design tokens, screen-reader status/progress semantics, and reduced-motion fallback. No dependencies, API changes or service restart needed; app asset version bumped.
- Verification: 18 frontend tests passed, including transfer events, indeterminate processing, duplicate-submit guard, multi-file failure/retry, network/timeout/abort cleanup. Managed browser had no account session; no authenticated upload end-to-end test claimed.

## 2026-11-06 — Playground refresh controls
- Document scope Refresh now has its own guarded action, loading state, visible success/error status and no-store document request. It refreshes the list without waiting for the unrelated health endpoint, retains selected scope/question/answer, and does not silently widen a deleted source.
- General refresh applies the document catalog before health checking, so an unavailable health endpoint no longer discards a successful catalog response.
- Refresh traces also shows pending/completion feedback and restores its control after errors. Refreshed data may be unchanged; feedback now makes that distinction visible.
- Verified 22 passing frontend tests, including new catalog content, uncached request, selection retention, duplicate click guard, failure/retry, health failure and trace-refresh cleanup. Evidence: outputs/playground-refresh-tests.log. App asset version bumped; no backend restart needed.

## 2026-11-06 — Personal header greeting
- Replaced the empty breadcrumb area with a time-aware greeting using the name from the authenticated /api/auth/me endpoint. Removed duplicate name from right-side controls.
- Uses browser-local time: morning05:00–11:59, afternoon12:00–16:59, evening17:00–04:59. Rechecks each minute and on returning to the tab. Missing names omit the comma; names render via textContent, never HTML.
- Header permits wrapping on small screens and long names. Existing theme/signout/connection controls remain.
- Verified 23 passing frontend tests; outputs/user-greeting-tests.log covers time boundaries, profile names, fallback and safe text rendering.

## 2026-11-06 — Readable validation errors
- Playground blocks blank, fractional or out-of-range Final top-K values before submitting; focuses the field and shows an accessible inline message: Enter a whole number between 1 and 100 for Final top-K chunks. Correcting the value clears the error.
- Shared API error formatting converts backend validation arrays into readable field messages and handles structured server messages without exposing raw JSON. Query errors display the actionable message directly rather than burying it in Error details.
- Backend validation remains unchanged and authoritative. Verified 25 frontend tests, including invalid inputs, valid bounds1/100 and simulated422 fallback: outputs/topk-validation-tests.log.
