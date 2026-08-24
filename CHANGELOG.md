# Changelog

## [Unreleased]

### Added
- Redaction allowlist for `traccia.policy.*` span attributes (same pattern as `traccia.prompt.*`)
- HTTP client skip for `@govern()` status/block calls and prompt-runtime fetches, matching the Node SDK

## [0.1.27] - 2026-08-14

### Added
- `evaluate()` for offline experiments (platform dataset, local+persist, local-only)
- Eval-runtime client; local built-in scorers (`exact_match`, `contains`, `json_valid`); platform judge/code via server score
- Eval span attrs (`traccia.experiment.*`, `traccia.eval.source`, `traccia.dataset.*`) with redaction allowlist
- `result.summary()`, `result.url`, progress `N/M`, per-item error isolation, default persist and max_concurrency=10

### Fixed
- `start_tracing()` / `init()` now apply `TRACCIA_API_KEY` / `TRACCIA_ENDPOINT` from env (flat config); nested env load previously dropped Authorization on OTLP export
- `evaluate()` auto-inits tracing when needed and only attaches `trace_id` when export is configured (avoids phantom Open Trace links)
- Builtin scores now include `type` + `scorer_name`; panel label is `Task` (or `prompt=` name); cells record `latency_ms` and LLM/`judge` cost when present

## [0.1.26] - 2026-08-09

### Fixed
- `LoadedPrompt.compile` now stamps `traccia.prompt.id` on the active span (with name, version, version_id, label, is_fallback). Prompt Metrics joins prefer this id over name-only matching.

## [0.1.25] - 2026-07-16

### Added
- `load_prompt` / `prefetch_prompts` with TTL cache (~60s), stale-while-revalidate, and explicit fallback
- `{{var}}` compile helpers (`LoadedPrompt.compile`) with shared golden fixtures
- Auto span attributes `traccia.prompt.*` on compile (name, version, version_id, label, is_fallback)
- `init(prompt_cache_ttl_s=...)` / `TRACCIA_PROMPT_CACHE_TTL_S` for cache TTL
- `init(prompt_api_base=...)` / `TRACCIA_PROMPT_API_BASE` when prompt-runtime host differs from the traces host (advanced deployments only)
- Redaction allowlist so `traccia.prompt.*` identity keys are not wiped by `"prompt"` substring matching

### Fixed
- `init(auto_start_trace=True)` now attaches the auto-started root span to OTel context so child spans share one trace
