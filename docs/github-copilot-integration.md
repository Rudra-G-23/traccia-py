# GitHub Copilot integration

Traccia integrates with Copilot through command hooks. Each hook invocation
sanitizes the payload and appends a JSONL record to the local session journal.
The completed journal is replayed into session, tool, and subagent spans by
the flush module.

## Supported surfaces

Copilot CLI is supported when Traccia and its configured exporter are installed
in the same environment as the hook command. Repository hook files use the
portable `python` command and must not contain a local developer's absolute
interpreter path.

Cloud-agent support requires a runtime containing Traccia and an allowlisted
OTLP endpoint. Cloud jobs have an ephemeral filesystem and restricted outbound
network access, so repository hooks use a synchronous session-end flush. If
those prerequisites are absent, cloud hooks cannot export after the job ends.

## Privacy

Metadata-only mode is the default. The journal keeps only allowlisted Copilot
fields; prompts, tool arguments/results, subagent descriptions and responses,
and compaction instructions are represented by lengths. Content capture is
explicitly opt-in, redacted, and size-capped.

## Failure and recovery

Session journals are claimed atomically before replay. A failed or incomplete
export is retained in `failed/` and can be retried with
`traccia copilot flush --retry-failed`. A journal that changes while it is
being exported is retained so late hook events are not lost.
