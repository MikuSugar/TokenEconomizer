# TokenEconomizer for the DeepSeek Harness

The DSH bundle. It runs the same dispatcher as the Claude Code, Codex, Copilot, Pi and opencode
integrations (`../scripts/te-hook --host dsh`) and applies its answers with DSH's own levers. The
mechanisms and what they learn are described in the [plugin README](../README.md); this file is
only about the harness.

## Install

DSH installs a bundle from a package directory. From a clone of this repository, in the
Harness's own profile:

1. Install `plugin/dsh` as a bundle (Plugin Manager → install from a local directory, or the
   `plugin_manager` tool with `install_bundle` and the absolute path of `plugin/dsh`).
2. Keep the clone where it is: the bundle runs `../scripts/te-hook` from it.
3. Start a new session. The row's title and description come from `locale/*.json`.

The bundle needs Python 3.10 or newer on `PATH`, or `TE_PYTHON` pointing at one.

## What it does, per session

- **Session start.** It calls `SessionStart` once per root agent, keeps the answer's facts as a
  system-prompt section named `tokeneconomizer`, and applies the tool deny list with
  `tools.restrict` on that agent's scope. Sub-agent sessions are left alone.
- **Each prompt.** `agent/pre-step` opens a prompt-turn; the repository map learned on the first
  prompt joins that step's messages.
- **Each tool result.** `tools/post-execute` folds the call into the loop guard. A redirect is
  added to the tool result as an additional context; a still-stuck notice is relayed through the
  model, because the Web UI does not show hook text.
- **Turn end / session end.** `turn/end` closes the turn; `session/disposed` hands the exported
  session to a detached fold. A `dsh headless` run exits without disposing its session, so its
  fold is left to the next session start, once the export has been quiet for ten minutes — the
  same fallback the other hosts use for a session that was killed. The bundle also fires
  `SessionEnd` from Node's own exit event where the host lets that event run.

The session is exported to `<TOKENECONOMIZER_HOME>/dsh-sessions/<session>.jsonl` (the harness's
own log is zstd-compressed, and the Python package is standard library only). The export is
removed once the session has been folded.

## Commands

Type `/te` in a session:

```
/te status
/te microskills
/te toolprune
/te config set microskills=on
/te config set toolprune=off loopguard=off
/te repos
/te doctor
```

The same commands run from a terminal in the repository as
`~/.tokeneconomizer/bin/te --host dsh <command>`.

## Tool pruning

The deny list is applied by this bundle, so nothing is written into the repository. It takes
effect from the next session; `/te toolprune restore` gives tools back. The catalog comes from
each session's own `request/header`, so there is no `refresh-catalog` to run. `toolprune measure`
is not available yet.

## Model calls

The plugin's own calls (the loop guard's judge and writer, micro-skills' proposer) run inside
the running harness: the bundle serves `POST /tokeneconomizer/sidecall` on the Host's local web
server, guarded by a per-process token, and runs one `llm.stream` call on the session's provider
and model. A `te` command run from a plain terminal has no harness to ask, so it makes no model
calls; facts that need none are still learned.

## Uninstall

1. In each repository where tools were left out: `/te toolprune restore`.
2. Remove the bundle from the profile.
3. Optional: `rm -rf ~/.tokeneconomizer` to delete what it learned.
