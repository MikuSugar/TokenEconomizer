"""DeepSeek Harness: the session-log reader, the host adapter, the deny list it applies in its
own state, and the `plugin/dsh/` bundle that translates DSH's event bus into the dispatcher.

The last test drives the real bundle through a stand-in Cordis context with `node`, so the
translation is exercised against the real Python hook (no DSH process needed)."""
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(REPO, "plugin", "dsh")
SID = "session-11111111-2222-3333-4444-555555555555"


def _events(cwd):
    return [
        {"type": "session", "version": 4, "id": SID, "createdAt": 1, "cwd": str(cwd),
         "isSeeded": False, "delegationDepth": 0},
        {"type": "permission/preset", "seq": 0, "time": 1,
         "data": {"preset": "workspace-write"}},
        {"type": "user/message", "seq": 1, "time": 2, "data": {
            "content": [{"type": "text", "text": "Run the tests"}],
            "source": {"kind": "user"}, "role": "user", "id": "u1"}, "surfaceOp": "append"},
        # DSH's own injected context is a user message too; it is not a prompt.
        {"type": "user/message", "seq": 2, "time": 3, "data": {
            "content": [{"type": "text", "text": "<system-reminder>workspace rules</system-reminder>"}],
            "source": {"kind": "runtime-context"}, "role": "user", "id": "u2"},
         "surfaceOp": "append"},
        {"type": "request/header", "seq": 3, "time": 4, "data": {"reason": "initial", "header": {
            "config": {"provider": "deepseek", "model": "deepseek-chat"},
            "tools": [{"name": "bash", "description": "Run a shell command", "parameters": {}},
                      {"name": "read", "description": "Read a file", "parameters": {}}]}}},
        {"type": "assistant/message", "seq": 4, "time": 5, "data": {
            "turn": 1, "step": 1, "stream": [],
            "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 120,
                      "cacheReadTokens": 10},
            "message": {"id": "a1", "role": "assistant", "content": [
                {"type": "text", "text": "Running the tests."},
                {"type": "tool-call", "id": "c1", "name": "bash",
                 "arguments": "{\"command\": \"python -m pytest -q\"}"}],
                "source": {"kind": "model", "provider": "deepseek", "model": "deepseek-chat"}}}},
        {"type": "tool/call", "seq": 5, "time": 5, "data": {
            "turn": 1, "step": 1, "callId": "c1", "name": "bash",
            "arguments": "{\"command\": \"python -m pytest -q\"}"}},
        {"type": "tool/result", "seq": 6, "time": 6, "data": {
            "turn": 1, "step": 1, "error": {"name": "CommandError", "code": "EXIT_1"},
            "message": {"role": "tool", "isError": True, "toolCallId": "c1",
                        "content": [{"type": "text", "text": "No module named pytest"}],
                        "source": {"kind": "tool", "callId": "c1"}}}},
        {"type": "turn/end", "seq": 7, "time": 7,
         "data": {"turn": 1, "reason": {"kind": "completed"}}},
        # Context this plugin adds is a user message too; it is not a prompt either.
        {"type": "user/message", "seq": 8, "time": 8, "data": {
            "content": [{"type": "text", "text": "repository map: ..."}],
            "source": {"kind": "user", "tokeneconomizer": True}, "role": "user", "id": "u3"},
         "surfaceOp": "append"},
    ]


def _session(tmp_path, cwd=None):
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in _events(cwd or tmp_path)) + "\n",
                    encoding="utf-8")
    return path


def test_the_session_log_reads_prompts_usage_and_the_tool_catalog(tmp_path):
    from tokeneconomizer import trace
    src = _session(tmp_path)
    assert trace.dsh_prompt_count(str(src)) == 1          # not the injected runtime context
    assert trace.dsh_session_usage(str(src)) == {
        "input_tokens": 100, "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 0, "output_tokens": 10,
        "requests": 1, "model": "deepseek-chat", "files": 1}
    assert trace.dsh_catalog(str(src)) == {
        "bash": "Run a shell command", "read": "Read a file"}
    assert trace.dsh_session_header(str(src))["cwd"] == str(tmp_path)


def test_a_converted_session_feeds_the_shared_readers(tmp_path):
    from tokeneconomizer import trace
    src = _session(tmp_path)
    out = tmp_path / "trace.jsonl"
    assert trace.dsh_session_to_claude(str(src), str(out)) == 1
    assert trace.load_trajectory(str(out)) == [
        ("bash", {"command": "python -m pytest -q"},
         "No module named pytest\nCommandError EXIT_1", False)]
    assert trace.tool_usage(str(out)) == {"bash": 1}
    # The last request's schemas ride along as one catalog line.
    assert trace.parse_tool_catalog_dsh(out.read_text(encoding="utf-8")) == {
        "bash": "Run a shell command", "read": "Read a file"}
    # A missing log is no trace, as on every host.
    assert trace.dsh_session_to_claude(str(tmp_path / "gone"), str(out)) is None


def test_the_host_adapter_maps_dsh_payloads(tmp_path):
    from tokeneconomizer.plugin import hosts
    h = hosts.Dsh()
    payload = {"session_id": SID, "prompt_id": f"{SID}:2", "tool_name": "bash",
               "tool_input": {"command": "make test"},
               "tool_response": {"content": "boom", "isError": True, "exit_code": 2}}
    assert h.name == "dsh"
    assert h.turn_key(payload) == f"{SID}:2"
    assert h.tool_call(payload) == ("bash", {"command": "make test"}, "boom", 2)
    assert h.stopped_normally({"stop_reason": "aborted"}) is False
    assert h.stopped_normally({"stop_reason": "completed"}) is True
    assert h.stopped_normally({}) is True
    assert h.tool_restrictions(["plugin_manager", "workflow"]) == {
        "toolRestrictions": ["plugin_manager", "workflow"]}
    assert h.tool_restrictions([]) == {}
    # A finished session's export converts; the transcript is the exported spool.
    out = tmp_path / "dsh_trace.jsonl"
    assert h.mechanism_trace(str(_session(tmp_path)), str(tmp_path)) == str(out)
    assert h.prompt_count(str(_session(tmp_path))) == 1
    assert h.session_usage(str(_session(tmp_path)))["requests"] == 1
    # Its own export is removed after the fold; DSH's own log never is.
    spool = tmp_path / "dsh-sessions" / f"{SID}.jsonl"
    spool.parent.mkdir()
    shutil.copyfile(_session(tmp_path), spool)
    h.settle(str(spool))
    assert not spool.exists()
    other = tmp_path / "session.jsonl"
    h.settle(str(other))
    assert other.exists()


def test_the_deny_list_lives_in_the_plugin_state_and_reaches_the_session_start(tmp_path):
    from tokeneconomizer.plugin import denylist, hook
    from tokeneconomizer.plugin.paths import repo_identity, repo_state_dir
    from tokeneconomizer.plugin.store import Store
    work = tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    (work / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    env = {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
           "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    repo = repo_identity(str(work), env)
    store = Store(repo_state_dir(repo, "dsh", env))
    assert denylist.apply(str(work), ["plugin_manager"], store, host="dsh") == ["plugin_manager"]
    assert not os.path.exists(os.path.join(str(work), ".dsh"))     # nothing in the repository
    assert denylist.applied_for(store, str(work), "dsh") == ["plugin_manager"]
    assert denylist.apply(str(work), ["plugin_manager"], store, host="dsh") == []
    payload = {"hook_event_name": "SessionStart", "session_id": SID, "cwd": str(work),
               "source": "startup", "transcript_path": str(tmp_path / "missing.jsonl")}
    out = hook.run(payload, env, host_hint="dsh")
    assert out["toolRestrictions"] == ["plugin_manager"]
    # A list given back stops reaching the sessions; nothing else changes.
    assert denylist.restore(str(work), store, ["plugin_manager"], host="dsh") == [
        "plugin_manager"]
    out = hook.run(payload, env, host_hint="dsh")
    assert "toolRestrictions" not in out


def test_folding_a_dsh_session_learns_its_tools_and_its_ledger_record(tmp_path):
    """The generic fold path over a DSH export: a Claude-shaped trace feeds the mechanisms, the
    catalog comes from the request header, and the record prices the session as `dsh`."""
    import time
    from tokeneconomizer.plugin import fold
    from tokeneconomizer.plugin.context import LiveContext
    from tokeneconomizer.plugin.hosts import Dsh
    work = tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    (work / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    env = {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_MICROSKILLS": "envfacts",
           "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    ctx = LiveContext.build(Dsh(), str(work), env)
    rec = None
    for n in (1, 2):
        events = [
            {"type": "session", "version": 4, "id": f"session-{n}", "createdAt": 1,
             "cwd": str(work), "isSeeded": False, "delegationDepth": 0},
            {"type": "user/message", "seq": 0, "time": 1, "data": {
                "content": [{"type": "text", "text": "run the tests"}],
                "source": {"kind": "user"}, "role": "user", "id": "u1"},
             "surfaceOp": "append"},
            {"type": "request/header", "seq": 1, "time": 2, "data": {
                "reason": "initial", "header": {
                    "config": {"provider": "p", "model": "m"},
                    "tools": [{"name": "bash", "description": "Run a shell command",
                               "parameters": {}},
                              {"name": "read", "description": "Read a file",
                               "parameters": {}}]}}},
            {"type": "assistant/message", "seq": 2, "time": 3, "data": {
                "turn": 1, "step": 1, "stream": [],
                "usage": {"inputTokens": 10, "outputTokens": 2},
                "message": {"id": f"a{n}", "role": "assistant", "content": [
                    {"type": "tool-call", "id": f"c{n}", "name": "bash",
                     "arguments": "{\"command\": \"pytest -q\"}"}],
                    "source": {"kind": "model", "provider": "p", "model": "m"}}}},
            {"type": "tool/call", "seq": 3, "time": 3, "data": {
                "turn": 1, "step": 1, "callId": f"c{n}", "name": "bash",
                "arguments": "{\"command\": \"pytest -q\"}"}},
            {"type": "tool/result", "seq": 4, "time": 4, "data": {
                "turn": 1, "step": 1, "error": {"name": "CommandError", "code": "EXIT_1"},
                "message": {"role": "tool", "isError": True, "toolCallId": f"c{n}",
                            "content": [{"type": "text",
                                         "text": "No module named pytest"}],
                            "source": {"kind": "tool", "callId": f"c{n}"}}}},
            {"type": "turn/end", "seq": 5, "time": 5,
             "data": {"turn": 1, "reason": {"kind": "completed"}}},
        ]
        path = tmp_path / f"session-{n}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
        ctx.store.write(f"sessions/session-{n}.json", {
            "session_id": f"session-{n}", "host": "dsh", "started": time.time(),
            "ended": time.time(), "transcript": str(path), "turns": []})
        rec = fold.fold_session(ctx, f"session-{n}")
        assert rec is not None

    assert rec["host"] == "dsh" and rec["prompts"] == 1 and "error" not in rec
    assert rec["input_tokens"] == 10 and rec["model"] == "m"
    unt = ctx.store.read("untool.json")
    assert sorted(unt["used_tools"]) == ["bash"]
    assert sorted(unt["catalog_tools"]) == ["bash", "read"]     # from the request header
    assert unt["sessions"] == 2
    assert (ctx.store.read("injection.json") or {}).get("sessions") == 2


def test_the_bundle_is_a_dsh_package():
    with open(os.path.join(BUNDLE, "package.json"), encoding="utf-8") as fh:
        pkg = json.load(fh)
    assert pkg["type"] == "module" and pkg["exports"]["."] == "./index.js"
    patch = pkg["dsh"]["bundle"]["patch"]
    with open(os.path.join(BUNDLE, patch), encoding="utf-8") as fh:
        text = fh.read()
    assert "name: tokeneconomizer-dsh" in text and "id: tokeneconomizer" in text
    for name in ("index.js", "cordis.patch.yml", "icon.svg", "locale/en.json", "locale/zh.json",
                 "README.md"):
        assert os.path.isfile(os.path.join(BUNDLE, name)), name
    for pattern in ("index.js", "cordis.patch.yml", "locale/*.json", "icon.svg", "README.md"):
        assert pattern in pkg["files"], pattern
    assert pkg["exports"]["./icon"] == "./icon.svg"
    assert pkg["exports"]["./locale/*.json"] == "./locale/*.json"
    for locale, word in (("en.json", "TokenEconomizer"), ("zh.json", "TokenEconomizer")):
        with open(os.path.join(BUNDLE, "locale", locale), encoding="utf-8") as fh:
            meta = json.load(fh)["meta"]
        assert meta["title"] == word and len(meta["description"]) > 20
    with open(os.path.join(BUNDLE, "icon.svg"), encoding="utf-8") as fh:
        icon = fh.read()
    assert icon.lstrip().startswith("<svg") and len(icon) < 256 * 1024
    src = open(os.path.join(BUNDLE, "index.js"), encoding="utf-8").read()
    for event in ("agent/created", "agent/pre-step", "tools/post-execute", "session/disposed"):
        assert f"'{event}'" in src, event
    assert "['--host', 'dsh']" in src and "TE_DSH_SIDECALL" in src
    assert "parentSession" in src                  # sub-agent sessions are left alone


HARNESS = r"""
import fs from "node:fs"
import { pathToFileURL } from "node:url"

const mod = await import(pathToFileURL(process.env.TE_PLUGIN_INDEX).href)
const pluginListeners = new Map()

function ctxFor(extra = {}) {
  const listeners = new Map()
  const ctx = {
    on(name, fn) {
      const list = listeners.get(name) || []
      list.push(fn)
      listeners.set(name, list)
      return () => {
        const i = list.indexOf(fn)
        if (i >= 0) list.splice(i, 1)
      }
    },
    effect(fn) {
      const d = fn()
      return d
    },
    inject(deps, cb) {
      if (!deps.every((dep) => ctx[dep])) return
      const child = ctxFor({})
      for (const dep of deps) child[dep] = ctx[dep]
      cb(child)
    },
    logger: { info() {}, warn() {} },
    get() { return undefined },
    listeners,
    ...extra,
  }
  return ctx
}

const calls = { section: "", restrict: [] }
const agentCtx = ctxFor({
  systemPrompt: {
    section(section) {
      calls.section = section
      return () => {}
    },
  },
  tools: {
    get: (name) => ({ name }),
    schemas: () => [],
    restrict(filter) {
      calls.restrict.push(filter)
      return () => {}
    },
  },
})

const ctx = ctxFor({
  tools: agentCtx.tools,
  systemPrompt: agentCtx.systemPrompt,
  commands: { register: (d) => { console.log("COMMAND " + d.name); return () => {} } },
})
await mod.apply(ctx, {})

const agent = {
  id: process.env.TE_SESSION,
  ctx: agentCtx,
  session: {
    header: { version: 4, id: process.env.TE_SESSION, createdAt: 1,
              cwd: process.env.TE_WORK, isSeeded: false, delegationDepth: 0 },
    seq: -1,
    snapshotEvents: () => [],
  },
}

async function fire(name, ...args) {
  for (const fn of ctx.listeners.get(name) || []) await fn(...args)
}
async function fireAgent(name, ...args) {
  for (const fn of agentCtx.listeners.get(name) || []) return fn(...args)
  return undefined
}

await fire("agent/created", { agent, source: "startup" })
const manual = calls.section.text ? calls.section.text() : ""
console.log("MANUAL " + JSON.stringify(manual))
console.log("SECTION " + calls.section.name)
console.log("RESTRICT " + JSON.stringify(calls.restrict))

const messages = [{ id: "u1", role: "user", source: { kind: "user" },
  content: [{ type: "text", text: "please fix the tests" }] }]
const decided = await fireAgent("agent/pre-step",
  { agent, turn: 1, step: 1, messages, signal: new AbortController().signal },
  async () => ({ kind: "enter", messages }))
console.log("PRESTEP " + decided.kind + " " + decided.messages.length)

const tool = await fireAgent("tools/post-execute",
  { agent, name: "bash", callId: "c1", arguments: { command: "echo hi" },
    signal: new AbortController().signal },
  { isError: false, content: [{ type: "text", text: "hi" }], meta: { exitCode: 0 } },
  async () => ({ kind: "accept", content: [{ type: "text", text: "hi" }] }))
console.log("TOOL " + tool.kind + " " + JSON.stringify(tool.additionalContexts || []))

await fire("session/event", agent.session,
  { type: "turn/end", seq: 9, time: 9, data: { turn: 1, reason: { kind: "completed" } } })
await fire("session/disposed", { id: process.env.TE_SESSION })
console.log("DONE")
"""


def test_the_bundle_translates_dsh_events_to_the_dispatcher(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    if os.name == "nt":
        pytest.skip("the harness drives the POSIX launcher; Windows uses the same bootstrap path")
    from tokeneconomizer.plugin import denylist
    from tokeneconomizer.plugin.paths import repo_identity, repo_state_dir
    from tokeneconomizer.plugin.store import Store
    work = tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    (work / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    home = tmp_path / "home"
    env = {"TOKENECONOMIZER_HOME": str(home), "TE_FOLD_SYNC": "1",
           "TE_MICROSKILLS": "envfacts", "TE_PYTHON": sys.executable,
           "PATH": os.environ.get("PATH", ""),
           "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    repo = repo_identity(str(work), env)
    store = Store(repo_state_dir(repo, "dsh", env))
    store.write("injection.json", {"rendered": "- this repository runs its tests with pytest -q"})
    assert denylist.apply(str(work), ["plugin_manager"], store, host="dsh") == ["plugin_manager"]

    harness = tmp_path / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    run = subprocess.run(
        [node, str(harness)],
        capture_output=True, text=True, timeout=180, cwd=str(work),
        env=dict(env, TE_PLUGIN_INDEX=os.path.join(BUNDLE, "index.js"), TE_SESSION=SID,
                 TE_WORK=str(work), HOME=os.environ.get("HOME", "")))
    out = run.stdout
    assert run.returncode == 0, (out, run.stderr)
    assert 'MANUAL "- this repository runs its tests with pytest -q"' in out, out
    assert "SECTION tokeneconomizer" in out
    assert 'RESTRICT [{"deny":["plugin_manager"]}]' in out, out
    assert "PRESTEP enter 1" in out                       # facts did not rewrite the prompt
    assert "TOOL accept []" in out                        # a normal call gets no redirect
    assert "DONE" in out

    # The dispatcher answered every event: a session record, its fold, and the ledger.
    sessions = list((home / "repos").glob("*/dsh/sessions/*.json"))
    assert sessions and sessions[0].name == f"{SID}.json", sessions
    record = json.loads(sessions[0].read_text(encoding="utf-8"))
    assert record["host"] == "dsh" and record["ended"]
    ledger = list((home / "repos").glob("*/dsh/ledger.jsonl"))
    assert ledger, list((home / "repos").glob("*"))
    rows = [json.loads(line) for line in ledger[0].read_text(encoding="utf-8").splitlines()]
    assert any(r.get("kind") == "session" and r.get("host") == "dsh" for r in rows), rows
    # The export is the harness's spool, not the session's own log: it is gone after the fold.
    spool = home / "dsh-sessions" / f"{SID}.jsonl"
    assert not spool.exists()
