"""DeepSeek Harness (deepseek-ai/deepseek-harness) session logs.

DSH stores a session as `session.v4.jsonl.zstd` under `~/.dsh/sessions/<escaped cwd>/`:
a zstd-compressed JSONL file whose first line is the session header

    {"type": "session", "version": 4, "id": "session-...", "createdAt": 1793587318600,
     "cwd": "/repo", ...}

and whose remaining lines are committed events, each `{"type", "seq", "time", "data"}`.
The plugin in `plugin/dsh/` writes the same lines uncompressed to the plugin's state
directory (it holds the live session, so it needs no zstd decoder and no Python package
beyond the standard library). The readers here work on either file.

What the mechanisms need from a session:

  * `user/message` with `data.source.kind == "user"` -- a prompt the person typed. DSH
    injects its own context as user messages too (workspace instructions, runtime context,
    the skill catalog, the time), each with its own `source.kind`; those are not prompts.
  * `assistant/message` -- the model's reply, its `tool-call` blocks, and the request's
    `usage` ({inputTokens, outputTokens, totalTokens, cacheReadTokens, cacheWriteTokens,
    reasoningTokens}). `inputTokens` excludes the cache fields: totalTokens is
    inputTokens + outputTokens + cacheReadTokens (verified on a session here).
  * `tool/call` / `tool/result` -- the call's name and arguments (a JSON string), and its
    result, paired by `callId`. A failed call sets `message.isError` and carries `error`.
  * `request/header` -- the exact tool schemas the request was sent with, which is the
    catalog tool pruning needs (there is no CLI probe on DSH).

`dsh_session_to_claude` writes all of it as the Claude Code envelope the other readers
already handle, so no mechanism needs a DSH branch.
"""
import json
import os

from .jsonl import lines as _lines


def dsh_session_header(path):
    """The session header line ({id, cwd, version, ...}), {} when absent."""
    for line in _lines(path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "session" and "seq" not in ev:
            return ev
        break                       # the header is the first line; anything else is no header
    return {}


def dsh_events(path):
    """[(seq, event)] in log order, header excluded."""
    out = []
    for line in _lines(path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict) or "type" not in ev:
            continue
        if ev.get("type") == "session" and "seq" not in ev:
            continue
        out.append((ev.get("seq"), ev))
    return out


def _text(content):
    """The text of a message's content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _prompt(ev):
    """Is this event a prompt the person typed? DSH's injected context is a user message too,
    and says so in `source.kind`; the ones this plugin adds carry `tokeneconomizer` there."""
    if ev.get("type") != "user/message":
        return False
    d = ev.get("data") or {}
    source = d.get("source") if isinstance(d.get("source"), dict) else {}
    return (source.get("kind") == "user" and not source.get("tokeneconomizer")
            and bool(_text(d.get("content")).strip()))


def dsh_prompt_count(path):
    """How many prompts the person gave the session: user messages whose source is the user."""
    return sum(1 for _seq, ev in dsh_events(path) if _prompt(ev))


def _usage(u):
    if not isinstance(u, dict):
        return None
    return {"input_tokens": int(u.get("inputTokens") or 0),
            "cache_read_input_tokens": int(u.get("cacheReadTokens") or 0),
            "cache_creation_input_tokens": int(u.get("cacheWriteTokens") or 0),
            "output_tokens": int(u.get("outputTokens") or 0)}


def dsh_session_usage(path):
    """Token breakdown of a DSH session, one entry per assistant message that carries usage."""
    by_id, models = {}, {}
    for _seq, ev in dsh_events(path):
        if ev.get("type") != "assistant/message":
            continue
        d = ev.get("data") or {}
        u = _usage(d.get("usage"))
        if u is None:
            continue
        msg = d.get("message") or {}
        mid = msg.get("id") or "seq-%s" % ev.get("seq")
        by_id[mid] = u
        model = ((msg.get("source") or {}) if isinstance(msg.get("source"), dict) else {}).get(
            "model")
        if model:
            models[model] = models.get(model, 0) + 1
    out = {k: sum(u[k] for u in by_id.values()) for k in
           ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
            "output_tokens")}
    out["requests"] = len(by_id)
    out["model"] = max(models, key=models.get) if models else None
    out["files"] = 1 if os.path.exists(path) else 0
    return out


def dsh_catalog(path):
    """{tool_name: description} the session's requests were sent with.

    The last `request/header` wins: a session that ran under a tool restriction was sent the
    restricted list, which is the catalog its evidence is about."""
    best = {}
    for _seq, ev in dsh_events(path):
        if ev.get("type") != "request/header":
            continue
        header = (ev.get("data") or {}).get("header") or {}
        tools = header.get("tools") if isinstance(header, dict) else None
        if isinstance(tools, list) and tools:
            best = {str(t.get("name")): str(t.get("description") or "")[:200]
                    for t in tools if isinstance(t, dict) and t.get("name")}
    return best


def _tool_result_text(d):
    """The text of a `tool/result` event, with the error the harness recorded, if any."""
    msg = d.get("message") or {}
    text = _text(msg.get("content"))
    err = d.get("error")
    if isinstance(err, dict):
        line = " ".join(str(err.get(k) or "") for k in ("name", "code", "reason")).strip()
        if line:
            text = (text + "\n" if text else "") + line
    return text


def dsh_session_to_claude(path, out_path):
    """Write a DSH session as Claude Code transcript lines; the number of tool calls written,
    or None when the file is missing.

    The last request's tool list rides along as one `{"type": "catalog", "tools": [...]}`
    line: the readers of the Claude envelope ignore an unknown type, and `trace.dsh_catalog`
    reads it back from the converted trace."""
    if not path or not os.path.exists(path):
        return None
    cwd = dsh_session_header(path).get("cwd")
    cwd = cwd if isinstance(cwd, str) else None
    calls = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for _seq, ev in dsh_events(path):
            t, d = ev.get("type"), ev.get("data") or {}
            if t == "user/message" and _prompt(ev):
                out.write(json.dumps({"type": "user", "cwd": cwd, "message": {
                    "role": "user", "content": [{"type": "text", "text": _text(d.get("content"))}]}},
                    ensure_ascii=False) + "\n")
            elif t == "assistant/message":
                msg = d.get("message") or {}
                content = []
                for b in msg.get("content") or []:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text" and b.get("text"):
                        content.append({"type": "text", "text": b["text"]})
                    elif b.get("type") == "tool-call":
                        try:
                            args = json.loads(b.get("arguments") or "{}")
                        except (TypeError, ValueError):
                            args = {}
                        content.append({"type": "tool_use", "id": b.get("id"),
                                        "name": b.get("name"),
                                        "input": args if isinstance(args, dict) else {}})
                        calls += 1
                entry = {"type": "assistant", "cwd": cwd, "message": {
                    "id": msg.get("id"), "role": "assistant",
                    "model": ((msg.get("source") or {}) if isinstance(msg.get("source"), dict)
                              else {}).get("model"),
                    "content": content}}
                usage = _usage(d.get("usage"))
                if usage:
                    entry["message"]["usage"] = usage
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            elif t == "tool/result":
                msg = d.get("message") or {}
                out.write(json.dumps({"type": "user", "cwd": cwd, "message": {
                    "role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": msg.get("toolCallId"),
                        "content": _tool_result_text(d),
                        "is_error": bool(msg.get("isError") or d.get("error"))}]}},
                    ensure_ascii=False) + "\n")
        cat = dsh_catalog(path)
        if cat:
            out.write(json.dumps({"type": "catalog", "tools": [
                {"name": n, "description": desc} for n, desc in cat.items()]},
                ensure_ascii=False) + "\n")
    return calls
