/**
 * TokenEconomizer for the DeepSeek Harness (deepseek-ai/deepseek-harness).
 *
 * DSH has no hook protocol -- a plugin is a Cordis module on the Host's event bus -- so this
 * bundle speaks the one the Claude Code, Codex and Pi integrations use, on DSH's behalf: every
 * event below becomes one call of the shared dispatcher (`../scripts/te-hook --host dsh`, JSON
 * in, JSON out), and the answer is applied with DSH's own levers. All of the mechanism logic
 * lives in the dispatcher; this file only translates.
 *
 *   agent/created      -> SessionStart     the learned facts, as a scoped system-prompt
 *                                          section (`tokeneconomizer`); the tool deny list as
 *                                          `tools.restrict` on the agent's scope
 *   agent/pre-step     -> UserPromptSubmit opens a prompt-turn; the map learned on the first
 *                                          prompt joins the messages that enter that step
 *   tools/post-execute -> PostToolUse      the guard; a redirect is added to the tool result as
 *                                          an additional context, a still-stuck notice goes
 *                                          through the model (the Web UI shows no hook text)
 *   turn/end           -> Stop             closes the turn, carrying DSH's reason
 *   session/disposed   -> SessionEnd       hands the exported session to a detached fold
 *
 * Sub-agent sessions (`parentSession`, `delegationDepth`) are left alone: they are not the
 * person's conversation, and folding them as tasks would inflate what the repository learned.
 *
 * The session itself is exported as JSONL into the plugin's state directory
 * (`<TOKENECONOMIZER_HOME>/dsh-sessions/<session>.jsonl`) because DSH's own log is
 * zstd-compressed and the Python package is standard-library only. The bundle holds the live
 * session, so it writes the events it already has.
 *
 * Side calls (the loop guard's judge and writer, micro-skills' proposer) go back INTO this
 * running harness: `POST /tokeneconomizer/sidecall` on the Host's local web server, guarded by a
 * per-process token, runs one `llm.stream` call on the session's provider and model. A `te`
 * command run from a terminal has no such endpoint, so it makes no model calls.
 *
 * Like the hooks on the other agents it fails open: a dispatcher that is missing, slow or broken
 * answers "no change". It does nothing under TE_SIDECALL=1 (the plugin's own side calls) or
 * TE_DISABLE=1. `/te <command>` runs the `te` command line for the repository of the session it
 * is typed in.
 */
import { spawn, spawnSync } from 'node:child_process'
import { randomBytes, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = (() => {
  try {
    return path.dirname(fileURLToPath(import.meta.url))
  } catch {
    return __dirname
  }
})()
const SCRIPTS = path.join(HERE, '..', 'scripts')
const SECTION = 'tokeneconomizer'
const SIDECALL_PATH = '/tokeneconomizer/sidecall'
const TOKEN = randomBytes(24).toString('hex')
const SIDECALL_TIMEOUT_MS = 180_000
const TIMEOUT_MS = {
  SessionStart: 15_000,
  UserPromptSubmit: 20_000,
  PostToolUse: 60_000,
  PostToolUseFailure: 60_000,
  Stop: 10_000,
  SessionEnd: 10_000,
}

/** Every live root agent this plugin knows about, by session id. */
const states = new Map()
/** The Host's web server, when this profile has one (it serves the side-call endpoint). */
let webServer = null

// The services a session cannot work without. `commands`, `webServer` and `llm` are optional:
// a headless harness has no Web UI to serve and no slash commands, and the plugin still learns
// and guards there.
export const inject = ['tools', 'systemPrompt']

const truthy = (v) => ['1', 'true', 'yes', 'on'].includes(String(v ?? '').trim().toLowerCase())

function stateRoot() {
  return process.env.TOKENECONOMIZER_HOME || path.join(os.homedir(), '.tokeneconomizer')
}

function safeName(id) {
  return String(id).replace(/[^A-Za-z0-9._-]/g, '_')
}

function log(message) {
  console.log(`[tokeneconomizer] ${message}`)
}

/** The program and arguments that run a launcher. The launchers are POSIX `sh`, which Windows
 * cannot start; there the bootstrap they run is started with Python itself (`TE_PYTHON`, else
 * `python`). */
function launch(script, args, platform = process.platform) {
  if (platform !== 'win32') return [path.join(SCRIPTS, script), args]
  const mode = script === 'te-hook' ? 'hook' : 'cli'
  return [process.env.TE_PYTHON || 'python', [path.join(SCRIPTS, 'te_bootstrap.py'), mode, ...args]]
}

/** Run one of the plugin's launchers with `input` on stdin; resolve to its stdout. */
function run(script, args, input, env, timeoutMs) {
  return new Promise((resolve) => {
    let out = ''
    let child
    try {
      const [program, argv] = launch(script, args)
      // UTF-8 mode as the launchers set it, for where they are bypassed (Windows).
      child = spawn(program, argv, {
        env: { ...process.env, PYTHONUTF8: '1', ...env },
        stdio: ['pipe', 'pipe', 'ignore'],
        windowsHide: true,
      })
    } catch {
      resolve('')
      return
    }
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      resolve('')
    }, timeoutMs)
    // Decoded as a stream, so a character split across two chunks arrives whole.
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', (d) => (out += d))
    child.on('error', () => {
      clearTimeout(timer)
      resolve('')
    })
    child.on('close', () => {
      clearTimeout(timer)
      resolve(out)
    })
    child.stdin.on('error', () => {})
    child.stdin.end(input)
  })
}

function hookEnv(state) {
  const env = {
    TE_DSH: '1',
    // The Web UI is the interactive one; a profile without its server (the headless app) is
    // told apart here, so a still-stuck notice says how to end the run.
    TE_DSH_MODE: webServer ? 'web' : 'headless',
    TE_DSH_SIDECALL: sidecallUrl(),
    TE_DSH_TOKEN: TOKEN,
  }
  if (state?.provider) env.TE_DSH_PROVIDER = state.provider
  if (state?.model) env.TE_DSH_MODEL = state.model
  return env
}

/** One dispatcher call; `{}` (no change) on any failure. */
async function hook(state, event, payload) {
  if (truthy(process.env.TE_SIDECALL) || truthy(process.env.TE_DISABLE)) return {}
  const out = await run(
    'te-hook',
    ['--host', 'dsh'],
    JSON.stringify(payload),
    hookEnv(state),
    TIMEOUT_MS[event] || 20_000,
  )
  const line = out.trim().split('\n').pop() || '{}'
  let answer = {}
  try {
    answer = JSON.parse(line) || {}
  } catch {
    return {}
  }
  // The hosts that show hook text put it here; the Web UI does not, so it goes to the log and
  // the still-stuck notice is relayed through the model (see `Dsh.relays_notices`).
  if (typeof answer.systemMessage === 'string' && answer.systemMessage) log(answer.systemMessage)
  return answer
}

function sidecallUrl() {
  if (!webServer) return ''
  try {
    return new URL(SIDECALL_PATH, webServer.getOrigin()).toString()
  } catch {
    return ''
  }
}

// ---- message shapes -------------------------------------------------------------------------
function textOf(content) {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .filter((b) => b && b.type === 'text')
    .map((b) => b.text || '')
    .join('\n')
}

/** A user-role message this plugin adds to a prompt step or a tool result. The marker keeps
 * `agent/pre-step` from mistaking it for a prompt the person typed. */
function userMessage(text) {
  return {
    id: randomUUID(),
    role: 'user',
    content: [{ type: 'text', text }],
    source: { kind: 'user', tokeneconomizer: true },
  }
}

function isPrompt(message) {
  return Boolean(
    message
      && message.role === 'user'
      && message.source?.kind === 'user'
      && !message.source?.tokeneconomizer,
  )
}

function resultText(result) {
  const text = textOf(result?.content)
  const failure = result?.error?.message
  return failure && !text.includes(failure) ? (text ? `${text}\n` : '') + failure : text
}

/** The exit code a shell result carries: DSH's own `[exit code: N]` line, or its metadata. */
function exitCodeOf(result) {
  const meta = result?.meta
  if (meta && typeof meta === 'object') {
    for (const key of ['exitCode', 'exit_code', 'returncode']) {
      if (typeof meta[key] === 'number') return meta[key]
    }
  }
  const m = /\[exit code: (-?\d+)\]|exit code[:\s]+(-?\d+)/i.exec(resultText(result))
  return m ? Number(m[1] ?? m[2]) : undefined
}

// ---- the session export ---------------------------------------------------------------------
/** Write the session's header and every event it already has, and return the file's path. */
function openTranscript(state) {
  const dir = path.join(stateRoot(), 'dsh-sessions')
  const file = path.join(dir, `${safeName(state.id)}.jsonl`)
  const lines = []
  const session = state.agent.session
  const header = session?.header
  if (header) lines.push(JSON.stringify({ type: 'session', ...header }))
  let events = []
  try {
    events = typeof session?.snapshotEvents === 'function' ? session.snapshotEvents() : []
  } catch {
    events = []
  }
  for (const event of events) lines.push(JSON.stringify(event))
  try {
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
    fs.writeFileSync(file, lines.length ? `${lines.join('\n')}\n` : '', { mode: 0o600 })
  } catch (error) {
    log(`could not export the session: ${error?.message || error}`)
    return ''
  }
  state.lastSeq = events.reduce(
    (top, event) => (typeof event.seq === 'number' ? Math.max(top, event.seq) : top), -1)
  return file
}

function appendEvent(state, event) {
  if (!state.transcript || typeof event?.type !== 'string') return
  if (typeof event.seq === 'number' && event.seq <= state.lastSeq) return
  try {
    fs.appendFileSync(state.transcript, `${JSON.stringify(event)}\n`)
    if (typeof event.seq === 'number') state.lastSeq = event.seq
  } catch {
    return
  }
}

// ---- per-agent wiring -----------------------------------------------------------------------
function install(agent) {
  const header = agent?.session?.header || {}
  if (header.parentSession || Number(header.delegationDepth || 0) > 0) return null
  const state = {
    agent,
    id: String(agent.id),
    cwd: header.cwd || agent.session?.header?.cwd || process.cwd(),
    transcript: '',
    manual: '',
    prompts: 0,
    promptId: '',
    lastSeq: -1,
    provider: '',
    model: '',
    ended: false,
    ending: false,
    // One hook call at a time per session: DSH can run tool calls in parallel, and the
    // dispatcher's per-session state is a read-modify-write.
    queue: Promise.resolve(),
  }
  states.set(state.id, state)
  state.transcript = openTranscript(state)
  const dispose = agent.ctx.effect(() => {
    const disposers = [
      agent.ctx.systemPrompt.section({
        name: SECTION,
        order: 900,
        text: () => state.manual,
      }),
      agent.ctx.on('agent/pre-step', (payload, next) => onPreStep(state, payload, next)),
      agent.ctx.on('tools/post-execute', (exec, result, next) => onTool(state, exec, result, next)),
    ]
    return () => {
      for (const off of disposers) {
        try {
          off()
        } catch {
          /* a service that is already gone */
        }
      }
    }
  })
  // Keyed in the plugin's own effect too: unloading the plugin disposes what it registered on
  // another context's scope.
  state.disposeOnUnload = ctxEffect(() => () => {
    try {
      if (typeof dispose === 'function') dispose()
    } catch {
      /* already disposed */
    }
    states.delete(state.id)
  })
  return state
}

/** Registered by `apply`; a module-level seam so `install` can own its cleanup even when the
 * test harness calls it directly. */
let ctxEffect = (callback) => {
  const disposer = callback()
  return disposer
}

function payloadOf(state) {
  return { session_id: state.id, transcript_path: state.transcript, cwd: state.cwd }
}

/** Run `task` after every hook call already queued for this session. */
function enqueue(state, task) {
  const run = state.queue.then(task, task)
  state.queue = run.catch(() => {})
  return run
}

async function onPreStep(state, payload, next) {
  const decision = await next()
  if (!decision || decision.kind !== 'enter') return decision
  if (payload.agent && String(payload.agent.id) !== state.id) return decision
  const prompt = (payload.messages || []).filter(isPrompt).pop()
  if (!prompt) return decision
  const text = textOf(prompt.content)
  if (!text.trim()) return decision
  state.prompts += 1
  state.promptId = `${state.id}:${state.prompts}`
  const extra = await enqueue(state, async () => {
    const answer = await hook(state, 'UserPromptSubmit', {
      ...payloadOf(state),
      hook_event_name: 'UserPromptSubmit',
      prompt_id: state.promptId,
      prompt: text,
    })
    return answer?.hookSpecificOutput?.additionalContext
  })
  // Spreading `decision` keeps fields such as `startsRequestSeries`.
  if (typeof extra === 'string' && extra.trim()) {
    return { ...decision, messages: [...decision.messages, userMessage(extra)] }
  }
  return decision
}

async function onTool(state, exec, result, next) {
  const decision = await next()
  if (!decision || decision.kind !== 'accept') return decision
  if (!exec || (exec.agent && String(exec.agent.id) !== state.id)) return decision
  const failed = Boolean(result?.isError)
  const extra = await enqueue(state, async () => {
    const answer = await hook(state, failed ? 'PostToolUseFailure' : 'PostToolUse', {
      ...payloadOf(state),
      hook_event_name: failed ? 'PostToolUseFailure' : 'PostToolUse',
      prompt_id: state.promptId,
      tool_name: exec.name,
      tool_use_id: exec.callId,
      tool_input: exec.arguments && typeof exec.arguments === 'object' ? exec.arguments : {},
      tool_response: {
        content: resultText(result),
        isError: failed,
        exit_code: exitCodeOf(result),
      },
    })
    return answer?.hookSpecificOutput?.additionalContext
  })
  if (typeof extra !== 'string' || !extra.trim()) return decision
  const contexts = Array.isArray(decision.additionalContexts)
    ? [...decision.additionalContexts]
    : []
  contexts.push(userMessage(extra))
  return { ...decision, additionalContexts: contexts }
}

function onEvent(state, event) {
  if (!event || typeof event.type !== 'string') return
  appendEvent(state, event)
  if (event.type === 'request/context') {
    if (typeof event.data?.provider === 'string') state.provider = event.data.provider
    if (typeof event.data?.model === 'string') state.model = event.data.model
  } else if (event.type === 'request/header') {
    const config = event.data?.header?.config
    if (config) {
      if (typeof config.provider === 'string') state.provider = config.provider
      if (typeof config.model === 'string') state.model = config.model
    }
  } else if (event.type === 'turn/end') {
    onEndTurn(state, event)
  }
}

function onEndTurn(state, event) {
  if (state.ended) return Promise.resolve()
  const reason = event?.data?.reason?.kind || 'completed'
  return enqueue(state, () => hook(state, 'Stop', {
    ...payloadOf(state),
    hook_event_name: 'Stop',
    prompt_id: state.promptId,
    stop_hook_active: false,
    stop_reason: reason,
  }))
}

function onEnd(state, reason) {
  if (state.ended || state.ending || !state.transcript) return Promise.resolve()
  state.ending = true
  return enqueue(state, async () => {
    await hook(state, 'SessionEnd', {
      ...payloadOf(state),
      hook_event_name: 'SessionEnd',
      reason: reason || 'disposed',
    })
    state.ended = true
    states.delete(state.id)
  })
}

/** The harness can quit without disposing its sessions (a `dsh headless` run does), so Node's
 * own exit event is a last chance where the host lets it run: a synchronous SessionEnd, whose
 * hook only hands the exported session to a detached fold. The electron host exits without
 * firing it, and such a session is folded by the next session start instead. */
function endLiveSessions() {
  for (const state of states.values()) {
    if (state.ended || !state.transcript) continue
    state.ended = true
    try {
      const [program, argv] = launch('te-hook', ['--host', 'dsh'])
      spawnSync(program, argv, {
        input: JSON.stringify({
          ...payloadOf(state),
          hook_event_name: 'SessionEnd',
          reason: 'exit',
        }),
        env: { ...process.env, PYTHONUTF8: '1', ...hookEnv(state) },
        stdio: ['pipe', 'ignore', 'ignore'],
        timeout: 5000,
        windowsHide: true,
      })
    } catch {
      /* exiting is not the moment to report anything */
    }
  }
}

/** The deny list the SessionStart answer carried, as a restriction on this agent's scope. */
function applyRestrictions(agent, items, tools) {
  if (!Array.isArray(items) || !items.length) return
  const names = items.filter((name) => {
    try {
      return Boolean(tools.get(name))
    } catch {
      return false
    }
  })
  if (!names.length) return
  try {
    agent.ctx.tools.restrict({ deny: names })
  } catch (error) {
    log(`could not restrict ${names.join(', ')}: ${error?.message || error}`)
  }
}

// ---- side calls -----------------------------------------------------------------------------
function reply(res, status, body) {
  try {
    res.statusCode = status
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify(body))
  } catch {
    /* the client is gone */
  }
}

function readBody(req, limit = 1 << 20) {
  return new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > limit) {
        reject(new Error('request too large'))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

async function runSideCall(llm, body) {
  const { prompt, provider, model } = body || {}
  if (typeof prompt !== 'string' || !provider || !model) {
    throw new Error('missing prompt/provider/model')
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), SIDECALL_TIMEOUT_MS)
  let text = ''
  let usage = null
  try {
    const stream = llm.stream({
      provider,
      model,
      signal: controller.signal,
      messages: [{ role: 'user', content: [{ type: 'text', text: prompt }] }],
    })
    for await (const chunk of stream) {
      if (chunk?.type === 'text-delta') text += chunk.text || ''
      else if (chunk?.type === 'usage') usage = chunk.usage || usage
      else if (chunk?.type === 'finish' && chunk.reason?.kind === 'error') {
        throw new Error(chunk.reason.failure?.message || 'the model call failed')
      } else if (chunk?.type === 'finish' && chunk.reason?.kind === 'aborted') {
        throw new Error('the model call was aborted')
      }
    }
  } finally {
    clearTimeout(timer)
  }
  return { text, usage }
}

function handleSideCall(llm, req, res) {
  const auth = String(req.headers?.authorization || '')
  if (auth !== `Bearer ${TOKEN}`) {
    reply(res, 403, { error: 'forbidden' })
    return
  }
  if (req.method !== 'POST') {
    reply(res, 405, { error: 'method not allowed' })
    return
  }
  readBody(req).then(async (raw) => {
    let body = {}
    try {
      body = JSON.parse(raw || '{}')
    } catch {
      reply(res, 400, { error: 'bad json' })
      return
    }
    try {
      reply(res, 200, await runSideCall(llm, body))
    } catch (error) {
      reply(res, 500, { error: String(error?.message || error) })
    }
  }).catch((error) => reply(res, 400, { error: String(error?.message || error) }))
}

// ---- the plugin -----------------------------------------------------------------------------
/** `te <command>` for the session the command is typed in. */
async function runTe(invocation) {
  const agent = invocation?.agent
  const state = agent ? states.get(String(agent.id)) : undefined
  const cwd = state?.cwd || agent?.session?.header?.cwd || process.cwd()
  const argv = String(invocation?.rawInput || '').trim().split(/\s+/).filter(Boolean)
  if (!argv.length) argv.push('status')
  const env = { TE_HOST: 'dsh' }
  if (state) {
    env.TE_SESSION_ID = state.id
    if (state.provider) env.TE_DSH_PROVIDER = state.provider
    if (state.model) env.TE_DSH_MODEL = state.model
  }
  const out = await run('te', ['--host', 'dsh', '--cwd', cwd, ...argv], '', env, 60_000)
  return { kind: 'success', text: out.trim() || '(no output; is Python 3.10+ on PATH?)' }
}

export function apply(ctx, config = {}) {
  ctxEffect = (callback) => ctx.effect(callback)
  ctx.on('agent/created', async ({ agent, source }) => {
    if (truthy(process.env.TE_DISABLE)) return
    const state = install(agent)
    if (!state) return
    const answer = await hook(state, 'SessionStart', {
      ...payloadOf(state),
      hook_event_name: 'SessionStart',
      source: ['startup', 'resume', 'clear', 'compact'].includes(source) ? source : 'startup',
    })
    state.manual = answer?.hookSpecificOutput?.additionalContext || ''
    applyRestrictions(agent, answer?.toolRestrictions, ctx.tools)
  })
  ctx.on('session/disposed', async (session) => {
    const state = states.get(String(session?.id))
    if (state) await onEnd(state, 'disposed')
  })
  // The append feed is watched from the plugin's own context and filtered by session id: the
  // session's carrier scope is the agent factory's, not the agent's, so a listener on
  // `agent.ctx` is not guaranteed to see it.
  ctx.on('session/event', (session, event) => {
    const state = states.get(String(session?.id))
    if (state) onEvent(state, event)
  })
  ctx.on('agent/disposed', ({ agent }) => {
    const state = states.get(String(agent?.id))
    if (state) onEnd(state, 'disposed')
  })
  // `/te <command>` exists where the profile has a command registry (the Web UI does; a
  // headless harness has no composer to type it in). Services outside `inject` are reached
  // through `ctx.inject`, which is also how the plugin stays inactive in a profile without
  // them instead of throwing at activation.
  if (typeof ctx.inject === 'function') {
    ctx.inject(['commands'], (child) => {
      child.effect(() => child.commands.register({
        name: 'te',
        description:
          'TokenEconomizer: what it learned and did in this repository (status | microskills | '
          + 'toolprune | config | repos | doctor)',
        input: { hint: 'status | microskills | toolprune | config ... | repos' },
        handler: (invocation) => runTe(invocation),
      }))
    })
  }
  // The side-call endpoint exists only where the Host serves the Web UI: a terminal `te` has no
  // harness to ask, and the dispatcher then learns only what needs no model call.
  if (typeof ctx.inject === 'function') {
    ctx.inject(['webServer', 'llm'], (child) => {
      if (!child.webServer || !child.llm) return
      webServer = child.webServer
      child.effect(() => child.webServer.register({
        kind: 'exact',
        path: SIDECALL_PATH,
        handler: (req, res) => handleSideCall(child.llm, req, res),
      }))
    })
  }
  // Unloading the plugin must also dispose what it registered on each agent's scope.
  ctx.effect(() => () => {
    for (const state of states.values()) {
      try {
        state.disposeOnUnload?.()
      } catch {
        /* already gone */
      }
    }
    states.clear()
    webServer = null
  })
  ctx.effect(() => {
    process.on('exit', endLiveSessions)
    return () => process.off('exit', endLiveSessions)
  })
}
