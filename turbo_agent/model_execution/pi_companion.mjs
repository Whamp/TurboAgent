// Turbo Agent Pi ModelRuntime companion.
//
// One Node process owning a Pi ModelRuntime so Python can execute models
// through Pi's providers and subscription credentials without ever seeing
// them. Speaks versioned newline-delimited JSON on stdio: requests in on
// stdin, events out on stdout, diagnostics on stderr only.
//
// Wire protocol (v1):
//   -> {"v":1,"id":"req-1","op":"complete","provider":"openai-codex",
//       "model":"<pi-model-id>","context":{},"options":{}}
//   -> {"v":1,"id":"req-1","op":"stream", ...as above...}
//   -> {"v":1,"id":"req-1","op":"cancel"}
//   <- {"v":1,"id":"req-1","event":"delta","delta":{"type":"text","text":".."}}
//   <- {"v":1,"id":"req-1","event":"done","result":{...canonical result...}}
//   <- {"v":1,"id":"req-1","event":"error","error":{"kind":"..","retryable":..,
//                                                   "message":".."}}
//
// The runtime implementation is injectable for contract tests through
// TURBO_PI_RUNTIME_SETUP (module exporting createRuntime()). Production
// resolves Pi's installed package and creates one ModelRuntime with Pi's
// standard auth path.

import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import readline from "node:readline";
import { pathToFileURL } from "node:url";
import path from "node:path";

const PROTOCOL_VERSION = 1;

const require_ = createRequire(import.meta.url);

function log(message) {
  process.stderr.write(`[turbo-pi-companion] ${message}\n`);
}

async function loadRuntime() {
  const setupPath = process.env.TURBO_PI_RUNTIME_SETUP;
  if (setupPath) {
    const mod = await import(pathToFileURL(path.resolve(setupPath)).href);
    return mod.createRuntime();
  }
  const pi = requirePiPackage();
  log("creating Pi ModelRuntime with standard auth path");
  return pi.ModelRuntime.create({ refreshOnCreate: true });
}

function requirePiPackage() {
  const moduleName = "@earendil-works/pi-coding-agent";
  const modulePath = process.env.TURBO_PI_MODULE_PATH;
  if (modulePath) {
    return require_(modulePath);
  }
  try {
    return require_(moduleName);
  } catch (localErr) {
    // Not resolvable from this cwd (the usual case when Turbo runs from an
    // arbitrary directory). Fall back to the user's global npm root, which
    // is where a global `pi` install lives.
    try {
      const globalRoot = execFileSync("npm", ["root", "-g"], {
        encoding: "utf8",
      }).trim();
      return require_(path.join(globalRoot, moduleName));
    } catch (globalErr) {
      throw new Error(
        `Cannot load ${moduleName}. Install it globally (npm i -g), or set ` +
        `TURBO_PI_MODULE_PATH to its dist/index.js. ` +
        `Tried plain resolution (${localErr.message.split("\n")[0]}) and ` +
        `the global npm root (${globalErr.message.split("\n")[0]}).`,
      );
    }
  }
}

// OpenAI-shaped prompt -> Pi Context. The Python side owns precedence and
// caps; this is a pure representation mapping.

function toPiMessages(openaiMessages) {
  const messages = [];
  for (const m of openaiMessages || []) {
    if (m.role === "user") {
      messages.push({ role: "user", content: m.content ?? "", timestamp: Date.now() });
    } else if (m.role === "assistant") {
      const content = [];
      if (m.content) content.push({ type: "text", text: m.content });
      for (const tc of m.tool_calls || []) {
        let arguments_ = {};
        try {
          arguments_ = JSON.parse(tc.function?.arguments || "{}");
        } catch {
          arguments_ = {};
        }
        content.push({
          type: "toolCall",
          id: tc.id || "",
          name: tc.function?.name || "",
          arguments: arguments_,
        });
      }
      messages.push({
        role: "assistant",
        content,
        api: "openai-completions",
        provider: "replay",
        model: "replay",
        usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        stopReason: "stop",
        timestamp: Date.now(),
      });
    } else if (m.role === "tool") {
      messages.push({
        role: "toolResult",
        toolCallId: m.tool_call_id || "",
        toolName: m.name || "",
        content: [{ type: "text", text: typeof m.content === "string" ? m.content : JSON.stringify(m.content ?? "") }],
      });
    }
    // system/developer messages are handled as systemPrompt elsewhere.
  }
  return messages;
}

function toPiContext(request) {
  let systemPrompt;
  const chat = [];
  for (const m of request.messages || []) {
    if ((m.role === "system" || m.role === "developer") && !systemPrompt) {
      systemPrompt = m.content;
    } else {
      chat.push(m);
    }
  }
  const context = { messages: toPiMessages(chat) };
  if (systemPrompt !== undefined) context.systemPrompt = systemPrompt;
  if (request.tools?.length) {
    context.tools = request.tools.map((t) => ({
      name: t.function?.name || t.name || "",
      description: t.function?.description || t.description || "",
      parameters: t.function?.parameters || t.parameters || { type: "object", properties: {} },
    }));
  }
  return context;
}

// Pi AssistantMessage -> canonical result consumed by the Python seam.

function toCanonicalResult(message) {
  let text = "";
  let thinking = "";
  const toolCalls = [];
  for (const block of message.content || []) {
    if (block.type === "text") text += block.text;
    else if (block.type === "thinking") thinking += block.thinking || "";
    else if (block.type === "toolCall") {
      toolCalls.push({
        id: block.id,
        name: block.name,
        arguments: block.arguments ?? {},
      });
    }
  }
  const finishMap = {
    stop: "stop",
    length: "length",
    toolUse: "tool_use",
    deferred: "other",
  };
  return {
    output: {
      text,
      thinking: thinking || null,
      tool_calls: toolCalls,
      finish_reason: finishMap[message.stopReason] ?? "other",
    },
    usage: message.usage
      ? {
          input_tokens: message.usage.input ?? 0,
          output_tokens: message.usage.output ?? 0,
          reasoning_tokens: message.usage.reasoning ?? null,
        }
      : null,
    response_id: message.responseId ?? null,
    // Identity metadata only; never credentials.
    meta: { provider: message.provider, model: message.model },
  };
}

function classifyError(err) {
  if (err && err.abort === true) return { kind: "cancelled", retryable: false };
  const name = err?.name || "";
  const message = String(err?.message || err || "unknown error");
  if (/rate limit|429|quota/i.test(message)) return { kind: "rate_limited", retryable: true };
  if (/unauthorized|authentication|401|403/i.test(message)) return { kind: "authentication", retryable: false };
  if (/timeout|ETIMEDOUT/i.test(name + message)) return { kind: "timeout", retryable: true };
  if (/ECONNREFUSED|ENOTFOUND|fetch failed|network/i.test(message)) return { kind: "unavailable", retryable: true };
  return { kind: "internal", retryable: false };
}

class Companion {
  constructor(runtime) {
    this.runtime = runtime;
    this.inflight = new Map(); // id -> { controller, started }
  }

  emit(payload) {
    process.stdout.write(JSON.stringify({ v: PROTOCOL_VERSION, ...payload }) + "\n");
  }

  emitError(id, kind, retryable, message) {
    this.emit({ id, event: "error", error: { kind, retryable, message } });
  }

  handleCancel(msg) {
    const entry = this.inflight.get(msg.id);
    if (entry) entry.controller.abort();
  }

  async handle(msg) {
    if (msg.v !== PROTOCOL_VERSION) {
      this.emitError(
        msg.id ?? "?", "invalid_request", false,
        `unsupported protocol version ${JSON.stringify(msg.v)}; expected ${PROTOCOL_VERSION}`,
      );
      return;
    }
    if (msg.op === "ping") {
      this.emit({ id: msg.id, event: "done", result: { pong: true } });
      return;
    }
    if (msg.op === "cancel") {
      this.handleCancel(msg);
      return;
    }
    if (msg.op !== "complete" && msg.op !== "stream") {
      this.emitError(msg.id, "invalid_request", false, `unknown op ${JSON.stringify(msg.op)}`);
      return;
    }

    const controller = new AbortController();
    this.inflight.set(msg.id, { controller, started: Date.now() });
    try {
      const model = this.runtime.getModel(msg.provider, msg.model);
      if (!model) {
        this.emitError(msg.id, "invalid_request", false, `unknown model ${msg.provider}/${msg.model}`);
        return;
      }
      const context = toPiContext(msg.request || {});
      const options = { ...msg.options, signal: controller.signal };
      if (msg.op === "complete") {
        const message = await this.runtime.completeSimple(model, context, options);
        this.emit({ id: msg.id, event: "done", result: toCanonicalResult(message) });
      } else {
        await this.doStream(msg.id, model, context, options);
      }
    } catch (err) {
      if (controller.signal.aborted) {
        this.emitError(msg.id, "cancelled", false, "request cancelled");
      } else {
        const { kind, retryable } = classifyError(err);
        this.emitError(msg.id, kind, retryable, String(err?.message || err));
      }
    } finally {
      this.inflight.delete(msg.id);
    }
  }

  async doStream(id, model, context, options) {
    const stream = this.runtime.streamSimple(model, context, options);
    for await (const ev of stream) {
      switch (ev.type) {
        case "text_delta":
          this.emit({ id, event: "delta", delta: { type: "text", text: ev.delta } });
          break;
        case "thinking_delta":
          this.emit({ id, event: "delta", delta: { type: "thinking", text: ev.delta } });
          break;
        case "toolcall_start": {
          const block = ev.partial?.content?.[ev.contentIndex];
          this.emit({
            id, event: "delta",
            delta: { type: "tool_call_started", id: block?.id || "", name: block?.name || "" },
          });
          break;
        }
        case "toolcall_delta":
          this.emit({ id, event: "delta", delta: { type: "tool_call_arguments", json_fragment: ev.delta } });
          break;
        case "toolcall_end":
          break; // final tool call arrives in the done result
        case "done":
          this.emit({ id, event: "done", result: toCanonicalResult(ev.message) });
          return;
        case "error": {
          if (options.signal?.aborted) {
            this.emitError(id, "cancelled", false, "request cancelled");
          } else {
            const { kind, retryable } = classifyError(
              new Error(ev.error?.errorMessage || "model stream error"),
            );
            this.emitError(id, kind, retryable, ev.error?.errorMessage || "model stream error");
          }
          return;
        }
        default:
          break; // start/text_start/text_end/thinking_* are informational
      }
    }
    this.emitError(id, "internal", false, "stream ended without a done event");
  }
}

async function main() {
  const runtime = await loadRuntime();
  const companion = new Companion(runtime);

  const rl = readline.createInterface({ input: process.stdin });
  rl.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let msg;
    try {
      msg = JSON.parse(trimmed);
    } catch (err) {
      companion.emitError("?", "invalid_request", false, `undecodable line: ${err.message}`);
      return;
    }
    companion.handle(msg).catch((err) => {
      log(`handler failure for ${msg.id}: ${err?.stack || err}`);
      companion.emitError(msg.id ?? "?", "internal", false, "companion handler failure");
    });
  });
  rl.on("close", async () => {
    // Drain in-flight calls briefly before exiting so a deliberate Python
    // shutdown does not strand responses that are milliseconds away.
    const deadline = Date.now() + 5000;
    while (companion.inflight.size > 0 && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    log("stdin closed; exiting");
    process.exit(0);
  });
}

main().catch((err) => {
  log(`fatal: ${err?.stack || err}`);
  process.exit(1);
});
