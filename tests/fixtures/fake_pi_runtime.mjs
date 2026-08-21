// Fake Pi ModelRuntime for companion contract tests.
// Exports createRuntime() returning the subset of ModelRuntime the
// companion uses: getModel(provider, modelId), completeSimple(),
// streamSimple(). Scriptable per-test via TURBO_PI_FAKE_SCRIPT (JSON).

function makeMessage(overrides = {}) {
  return {
    role: "assistant",
    content: [{ type: "text", text: "ok" }],
    api: "openai-completions",
    provider: "fake",
    model: "fake-model",
    usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0 },
    stopReason: "stop",
    timestamp: 0,
    ...overrides,
  };
}

export async function createRuntime() {
  let script = {
    complete: makeMessage(),
    streamEvents: [
      { type: "start", partial: makeMessage({ content: [] }) },
      { type: "text_delta", contentIndex: 0, delta: "he", partial: makeMessage() },
      { type: "text_delta", contentIndex: 0, delta: "llo", partial: makeMessage() },
      {
        type: "done", reason: "stop",
        message: makeMessage({ content: [{ type: "text", text: "hello" }] }),
      },
    ],
    // Deliberately planted secret: contract tests assert it never reaches stdout.
    leak: "sk-fake-oauth-token-1234567890",
    delayMs: 0,
    failModel: null,
  };
  if (process.env.TURBO_PI_FAKE_SCRIPT) {
    script = { ...script, ...JSON.parse(process.env.TURBO_PI_FAKE_SCRIPT) };
  }

  const runtime = {
    leakedSecret: script.leak,
    calls: [],
    getModel(providerId, modelId) {
      if (script.failModel === `${providerId}/${modelId}`) return undefined;
      return { provider: providerId, id: modelId };
    },
    async completeSimple(model, context, options) {
      runtime.calls.push({ op: "complete", model, context, options });
      if (script.delayMs) await new Promise((r) => setTimeout(r, script.delayMs));
      if (script.completeError) throw Object.assign(new Error(script.completeError.message), {
        kind: script.completeError.kind,
      });
      if (script.echoContext) {
        return makeMessage({
          content: [{
            type: "text",
            text: JSON.stringify({
              systemPrompt: context.systemPrompt ?? null,
              firstRole: context.messages[0]?.role ?? null,
              toolNames: (context.tools || []).map((t) => t.name),
            }),
          }],
        });
      }
      return typeof script.complete === "function"
        ? script.complete(model, context, options)
        : makeMessage(script.complete || {});
    },
    streamSimple(model, context, options) {
      runtime.calls.push({ op: "stream", model, context, options });
      const events = typeof script.streamEvents === "function"
        ? script.streamEvents(model, context, options)
        : script.streamEvents;
      return {
        [Symbol.asyncIterator]() {
          let i = 0;
          return {
            async next() {
              if (script.delayMs && i === 0) {
                await new Promise((r) => setTimeout(r, script.delayMs));
              }
              if (i < events.length) return { value: events[i++], done: false };
              return { done: true };
            },
            async return() {
              i = events.length;
              return { done: true };
            },
          };
        },
      };
    },
  };
  return runtime;
}
