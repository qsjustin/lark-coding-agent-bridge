import { createInterface } from 'node:readline';
import type { Readable, Writable } from 'node:stream';
import { log } from '../../core/logger';
import { mergeProcessEnv, spawnProcess, type SpawnedProcessByStdio } from '../../platform/spawn';
import { buildBridgeSystemPrompt } from '../bridge-system-prompt';
import { buildLarkChannelEnv, type LarkChannelEnvContext } from '../lark-channel-env';
import { checkAgentAvailability, type AgentAvailability } from '../preflight';
import type {
  AgentAdapter,
  AgentBotIdentity,
  AgentEvent,
  AgentRun,
  AgentRunOptions,
} from '../types';

export interface PiAgentAdapterOptions {
  /** Path to the pi binary. Defaults to "pi". */
  binary?: string;
  /** Extra arguments to pass to pi. */
  extraArgs?: string[];
  larkChannel?: LarkChannelEnvContext;
}

type PiChild = SpawnedProcessByStdio<Writable, Readable, Readable>;

/**
 * PiAgentAdapter bridges pi (OWL) coding agent into lark-channel-bridge.
 *
 * It spawns `pi --mode rpc --no-session` as a child process and communicates
 * via the JSON-RPC protocol over stdin/stdout. Pi's RPC events are translated
 * into the bridge's AgentEvent format.
 */
export class PiAgentAdapter implements AgentAdapter {
  readonly id = 'pi';
  readonly displayName = 'pi (OWL)';

  private readonly binary: string;
  private readonly extraArgs: string[];
  private readonly larkChannel: LarkChannelEnvContext | undefined;
  private botIdentity: AgentBotIdentity | undefined;

  constructor(opts: PiAgentAdapterOptions = {}) {
    this.binary = opts.binary ?? 'pi';
    this.extraArgs = opts.extraArgs ?? [];
    this.larkChannel = opts.larkChannel;
  }

  setBotIdentity(identity: AgentBotIdentity): void {
    this.botIdentity = identity;
  }

  async isAvailable(): Promise<boolean> {
    return (await this.checkAvailability()).ok;
  }

  async checkAvailability(): Promise<AgentAvailability> {
    return checkAgentAvailability({
      agentId: 'pi',
      agentName: 'pi (OWL)',
      command: this.binary,
      binaryPath: this.binary,
    });
  }

  run(opts: AgentRunOptions): AgentRun {
    if (!opts.cwd) {
      throw new Error('cwd is required for PiAgentAdapter.run');
    }

    const args = [
      '--mode', 'rpc',
      '--no-session',
      ...this.extraArgs,
    ];

    const envOverrides = buildLarkChannelEnv(this.larkChannel);

    const child = spawnProcess(this.binary, args, {
      cwd: opts.cwd,
      env: mergeProcessEnv(process.env, envOverrides),
      stdio: ['pipe', 'pipe', 'pipe'],
    }) as PiChild;

    log.info('agent', 'spawn', {
      pid: child.pid ?? null,
      cwd: opts.cwd ?? process.cwd(),
      promptChars: opts.prompt.length,
    });

    // Set up stderr logging
    const stderrChunks: Buffer[] = [];
    let runtimeError: Error | null = null;
    let stderrBuffer = '';
    child.stderr.on('data', (chunk: Buffer) => {
      stderrChunks.push(chunk);
      stderrBuffer += chunk.toString('utf8');
      let nl = stderrBuffer.indexOf('\n');
      while (nl !== -1) {
        const line = stderrBuffer.slice(0, nl);
        stderrBuffer = stderrBuffer.slice(nl + 1);
        if (line.trim()) log.warn('agent', 'stderr', { line });
        nl = stderrBuffer.indexOf('\n');
      }
    });

    child.on('error', (err) => {
      runtimeError = err;
    });
    child.on('exit', (code, signal) => {
      log.info('agent', 'exit', { pid: child.pid ?? null, code, signal });
    });
    child.stdin.on('error', (err) => {
      log.warn('agent', 'stdin-error', { message: err.message });
    });

    // Build the full prompt with bridge context
    const systemPrompt = buildBridgeSystemPrompt(this.botIdentity);
    const fullPrompt = `${systemPrompt}\n\n## user_message\n\n${opts.prompt}`;

    // Send the prompt via RPC protocol
    const promptCmd = JSON.stringify({
      type: 'prompt',
      message: fullPrompt,
    }) + '\n';
    child.stdin.write(promptCmd, 'utf8');

    const stopGraceMs = opts.stopGraceMs ?? 5000;

    return {
      runId: opts.runId,
      events: createEventStream(child, stderrChunks, () => runtimeError),
      async stop() {
        if (child.exitCode !== null || child.signalCode !== null) return;
        // Try graceful abort via RPC first
        try {
          child.stdin.write(JSON.stringify({ type: 'abort' }) + '\n');
        } catch {
          // stdin may already be closed
        }
        // Wait a bit, then SIGTERM
        await new Promise<void>((resolve) => {
          const timer = setTimeout(() => {
            if (child.exitCode === null && child.signalCode === null) {
              log.warn('agent', 'stop-sigterm', {
                pid: child.pid ?? null,
                graceMs: stopGraceMs,
                reason: 'grace-period-expired',
              });
              child.kill('SIGTERM');
            }
            resolve();
          }, stopGraceMs);
          child.once('exit', () => {
            clearTimeout(timer);
            resolve();
          });
        });
      },
      waitForExit(timeoutMs: number): Promise<boolean> {
        if (child.exitCode !== null || child.signalCode !== null) {
          return Promise.resolve(true);
        }
        return new Promise<boolean>((resolve) => {
          const onExit = (): void => {
            clearTimeout(timer);
            resolve(true);
          };
          const timer = setTimeout(() => {
            child.removeListener('exit', onExit);
            resolve(false);
          }, timeoutMs);
          child.once('exit', onExit);
        });
      },
    };
  }
}

async function* createEventStream(
  child: PiChild,
  stderrChunks: Buffer[],
  getError: () => Error | null,
): AsyncGenerator<AgentEvent> {
  if (!child.pid) {
    const err = getError();
    yield {
      type: 'error',
      message: err ? `failed to spawn pi: ${err.message}` : 'spawn returned no pid',
      terminationReason: 'failed',
    };
    return;
  }

  const rl = createInterface({ input: child.stdout, crlfDelay: Infinity });
  let sawStdout = false;
  let silentExitTimer: ReturnType<typeof setTimeout> | undefined;
  const closeSilentStdout = (): void => {
    silentExitTimer = setTimeout(() => {
      if (!sawStdout && !child.stdout.readableEnded) child.stdout.destroy();
    }, 50);
  };
  child.once('exit', closeSilentStdout);

  // Track tool calls for tool_result correlation
  const toolCallNames = new Map<string, string>();

  try {
    for await (const line of rl) {
      sawStdout = true;
      const trimmed = line.trim();
      if (!trimmed) continue;

      let parsed: unknown;
      try {
        parsed = JSON.parse(trimmed);
      } catch {
        continue;
      }

      const event = parsed as Record<string, unknown>;
      const eventType = event.type as string;

      // Skip RPC responses (type: "response")
      if (eventType === 'response') continue;

      // Translate pi RPC events to bridge AgentEvents
      const events = translatePiEvent(event, toolCallNames);
      for (const evt of events) {
        yield evt;
        // Stop on terminal events
        if (evt.type === 'done' || evt.type === 'error') return;
      }
    }
  } finally {
    if (silentExitTimer) clearTimeout(silentExitTimer);
    child.removeListener('exit', closeSilentStdout);
    rl.close();
  }

  // Handle process exit
  const exitCode = await new Promise<number | null>((resolve) => {
    if (child.exitCode !== null || child.signalCode !== null) {
      resolve(child.exitCode);
    } else {
      child.once('exit', (code) => resolve(code));
    }
  });

  const runtimeError = getError();
  if (exitCode !== 0 && exitCode !== null) {
    const stderr = Buffer.concat(stderrChunks).toString('utf8').trim();
    const detail = stderr ? `: ${stderr.slice(0, 500)}` : '';
    yield {
      type: 'error',
      message: `pi exited with code ${exitCode}${detail}`,
      terminationReason: 'failed',
    };
  } else if (runtimeError) {
    yield {
      type: 'error',
      message: `pi runtime error: ${runtimeError.message}`,
      terminationReason: 'failed',
    };
  } else {
    // Normal exit without explicit done event
    yield {
      type: 'done',
      terminationReason: 'normal',
    };
  }
}

/**
 * Translate a pi RPC event into bridge AgentEvent(s).
 */
function translatePiEvent(
  event: Record<string, unknown>,
  toolCallNames: Map<string, string>,
): AgentEvent[] {
  const eventType = event.type as string;

  switch (eventType) {
    case 'agent_start':
      return [{ type: 'system' }];

    case 'message_update': {
      const assistantEvent = event.assistantMessageEvent as Record<string, unknown> | undefined;
      if (!assistantEvent) return [];
      const aeType = assistantEvent.type as string;

      switch (aeType) {
        case 'text_delta':
          return [{ type: 'text', delta: (assistantEvent.delta as string) ?? '' }];
        case 'thinking_delta':
          return [{ type: 'thinking', delta: (assistantEvent.delta as string) ?? '' }];
        case 'toolcall_start': {
          // Record the tool call name for later correlation, but don't emit
          // tool_use yet — wait for tool_execution_start which has the full
          // args and is the actual execution boundary.
          const toolCall = assistantEvent.partial as Record<string, unknown> | undefined;
          const id = (toolCall?.id as string) ?? `tool_${Date.now()}`;
          const name = (toolCall?.name as string) ?? 'unknown';
          toolCallNames.set(id, name);
          return [];
        }
        case 'toolcall_delta':
        case 'toolcall_end':
          // Skip incremental tool call events — the bridge card reducer
          // would create duplicate ToolEntries for the same id.
          // Full args arrive via tool_execution_start instead.
          return [];
        default:
          return [];
      }
    }

    case 'tool_execution_start': {
      const toolCallId = (event.toolCallId as string) ?? '';
      const toolName = (event.toolName as string) ?? 'unknown';
      toolCallNames.set(toolCallId, toolName);
      return [{ type: 'tool_use', id: toolCallId, name: toolName, input: event.args ?? {} }];
    }

    case 'tool_execution_update':
      // Bridge doesn't handle streaming tool output updates;
      // wait for tool_execution_end with the final result.
      return [];

    case 'tool_execution_end': {
      const toolCallId = (event.toolCallId as string) ?? '';
      const result = event.result as Record<string, unknown> | undefined;
      const content = result?.content as Array<{ type: string; text: string }> | undefined;
      const output = content?.map(c => c.text).join('\n') ?? '';
      const isError = (event.isError as boolean) ?? false;
      return [{ type: 'tool_result', id: toolCallId, output, isError }];
    }

    case 'turn_end': {
      const message = event.message as Record<string, unknown> | undefined;
      const usage = message?.usage as Record<string, unknown> | undefined;
      if (usage) {
        // Sanitize usage values: cost may be null/undefined/string from some agents,
        // but bridge expects a number or undefined (never null).
        const rawCost = usage.cost;
        const costUsd = (typeof rawCost === 'number' && Number.isFinite(rawCost)) ? rawCost : undefined;
        const inputTokens = typeof usage.input === 'number' ? usage.input : undefined;
        const outputTokens = typeof usage.output === 'number' ? usage.output : undefined;
        if (costUsd !== undefined || inputTokens !== undefined || outputTokens !== undefined) {
          return [{
            type: 'usage',
            inputTokens,
            outputTokens,
            costUsd,
          }];
        }
      }
      return [];
    }

    case 'agent_end':
      return [{ type: 'done', terminationReason: 'normal' }];

    case 'compaction_start':
    case 'compaction_end':
    case 'queue_update':
    case 'auto_retry_start':
    case 'auto_retry_end':
      // Internal events, not surfaced to bridge
      return [];

    case 'extension_error':
      return [{ type: 'error', message: (event.error as string) ?? 'extension error', terminationReason: 'failed' }];

    default:
      return [];
  }
}
