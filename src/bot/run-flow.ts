import { mkdir, realpath } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { log } from '../core/logger';
import type { AgentCapability } from '../agent/capability';
import type { AgentEvent } from '../agent/types';
import type { ProfileConfig } from '../config/profile-schema';
import type { AccessDecision } from '../policy/access';
import {
  evaluateRunPolicy,
  type AgentAttachment,
  type RunPolicyAllow,
  type RunPolicyReject,
  type ScopeContext,
} from '../policy/run-policy';
import {
  resolveWorkingDirectory,
  type WorkingDirectoryRejectReason,
  type WorkingDirectoryResolveResult,
} from '../policy/workspace';
import type { RunExecution, RunExecutor } from '../runtime/run-executor';
import { RunRejected, type RunRejectedCode } from '../runtime/errors';
import type { SessionCatalog } from '../session/catalog';
import type { SessionStore } from '../session/store';
import type { WorkspaceStore } from '../workspace/store';

export interface StartRunFlowInput {
  scopeId: string;
  scope: ScopeContext;
  prompt: string;
  attachments: AgentAttachment[];
  access: AccessDecision;
  capability: AgentCapability;
  profileConfig: ProfileConfig;
  sessions: SessionStore;
  sessionCatalog?: SessionCatalog;
  workspaces: WorkspaceStore;
  executor: RunExecutor;
  now: number;
  stopGraceMs?: number;
  observability?: {
    profile: string;
    agent: string;
    source: string;
    stage: string;
  };
  /**
   * Default workspace directory root (e.g. ~/.lark-channel-workspaces/pi/default).
   * When neither a per-scope workspace nor a profile-level default is set,
   * a per-scope subdirectory (<defaultWorkspaceDir>/<scopeId>) is created
   * automatically so that different users/chats get isolated working directories.
   */
  defaultWorkspaceDir?: string;
}

export type RunFlowRejectCode =
  | WorkingDirectoryRejectReason
  | RunPolicyReject['rejectReason']['code']
  | RunRejectedCode;

export type StartRunFlowResult =
  | {
      ok: true;
      execution: RunExecution;
      policy: RunPolicyAllow;
      cwdRealpath: string;
      resumeFrom?: string;
    }
  | {
      ok: false;
      rejectReason: {
        code: RunFlowRejectCode;
        userVisible: string;
      };
      workspace?: WorkingDirectoryResolveResult;
    };

export interface RecordRunSessionEventInput {
  scopeId: string;
  sessions: SessionStore;
  sessionCatalog?: SessionCatalog;
  capability: AgentCapability;
  policy: RunPolicyAllow;
  event: AgentEvent;
}

/**
 * Resolve the working directory for a scope.
 *
 * Priority:
 * 1. Per-scope workspace explicitly set via `workspaces.cwdFor(scopeId)`
 *    (set by the /workspace command)
 * 2. Auto-generated per-scope directory as a sibling of `defaultWorkspaceDir`
 *    (parent of defaultWorkspaceDir / <scopeId>). This is the default behavior
 *    for multi-user isolation: each chat/topic gets its own working directory
 *    on first use, persisted to the workspace store for subsequent runs.
 *    e.g. defaultWorkspaceDir=.../pi/default → scopeDir=.../pi/<scopeId>
 * 3. Profile-level `workspaces.default` as a last-resort fallback
 *    (set in config.json; shared by all scopes when no per-scope dir exists)
 */
async function resolveScopeCwd(input: StartRunFlowInput): Promise<string> {
  // 1. Check for explicitly set per-scope workspace (/workspace command)
  const explicitCwd = input.workspaces.cwdFor(input.scopeId);
  if (explicitCwd) return explicitCwd;

  // 2. Auto-generate per-scope directory if defaultWorkspaceDir is provided
  if (input.defaultWorkspaceDir) {
    // Sanitize scopeId for use as directory name (replace ':' with '_')
    const safeScopeId = input.scopeId.replace(/:/g, '_');
    // Place per-scope dirs as siblings of the default workspace dir:
    //   defaultWorkspaceDir = .../pi/default  →  scopeDir = .../pi/<scopeId>
    const scopeDir = join(dirname(input.defaultWorkspaceDir), safeScopeId);
    try {
      await mkdir(scopeDir, { recursive: true, mode: 0o700 });
      const resolved = await realpath(scopeDir);
      // Persist so subsequent runs reuse the same directory
      input.workspaces.setCwd(input.scopeId, resolved);
      return resolved;
    } catch (err) {
      // If auto-creation fails, fall through to profile default
      log.warn('workspace', 'auto-scope-dir-failed', {
        scopeId: input.scopeId,
        scopeDir,
        err: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // 3. Fallback to profile-level default workspace (shared directory)
  if (input.profileConfig.workspaces.default) {
    const defaultDir = input.profileConfig.workspaces.default;
    try {
      await mkdir(defaultDir, { recursive: true, mode: 0o700 });
      return await realpath(defaultDir);
    } catch (err) {
      log.warn('workspace', 'default-dir-creation-failed', {
        defaultDir,
        err: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return '';
}

export async function startRunFlow(input: StartRunFlowInput): Promise<StartRunFlowResult> {
  const requestedCwd = await resolveScopeCwd(input);
  const workspace = await resolveWorkingDirectory(requestedCwd);
  if (!workspace.ok) {
    return {
      ok: false,
      rejectReason: {
        code: workspace.reason,
        userVisible: workspace.userVisible,
      },
      workspace,
    };
  }

  const policy = evaluateRunPolicy({
    scope: input.scope,
    attachments: input.attachments,
    prompt: input.prompt,
    requestedCwd,
    cwdRealpath: workspace.cwdRealpath,
    access: input.access,
    capability: input.capability,
    profileConfig: input.profileConfig,
    now: input.now,
    codexHome: input.profileConfig.codex?.codexHome,
    inheritCodexHome: input.profileConfig.codex?.inheritCodexHome,
  });
  if (!policy.ok) {
    return {
      ok: false,
      rejectReason: policy.rejectReason,
      workspace,
    };
  }

  let resumeFrom: string | undefined;
  let sessionId: string | undefined;
  let threadId: string | undefined;
  if (input.sessionCatalog) {
    const catalogEntry = input.sessionCatalog.activeFor({
      scopeId: input.scopeId,
      agentId: input.capability.agentId,
      cwdRealpath: workspace.cwdRealpath,
      policyFingerprint: policy.policyFingerprint,
    });
    if (catalogEntry?.agentId === 'claude') {
      sessionId = catalogEntry.sessionId;
      resumeFrom = sessionId;
    } else if (catalogEntry?.agentId === 'codex') {
      threadId = catalogEntry.threadId;
      resumeFrom = threadId;
    }
  }
  if (!resumeFrom && input.capability.agentId === 'claude') {
    resumeFrom = input.sessions.resumeFor(input.scopeId, workspace.cwdRealpath);
    sessionId = resumeFrom;
    const stale = input.sessions.getRaw(input.scopeId);
    if (!resumeFrom && stale?.cwd && stale.cwd !== workspace.cwdRealpath) {
      input.sessions.clear(input.scopeId);
    }
  }

  // For pi agent: derive session-dir from the resolved cwd so that sessions
  // are persisted per-scope and survive bridge restarts.
  const sessionDir = input.capability.agentId === 'pi'
    ? `${workspace.cwdRealpath}/.pi-sessions`
    : undefined;

  let execution: RunExecution;
  try {
    execution = await input.executor.submit({
      scopeId: input.scopeId,
      policy,
      sessionId,
      threadId,
      sessionDir,
      images:
        input.capability.agentId === 'codex'
          ? policy.attachments
              .filter((attachment) => attachment.kind === 'image' && attachment.decision === 'accepted')
              .map((attachment) => attachment.path)
              .filter((path): path is string => Boolean(path))
          : undefined,
      stopGraceMs: input.stopGraceMs,
      observability: input.observability,
    });
  } catch (err) {
    if (err instanceof RunRejected) {
      return {
        ok: false,
        rejectReason: {
          code: err.code,
          userVisible:
            err.code === 'reconnect-in-progress'
              ? '当前 bot 正在重连，稍后会继续处理新消息。'
              : err.code === 'run-already-active'
                ? '当前会话已有运行在执行，请稍后再试或先停止当前运行。'
              : '当前无法发起运行，请稍后重试。',
        },
        workspace,
      };
    }
    throw err;
  }

  return {
    ok: true,
    execution,
    policy,
    cwdRealpath: workspace.cwdRealpath,
    ...(resumeFrom ? { resumeFrom } : {}),
  };
}

export function recordRunSessionEvent(input: RecordRunSessionEventInput): void {
  if (input.event.type !== 'system') return;
  if (input.capability.agentId === 'claude' && input.event.sessionId) {
    const cwdRealpath = input.event.cwd ?? input.policy.cwdRealpath;
    input.sessions.set(input.scopeId, input.event.sessionId, cwdRealpath);
    input.sessionCatalog?.upsertActive({
      scopeId: input.scopeId,
      agentId: 'claude',
      cwdRealpath,
      policyFingerprint: input.policy.policyFingerprint,
      sessionId: input.event.sessionId,
    });
    return;
  }
  if (input.capability.agentId === 'codex' && input.event.threadId) {
    input.sessionCatalog?.upsertActive({
      scopeId: input.scopeId,
      agentId: 'codex',
      cwdRealpath: input.policy.cwdRealpath,
      policyFingerprint: input.policy.policyFingerprint,
      threadId: input.event.threadId,
    });
  }
}
