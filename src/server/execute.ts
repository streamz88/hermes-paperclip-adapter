/**
 * Server-side execution logic for the Hermes Agent adapter.
 *
 * Spawns `hermes chat -q "..."` as a child process, streams output,
 * and returns structured results to Paperclip.
 */

import type {
  AdapterExecutionContext,
  AdapterExecutionResult,
  UsageSummary,
} from "@paperclipai/adapter-utils";

import {
  runChildProcess,
  buildPaperclipEnv,
  renderTemplate,
  ensureAbsoluteDirectory,
} from "@paperclipai/adapter-utils/server-utils";

import {
  HERMES_CLI,
  DEFAULT_TIMEOUT_SEC,
  DEFAULT_GRACE_SEC,
  DEFAULT_MAX_ITERATIONS,
  DEFAULT_MODEL,
  SESSION_ID_REGEX,
  TOKEN_USAGE_REGEX,
  COST_REGEX,
} from "../shared/constants.js";

// ---------------------------------------------------------------------------
// Config helpers (wrapping adapter-utils versions with optional semantics)
// ---------------------------------------------------------------------------

function cfgString(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}
function cfgNumber(v: unknown): number | undefined {
  return typeof v === "number" ? v : undefined;
}
function cfgBoolean(v: unknown): boolean | undefined {
  return typeof v === "boolean" ? v : undefined;
}
function cfgStringArray(v: unknown): string[] | undefined {
  return Array.isArray(v) && v.every((i) => typeof i === "string")
    ? (v as string[])
    : undefined;
}

// ---------------------------------------------------------------------------
// Wake-up prompt builder
// ---------------------------------------------------------------------------

const DEFAULT_PROMPT_TEMPLATE = `You are an AI agent working as an employee in a company managed by Paperclip.

Your agent ID is {{agentId}} and you work for company {{companyId}}.
Your agent name is "{{agentName}}".

{{#taskId}}
You have been assigned a task:
  Issue ID: {{taskId}}
  Title: {{taskTitle}}

Instructions:
{{taskBody}}

When you are done, report your results clearly. If you made code changes,
summarize what you changed and why.
{{/taskId}}

{{#noTask}}
You have been woken by a heartbeat. Check for any pending work:

1. Use your tools to check the current state of your project
2. Look for any issues or improvements you can make
3. If there is nothing to do, report that briefly

API URL: {{paperclipApiUrl}}
{{/noTask}}`;

function buildPrompt(
  ctx: AdapterExecutionContext,
  config: Record<string, unknown>,
): string {
  const template = cfgString(config.promptTemplate) || DEFAULT_PROMPT_TEMPLATE;

  const taskId = cfgString(ctx.config?.taskId);
  const taskTitle = cfgString(ctx.config?.taskTitle) || "";
  const taskBody = cfgString(ctx.config?.taskBody) || "";
  const agentName = ctx.agent?.name || "Hermes Agent";
  const companyName = cfgString(ctx.config?.companyName) || "";
  const projectName = cfgString(ctx.config?.projectName) || "";
  const paperclipApiUrl =
    cfgString(config.paperclipApiUrl) ||
    process.env.PAPERCLIP_API_URL ||
    "http://localhost:3100/api";

  const vars: Record<string, unknown> = {
    agentId: ctx.agent?.id || "",
    agentName,
    companyId: ctx.agent?.companyId || "",
    companyName,
    runId: ctx.runId || "",
    taskId: taskId || "",
    taskTitle,
    taskBody,
    projectName,
    paperclipApiUrl,
  };

  // Handle conditional sections: {{#key}}...{{/key}}
  let rendered = template;

  // {{#taskId}}...{{/taskId}} — include if task is assigned
  rendered = rendered.replace(
    /\{\{#taskId\}\}([\s\S]*?)\{\{\/taskId\}\}/g,
    taskId ? "$1" : "",
  );

  // {{#noTask}}...{{/noTask}} — include if no task
  rendered = rendered.replace(
    /\{\{#noTask\}\}([\s\S]*?)\{\{\/noTask\}\}/g,
    taskId ? "" : "$1",
  );

  // Replace remaining {{variable}} placeholders
  return renderTemplate(rendered, vars);
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

interface ParsedOutput {
  sessionId?: string;
  usage?: UsageSummary;
  costUsd?: number;
  errorMessage?: string;
}

function parseHermesOutput(stdout: string, stderr: string): ParsedOutput {
  const combined = stdout + "\n" + stderr;
  const result: ParsedOutput = {};

  // Extract session ID (Hermes prints it on exit)
  const sessionMatch = combined.match(SESSION_ID_REGEX);
  if (sessionMatch?.[1]) {
    result.sessionId = sessionMatch[1];
  }

  // Extract token usage
  const usageMatch = combined.match(TOKEN_USAGE_REGEX);
  if (usageMatch) {
    result.usage = {
      inputTokens: parseInt(usageMatch[1], 10) || 0,
      outputTokens: parseInt(usageMatch[2], 10) || 0,
    };
  }

  // Extract cost
  const costMatch = combined.match(COST_REGEX);
  if (costMatch?.[1]) {
    result.costUsd = parseFloat(costMatch[1]);
  }

  // Check for error patterns in stderr
  if (stderr.trim()) {
    const errorLines = stderr
      .split("\n")
      .filter((line) => /error|exception|traceback|failed/i.test(line));
    if (errorLines.length > 0) {
      result.errorMessage = errorLines.slice(0, 5).join("\n");
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Main execute
// ---------------------------------------------------------------------------

export async function execute(
  ctx: AdapterExecutionContext,
): Promise<AdapterExecutionResult> {
  const config = (ctx.agent?.adapterConfig ?? {}) as Record<string, unknown>;

  // ── Resolve configuration ──────────────────────────────────────────────
  const hermesCmd = cfgString(config.hermesCommand) || HERMES_CLI;
  const model = cfgString(config.model) || DEFAULT_MODEL;
  const provider = cfgString(config.provider);
  const maxIterations =
    cfgNumber(config.maxIterations) || DEFAULT_MAX_ITERATIONS;
  const timeoutSec = cfgNumber(config.timeoutSec) || DEFAULT_TIMEOUT_SEC;
  const graceSec = cfgNumber(config.graceSec) || DEFAULT_GRACE_SEC;
  const enabledToolsets = cfgStringArray(config.enabledToolsets);
  const disabledToolsets = cfgStringArray(config.disabledToolsets);
  const contextFiles = cfgStringArray(config.contextFiles);
  const extraArgs = cfgStringArray(config.extraArgs);
  const persistSession = cfgBoolean(config.persistSession) !== false; // default true
  const skipMemory = cfgBoolean(config.skipMemory) === true;
  const worktreeMode = cfgBoolean(config.worktreeMode) === true;
  const quietMode = cfgBoolean(config.quietMode) !== false; // default true

  // ── Build prompt ───────────────────────────────────────────────────────
  const prompt = buildPrompt(ctx, config);

  // ── Build command args ─────────────────────────────────────────────────
  const args: string[] = ["chat", "-q", prompt];

  args.push("--model", model);
  if (provider) args.push("--provider", provider);
  args.push("--max-iterations", String(maxIterations));

  if (quietMode) args.push("--quiet");
  if (skipMemory) args.push("--skip-memory");
  if (worktreeMode) args.push("-w");

  if (enabledToolsets?.length) {
    args.push("--toolsets", enabledToolsets.join(","));
  }
  if (disabledToolsets?.length) {
    args.push("--disable-toolsets", disabledToolsets.join(","));
  }
  if (contextFiles?.length) {
    for (const f of contextFiles) {
      args.push("--context-file", f);
    }
  }

  // Session resume: if we have a previous session, resume it
  const prevSessionId = cfgString(
    (ctx.runtime?.sessionParams as Record<string, unknown> | null)?.sessionId,
  );
  if (persistSession && prevSessionId) {
    args.push("--resume", prevSessionId);
  }

  if (extraArgs?.length) {
    args.push(...extraArgs);
  }

  // ── Build environment ──────────────────────────────────────────────────
  const env: Record<string, string> = {
    ...(process.env as Record<string, string>),
    ...buildPaperclipEnv(ctx.agent),
  };

  // Pass Paperclip context via environment
  if (ctx.runId) env.PAPERCLIP_RUN_ID = ctx.runId;
  const taskId = cfgString(ctx.config?.taskId);
  if (taskId) env.PAPERCLIP_TASK_ID = taskId;

  // Merge user-specified env vars
  const userEnv = config.env as Record<string, string> | undefined;
  if (userEnv && typeof userEnv === "object") {
    Object.assign(env, userEnv);
  }

  // ── Resolve working directory ──────────────────────────────────────────
  const cwd =
    cfgString(config.cwd) || cfgString(ctx.config?.workspaceDir) || ".";
  try {
    await ensureAbsoluteDirectory(cwd);
  } catch {
    // Non-fatal: let the process start and fail with a better error
  }

  // ── Log start ──────────────────────────────────────────────────────────
  await ctx.onLog(
    "stdout",
    `[hermes] Starting Hermes Agent (model=${model})\n`,
  );
  await ctx.onLog(
    "stdout",
    `[hermes] Command: ${hermesCmd} ${args.slice(0, 3).join(" ")} ...\n`,
  );
  if (prevSessionId) {
    await ctx.onLog(
      "stdout",
      `[hermes] Resuming session: ${prevSessionId}\n`,
    );
  }

  // ── Execute ────────────────────────────────────────────────────────────
  const result = await runChildProcess(ctx.runId, hermesCmd, args, {
    cwd,
    env,
    timeoutSec,
    graceSec,
    onLog: ctx.onLog,
  });

  // ── Parse output ───────────────────────────────────────────────────────
  const parsed = parseHermesOutput(result.stdout || "", result.stderr || "");

  await ctx.onLog(
    "stdout",
    `[hermes] Exit code: ${result.exitCode ?? "null"}, timed out: ${result.timedOut ?? false}\n`,
  );

  // ── Build result ───────────────────────────────────────────────────────
  const executionResult: AdapterExecutionResult = {
    exitCode: result.exitCode,
    signal: result.signal,
    timedOut: result.timedOut,
    provider: provider || "anthropic",
    model,
  };

  if (parsed.errorMessage) {
    executionResult.errorMessage = parsed.errorMessage;
  }

  if (parsed.usage) {
    executionResult.usage = parsed.usage;
  }

  if (parsed.costUsd !== undefined) {
    executionResult.costUsd = parsed.costUsd;
  }

  // Store session ID for next run via sessionParams
  if (persistSession && parsed.sessionId) {
    executionResult.sessionParams = { sessionId: parsed.sessionId };
    executionResult.sessionDisplayId = parsed.sessionId.slice(0, 16);
  }

  return executionResult;
}
