// Query state: QueryContext class + context stack.
//
// All per-query and per-turn mutable state lives here. Reentrant queries
// (subagents) push the parent context onto a stack and get a fresh instance.
// Adding a new field = one property on the class.
//
// Extracted from index.ts so tests can import without activating the extension.

import type { AssistantMessage, AssistantMessageEventStream, Context, Model, ProviderSessionState, SimpleStreamOptions } from "@oh-my-pi/pi-coding-agent/extensibility/legacy-pi-ai-shim";
import type { McpResult } from "./extract-tool-results.js";

export interface PendingToolCall {
	toolName: string;
	resolve: (result: McpResult) => void;
}

// Claude Code session state backing one caller's resumable conversation.
// Lives on the owning QueryContext (see `sharedSession` below) — NEVER as a
// bare module-level global. A module-level `let sharedSession` is shared by
// every caller that runs through the same registered provider function,
// including reentrant subagent queries (index.ts's "Subsequent instance...
// route through the parent's streamSimple via reentrant QueryContexts").
// Without per-context ownership, a subagent whose own prior-message count
// happens to satisfy the REUSE guard resumes a DIFFERENT caller's (e.g. the
// root session's) Claude Code session — see omp-claude-bridge issue: worker
// "finished but lost its task context, answering as the root session
// instead".
export interface SessionState {
	sessionId: string;
	cursor: number;
	cwd: string;
	// Force the next syncSharedSession call down the REBUILD path. Set when
	// pi has mutated its messages array out from under us (compact, tree
	// navigation) or after an abort left the JSONL in an indeterminate state.
	// REBUILD wipes and rewrites the file to match pi's current history.
	needsRebuild?: boolean;
	// Set ONLY after an abort. The killed CC subprocess may still be flushing
	// a late "[Request interrupted by user]" record to the session JSONL.
	// Reusing the same sessionId/path would race that orphan write into our
	// fresh file and break CC's parent-uuid chain on the next resume. When
	// this flag is set, REBUILD takes a fresh UUID and skips deleteSession
	// so the orphan writes land on a dead inode. Compact/tree do NOT set
	// this — there's no concurrent CC writer during those events, so
	// in-place rebuild (preserve UUID, deleteSession + createSession) is safe.
	forceRotate?: boolean;
}

export class QueryContext {
	// Query-scoped (fully isolated per query)
	activeQuery: unknown | null = null;
	currentPiStream: AssistantMessageEventStream | null = null;
	latestCursor = 0;
	pendingToolCalls = new Map<string, PendingToolCall>();
	pendingResults = new Map<string, McpResult>();
	turnToolCallIds: string[] = [];
	nextHandlerIdx = 0;
	deferredUserMessages: string[] = [];
	// This caller's own Claude Code session — see SessionState doc above.
	// A fresh reentrant QueryContext starts with no session, so a subagent
	// always gets its own CC session on first turn instead of inheriting
	// whichever session happened to be shared at module scope.
	sharedSession: SessionState | null = null;
	// Installed by the provider while a query is running. Closing the host's
	// state store must stop that query, not just forget its resumable session.
	onOwnerClose: (() => void) | null = null;

	close(): void {
		try {
			this.onOwnerClose?.();
		} finally {
			this.onOwnerClose = null;
			this.activeQuery = null;
			this.sharedSession = null;
			this.deferredUserMessages = [];
			for (const pending of this.pendingToolCalls.values()) {
				pending.resolve({ content: [{ type: "text", text: "Session closed" }] });
			}
			this.pendingToolCalls.clear();
			this.pendingResults.clear();
			this.turnToolCallIds = [];
		}
	}

	// Per-turn (reset together)
	turnOutput: AssistantMessage | null = null;
	turnStarted = false;
	turnSawStreamEvent = false;
	turnSawToolCall = false;

	get turnBlocks(): Array<any> {
		if (!this.turnOutput) throw new Error("turnBlocks accessed before resetTurnState");
		return this.turnOutput.content;
	}

	resetTurnState(model: Model<any>): void {
		this.turnOutput = {
			role: "assistant", content: [],
			api: model.api, provider: model.provider, model: model.id,
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			stopReason: "stop", timestamp: Date.now(),
		};
		this.turnStarted = false;
		this.turnSawStreamEvent = false;
		this.turnSawToolCall = false;
		// turnToolCallIds and nextHandlerIdx are NOT reset — they persist across
		// tool-result delivery callbacks within the same assistant message.
	}
}

let _ctx = new QueryContext();
const contextStack: QueryContext[] = [];

export function ctx(): QueryContext { return _ctx; }

// --- Caller-keyed contexts ---
//
// "Is a query active right now" is NOT a caller identity: a subagent (or another
// extension instance) that calls the globally registered provider after the
// root query finished would otherwise be handed the root ctx() and could resume
// the root's Claude Code session. `owner` is the host's per-agent-session
// `options.providerSessionState` Map (stable for one caller, distinct between
// callers). The first owner to arrive while the root slot is free claims the
// root ctx(); every other owner gets its own persistent QueryContext, so its
// own cursor/session is never compared against another caller's.
// Map (not WeakMap): session_compact / session_tree handlers must enumerate
// idle contexts too. The host closes its typed provider states on fresh,
// switch and dispose; our entry evicts precisely that owner's cached state.
type CallerOwner = NonNullable<SimpleStreamOptions["providerSessionState"]>;
interface CallerState extends ProviderSessionState {
	context: QueryContext;
	contexts: Set<QueryContext>;
}
const CALLER_STATE_KEY = "omp-claude-bridge:query-context";
let callerContexts = new Map<CallerOwner, CallerState>();
let rootOwner: CallerOwner | undefined;
// Set when an unowned call ran on the root ctx(): the cached session/cursor
// may belong to that caller, so the first owned caller must get a fresh
// context instead of inheriting it.
let rootServedUnowned = false;

export function queryContextForCaller(owner: CallerOwner | undefined, rootBusy: boolean): QueryContext {
	if (!owner) {
		if (rootBusy) return new QueryContext();
		// Owned caller claimed the root slot: its sharedSession/cursor lives on
		// _ctx. An unowned call must never resume it (4296223752).
		if (rootOwner !== undefined) return new QueryContext();
		rootServedUnowned = true;
		return _ctx;
	}
	const existing = callerContexts.get(owner);
	if (existing) {
		if (!existing.context.activeQuery) return existing.context;
		const concurrent = new QueryContext();
		existing.contexts.add(concurrent);
		return concurrent;
	}
	const claimRoot = !rootBusy && !rootServedUnowned && rootOwner === undefined;
	const context = claimRoot ? _ctx : new QueryContext();
	if (claimRoot) rootOwner = owner;
	let closed = false;
	const state: CallerState = {
		context,
		contexts: new Set([context]),
		close() {
			if (closed) return;
			closed = true;
			// A delayed/repeated close must never evict a replacement generation.
			if (callerContexts.get(owner) === state) {
				callerContexts.delete(owner);
				if (rootOwner === owner) {
					rootOwner = undefined;
					rootServedUnowned = false;
				}
			}
			// A stacked root frame must not resurrect the retired context on pop.
			for (let i = 0; i < contextStack.length; i++) {
				if (state.contexts.has(contextStack[i])) contextStack[i] = new QueryContext();
			}
			if (state.contexts.has(_ctx)) _ctx = new QueryContext();
			if (owner.get(CALLER_STATE_KEY) === state) owner.delete(CALLER_STATE_KEY);
			let firstError: unknown;
			try {
				for (const owned of state.contexts) {
					try {
						owned.close();
					} catch (error) {
						firstError ??= error;
					}
				}
			} finally {
				state.contexts.clear();
			}
			if (firstError !== undefined) throw firstError;
		},
	};
	callerContexts.set(owner, state);
	owner.set(CALLER_STATE_KEY, state);
	return context;
}

// A concurrent call's context only needs owner tracking while it is running;
// retaining every completed transient would recreate the cache leak.
export function releaseConcurrentQueryContext(owner: CallerOwner | undefined, context: QueryContext): void {
	if (!owner) return;
	const state = callerContexts.get(owner);
	if (state && state.context !== context) state.contexts.delete(context);
}


// All live caller contexts, including idle ones, remain enumerable until the
// owning host closes its provider state (compact/tree must invalidate them).
export function* cachedCallerContexts(): Iterable<QueryContext> {
	for (const state of callerContexts.values()) yield* state.contexts;
}

export function stackDepth(): number { return contextStack.length; }

export function pushContext(): void {
	if (!_ctx.activeQuery) throw new Error("pushContext() called with no active query");
	contextStack.push(_ctx);
	_ctx = new QueryContext();
}

export function popContext(): void {
	if (contextStack.length === 0) throw new Error("popContext() called with empty stack");
	const parent = contextStack[contextStack.length - 1];
	parent.deferredUserMessages.push(..._ctx.deferredUserMessages);
	_ctx = contextStack.pop()!;
}

// Test-only: drop all state so test files can start from a clean module.
// Not called from production.
export function resetStack(): void {
	_ctx = new QueryContext();
	contextStack.length = 0;
	callerContexts = new Map();
	rootOwner = undefined;
	rootServedUnowned = false;
}

// --- Session sync decision ---
//
// Pure REUSE/REBUILD decision for one QueryContext's Claude Code session,
// extracted out of index.ts's syncSharedSession so it can be unit-tested
// without pulling in the rest of the extension's dependency graph (index.ts
// transitively imports @oh-my-pi/pi-coding-agent modules that require the
// `bun` runtime, so it can't be imported from a plain-Node test; this file
// deliberately has none of that — see the file header).
//
// index.ts's syncSharedSession calls this first, then performs the actual
// cc-session-io file I/O for a "rebuild" plan (this function does none) and
// applies the resulting SessionState back onto `qc`. The "reuse" case is
// itself pure aside from advancing `qc.sharedSession.cursor` past a trailing
// assistant-only message — no I/O either way.
export type SessionSyncPlan =
	| { kind: "clean-start" }
	| { kind: "clean-start-preserve" }
	| { kind: "reuse"; sessionId: string; advancedPastTrailingAssistant: boolean }
	| { kind: "rebuild"; previousSessionId: string | undefined; previousCursor: number; preserveId: boolean };

/**
 * `qc` must be the QueryContext that OWNS the Claude Code session for the
 * caller making this request — the caller's own reentrant context for a
 * subagent, or the root ctx() for the top-level session. Never a
 * QueryContext shared across callers; see SessionState's doc for why.
 */
export function planSessionSync(qc: QueryContext, priorMessages: Context["messages"], cwd: string): SessionSyncPlan {
	const session = qc.sharedSession;

	// REUSE path
	//
	// Guard on priorMessages.length >= cursor: a shorter incoming context cannot
	// be a continuation of the cached session. This is the general invariant for
	// pi-side history rewrites such as /compact and session_tree: without it,
	// missed = [].slice(cursor) can falsely hit REUSE and resume an unrelated
	// longer CC session. See issue #25. It also protects across callers now that
	// sharedSession is per-QueryContext: a subagent's own priorMessages/cursor
	// are always compared against ITS OWN qc.sharedSession, never another
	// caller's, so a subagent can no longer satisfy this guard against a
	// differently-sized root (or sibling subagent) session.
	if (session && !session.needsRebuild && priorMessages.length >= session.cursor) {
		const missed = priorMessages.slice(session.cursor);
		const trailingAssistantOnly = missed.length === 1 && (missed[0] as { role?: string }).role === "assistant";
		if (missed.length === 0 || trailingAssistantOnly) {
			if (trailingAssistantOnly) {
				qc.sharedSession = { ...session, cursor: priorMessages.length, cwd };
			}
			return { kind: "reuse", sessionId: session.sessionId, advancedPastTrailingAssistant: trailingAssistantOnly };
		}
	}
	// Only reachable when needsRebuild is false — user-facing history rewrites
	// (/compact, session_tree, /new, fork) always set needsRebuild or clear
	// sharedSession before the next syncSharedSession call. In practice this
	// fires only for isolated compact-summary subprocesses.
	if (session && !session.needsRebuild && priorMessages.length < session.cursor) {
		return { kind: "clean-start-preserve" };
	}

	// REBUILD path
	if (priorMessages.length === 0) {
		return { kind: "clean-start" };
	}
	return {
		kind: "rebuild",
		previousSessionId: session?.sessionId,
		previousCursor: session?.cursor ?? 0,
		// preserveId: rebuild in place (deleteSession + createSession with the
		// existing UUID), so prompt-cache UUIDs stay stable for log correlation
		// and for any tools that key off them. Skipped only when there's a
		// concurrent writer we shouldn't race — see SessionState.forceRotate.
		preserveId: session?.sessionId !== undefined && !session?.forceRotate,
	};
}
