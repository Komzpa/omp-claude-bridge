// Regression tests for the shared-session cross-caller bug: syncSharedSession
// used to read/write a single module-level `let sharedSession`, so every
// caller that runs through the registered provider function — including a
// reentrant subagent's own QueryContext — observed and mutated the SAME
// session state. A subagent whose own prior-message count happened to equal
// (or trail by one assistant message) another caller's cursor would then
// have its request resumed against that OTHER caller's Claude Code session,
// which is exactly what was reported live: a worker "finished but lost its
// task context, answering as the root session instead".
//
// The REUSE/REBUILD decision now lives in query-state.ts's planSessionSync,
// a pure function keyed on the QueryContext passed in (`qc.sharedSession`),
// never a bare module global. These tests exercise it directly against two
// independent QueryContext instances (matching how streamClaudeAgentSdk uses
// ctx() for the root vs. `new QueryContext()` per reentrant subagent).
//
// query-state.ts and convert.ts (developer-role prompt extraction, below)
// are deliberately dependency-light — see each file's header comment — so
// they can be imported from a plain-Node test. index.ts itself transitively
// requires the `bun` runtime (via @oh-my-pi/pi-coding-agent -> pi-utils's
// frontmatter.ts) and can't be imported here.

import test from "node:test";
import assert from "node:assert/strict";

import { QueryContext, planSessionSync } from "../src/query-state.ts";
import { extractUserPrompt } from "../src/convert.ts";

function userMsg(text) {
	return { role: "user", content: text, timestamp: Date.now() };
}

function assistantMsg(text) {
	return { role: "assistant", content: [{ type: "text", text }], timestamp: Date.now() };
}

const CWD = "/repo";

test("two interleaved callers with different histories each get their own session plan", () => {
	const qcA = new QueryContext();
	const qcB = new QueryContext();

	// A's first turn: 2 prior messages, no session yet -> rebuild.
	const aTurn1Prior = [userMsg("A prior question"), assistantMsg("A prior answer")];
	const planA1 = planSessionSync(qcA, aTurn1Prior, CWD);
	assert.equal(planA1.kind, "rebuild");
	assert.equal(planA1.previousSessionId, undefined);
	// Simulate the caller applying the rebuild (this is what index.ts's
	// syncSharedSession does after actually creating the CC session file).
	qcA.sharedSession = { sessionId: "session-A", cursor: aTurn1Prior.length, cwd: CWD };

	// Interleaved: B's first turn, with a DIFFERENT prior message count (4).
	const bTurn1Prior = [
		userMsg("B prior 1"), assistantMsg("B prior 1 reply"),
		userMsg("B prior 2"), assistantMsg("B prior 2 reply"),
	];
	const planB1 = planSessionSync(qcB, bTurn1Prior, CWD);
	assert.equal(planB1.kind, "rebuild");
	assert.equal(planB1.previousSessionId, undefined, "B must not inherit A's session as its own 'previous' session");
	qcB.sharedSession = { sessionId: "session-B", cursor: bTurn1Prior.length, cwd: CWD };

	// Back to A: A's own state must be untouched by B's call in between.
	assert.equal(qcA.sharedSession.sessionId, "session-A");
	assert.equal(qcA.sharedSession.cursor, 2);

	// A's second turn: prior length now matches A's own cursor (2) exactly ->
	// REUSE of A's own session, never B's.
	const aTurn2Prior = [userMsg("A prior question"), assistantMsg("A prior answer")];
	const planA2 = planSessionSync(qcA, aTurn2Prior, CWD);
	assert.equal(planA2.kind, "reuse");
	assert.equal(planA2.sessionId, "session-A", "A's continuation must reuse A's own session, not B's");

	// And B's own state must likewise be untouched by A's second call.
	assert.equal(qcB.sharedSession.sessionId, "session-B");
	assert.equal(qcB.sharedSession.cursor, 4);
});

test("a subagent whose history satisfies the REUSE guard against the root's cursor does not resume the root's session", () => {
	const qcRoot = new QueryContext();
	const qcSub = new QueryContext();

	// Root: establish a session with cursor=2.
	const rootPrior = [userMsg("root prior question"), assistantMsg("root prior answer")];
	const rootPlan = planSessionSync(qcRoot, rootPrior, CWD);
	assert.equal(rootPlan.kind, "rebuild");
	qcRoot.sharedSession = { sessionId: "session-root", cursor: rootPrior.length, cwd: CWD };

	// Subagent: a brand-new, unrelated conversation whose own prior-message
	// count (2) happens to equal the root's cursor (2), with completely
	// different content. The pre-fix guard only compared LENGTH
	// (`priorMessages.length >= sharedSession.cursor`, taking REUSE whenever
	// the resulting `missed` slice was empty) — it never checked message
	// identity/content — and read that cursor off a single shared global, so
	// a subagent merely reaching "2 prior messages" was enough to trigger a
	// false REUSE of the root's session under the old code.
	const subPrior = [userMsg("completely unrelated subagent topic"), assistantMsg("unrelated subagent reply")];
	const subPlan = planSessionSync(qcSub, subPrior, CWD);

	// qcSub has never had a session, so this must REBUILD into a session of
	// its own — it can never satisfy qcSub's own (null) REUSE guard, and it
	// has no way to observe qcRoot's session at all.
	assert.equal(subPlan.kind, "rebuild", "a subagent with no session of its own must always rebuild, never reuse another caller's");
	assert.notEqual(subPlan.kind, "reuse");

	// Root's own session must be entirely unaffected by the subagent's call.
	assert.equal(qcRoot.sharedSession.sessionId, "session-root");
	assert.equal(qcRoot.sharedSession.cursor, 2);
});

test("planSessionSync REUSE only fires on the QueryContext's OWN cached session, never a same-shaped session from elsewhere", () => {
	// Belt-and-suspenders version of the test above, phrased directly in
	// terms of the REUSE guard: give two QueryContexts numerically identical
	// sharedSession bookkeeping (same cursor) but different session ids and
	// different cwds, and confirm each only ever reuses its own.
	const qcA = new QueryContext();
	const qcB = new QueryContext();
	qcA.sharedSession = { sessionId: "session-A", cursor: 2, cwd: "/repo/a" };
	qcB.sharedSession = { sessionId: "session-B", cursor: 2, cwd: "/repo/b" };

	const prior = [userMsg("same-shaped prior 1"), assistantMsg("same-shaped prior 2")];

	const planA = planSessionSync(qcA, prior, "/repo/a");
	const planB = planSessionSync(qcB, prior, "/repo/b");

	assert.equal(planA.kind, "reuse");
	assert.equal(planA.sessionId, "session-A");
	assert.equal(planB.kind, "reuse");
	assert.equal(planB.sessionId, "session-B");
	assert.notEqual(planA.sessionId, planB.sessionId);
});

test("extractUserPrompt treats a trailing developer-role message as prompt-bearing", () => {
	const developerMessages = [
		{ role: "user", content: "earlier turn", timestamp: Date.now() },
		{ role: "assistant", content: [{ type: "text", text: "earlier reply" }], timestamp: Date.now() },
		{ role: "developer", content: "<file path=\"notes.md\">hello from a file mention</file>", timestamp: Date.now() },
	];

	// Before the fix this returned null (only "user" was accepted), which
	// made streamClaudeAgentSdk hit its empty-prompt guard and silently
	// replace real developer-role content with a "[continue]" placeholder.
	// Confirmed live in ~/.omp/agent/claude-bridge-diag.log: dozens of
	// empty_prompt dumps with lastMsgRole="developer".
	assert.equal(extractUserPrompt(developerMessages), "<file path=\"notes.md\">hello from a file mention</file>");

	// A non-prompt-bearing trailing role (e.g. a bare tool result) still
	// correctly yields null.
	const toolResultTail = [...developerMessages, { role: "toolResult", toolCallId: "x", content: [{ type: "text", text: "result" }], timestamp: Date.now() }];
	assert.equal(extractUserPrompt(toolResultTail), null);
});

import { queryContextForCaller, resetStack, ctx } from "../src/query-state.ts";
import { convertPiMessages } from "../src/convert.ts";

test("idle callers are keyed by owner, never handed the root context", () => {
	resetStack();
	const rootOwner = new Map();
	const childOwner = new Map();
	const root = queryContextForCaller(rootOwner, false);
	assert.equal(root, ctx());
	root.sharedSession = { sessionId: "root-session", cursor: 3, cwd: CWD };
	// Root query finished (not busy); a different caller arrives.
	const child = queryContextForCaller(childOwner, false);
	assert.notEqual(child, root);
	assert.equal(child.sharedSession, null);
	// Same callers get the same contexts back on later idle turns.
	assert.equal(queryContextForCaller(childOwner, false), child);
	assert.equal(queryContextForCaller(rootOwner, false), root);
	// Same owner concurrently busy -> fresh context, not the busy one.
	child.activeQuery = {};
	assert.notEqual(queryContextForCaller(childOwner, true), child);
	// No owner info while root is busy -> fresh context.
	assert.notEqual(queryContextForCaller(undefined, true), root);
});

test("convertPiMessages replays developer messages as user input", () => {
	const { anthropicMessages } = convertPiMessages([
		{ role: "developer", content: "file mention", timestamp: 1 },
	]);
	assert.deepEqual(anthropicMessages, [{ role: "user", content: "file mention" }]);
});
