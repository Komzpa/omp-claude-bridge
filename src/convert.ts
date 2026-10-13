// Pure pi→Anthropic message conversion helpers.
// Extracted so they can be tested without pulling in the full extension runtime.

import type { Context, Message as PiMessage } from "@oh-my-pi/pi-coding-agent/extensibility/legacy-pi-ai-shim";
import type { Base64ImageSource, ContentBlockParam } from "@anthropic-ai/sdk/resources";
import type { Message as SessionMessage } from "cc-session-io";
import { pascalCase } from "change-case";

export const PROVIDER_ID = "claude-bridge";

export const PI_TO_SDK_TOOL_NAME: Record<string, string> = {
	read: "Read", write: "Write", edit: "Edit", bash: "Bash",
};

export function sanitizeToolId(id: string, cache: Map<string, string>): string {
	const existing = cache.get(id);
	if (existing) return existing;
	const clean = id.replace(/[^a-zA-Z0-9_-]/g, "_");
	cache.set(id, clean);
	return clean;
}

export function mapPiToolNameToSdk(name: string, customToolNameToSdk?: Map<string, string>): string {
	if (!name) return "";
	const normalized = name.toLowerCase();
	if (customToolNameToSdk) {
		const mapped = customToolNameToSdk.get(name) ?? customToolNameToSdk.get(normalized);
		if (mapped) return mapped;
	}
	if (PI_TO_SDK_TOOL_NAME[normalized]) return PI_TO_SDK_TOOL_NAME[normalized];
	return pascalCase(name);
}

export function messageContentToText(
	content: string | Array<{ type: string; text?: string; data?: string; mimeType?: string }>,
): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	const parts = [];
	let hasText = false;
	for (const block of content) {
		if (block.type === "text" && block.text) { parts.push(block.text); hasText = true; }
		else if (block.type !== "text" && block.type !== "image") { parts.push(`[${block.type}]`); }
	}
	return hasText ? parts.join("\n") : "";
}

// "developer" carries the exact same `content: string | (TextContent |
// ImageContent)[]` shape as "user" (pi-ai's DeveloperMessage/UserMessage) and
// legitimately lands as the trailing context message — e.g. a file-mention or
// hook-message turn pi emits on the higher-priority developer slot ahead of
// (or instead of) a "user" message. Treating only "user" as prompt-bearing
// made every such turn fall through to streamClaudeAgentSdk's "empty prompt"
// guard and get silently replaced with a "[continue]" placeholder, dropping
// the actual content. Confirmed live via ~/.omp/agent/claude-bridge-diag.log:
// dozens of empty_prompt dumps with lastMsgRole="developer" and no active
// query (i.e. not a tool-result or reentrancy issue — a genuine gap here).
export function isPromptBearingRole(role: string | undefined): boolean {
	return role === "user" || role === "developer";
}

/** Extract the last user/developer message from context as a prompt string.
 *  Returns null if the last message isn't prompt-bearing. */
export function extractUserPrompt(messages: Context["messages"]): string | null {
	const last = messages[messages.length - 1];
	if (!last || !isPromptBearingRole(last.role)) return null;
	if (typeof last.content === "string") return last.content;
	return messageContentToText(last.content) || "";
}

/** Extract the last user/developer message as ContentBlockParam[] (preserving images).
 *  Returns null if no images — caller should fall back to string prompt. */
export function extractUserPromptBlocks(messages: Context["messages"]): ContentBlockParam[] | null {
	const last = messages[messages.length - 1];
	if (!last || !isPromptBearingRole(last.role)) return null;
	if (typeof last.content === "string") return null;
	if (!Array.isArray(last.content)) return null;
	let hasImage = false;
	const blocks: ContentBlockParam[] = [];
	for (const block of last.content) {
		if (block.type === "text" && block.text) {
			blocks.push({ type: "text", text: block.text });
		} else if (block.type === "image") {
			if (!(block as any).data || !(block as any).mimeType) continue;
			hasImage = true;
			blocks.push({
				type: "image",
				source: { type: "base64", media_type: block.mimeType as Base64ImageSource["media_type"], data: block.data },
			});
		}
	}
	return hasImage ? blocks : null;
}

/** Convert OMP message array to Anthropic API format. */
export function convertPiMessages(
	messages: PiMessage[],
	customToolNameToSdk?: Map<string, string>,
): { anthropicMessages: SessionMessage[]; sanitizedIds: Map<string, string> } {
	const anthropicMessages = [];
	const sanitizedIds = new Map();

	for (const msg of messages) {
		// "developer" (file mentions, hook context) is replayed as user input so a
		// rebuild after compaction / tree navigation keeps it in Claude's history.
		if (msg.role === "user" || msg.role === "developer") {
			if (typeof msg.content === "string") {
				anthropicMessages.push({ role: "user", content: msg.content || "[empty]" });
			} else if (Array.isArray(msg.content)) {
				const parts = [];
				for (const block of msg.content) {
					if (block.type === "text" && block.text) parts.push({ type: "text", text: block.text });
					else if (block.type === "image" && block.data && block.mimeType) {
						parts.push({ type: "image", source: { type: "base64", media_type: block.mimeType, data: block.data } });
					}
				}
				anthropicMessages.push({ role: "user", content: parts.length ? parts : "[image]" });
			} else {
				anthropicMessages.push({ role: "user", content: "[empty]" });
			}
		} else if (msg.role === "assistant") {
			const content = Array.isArray(msg.content) ? msg.content : [];
			const blocks = [];
			for (const block of content) {
				if (block.type === "text" && block.text) {
					blocks.push({ type: "text", text: block.text });
				} else if (block.type === "thinking") {
					const sig = block.thinkingSignature;
					const isAnthropicProvider = msg.provider === PROVIDER_ID || msg.api === "anthropic";
					if (isAnthropicProvider && sig) {
						blocks.push({ type: "thinking", thinking: block.thinking ?? "", signature: sig });
					}
				} else if (block.type === "toolCall") {
					const toolName = mapPiToolNameToSdk(block.name, customToolNameToSdk);
					blocks.push({ type: "tool_use", id: sanitizeToolId(block.id, sanitizedIds), name: toolName, input: block.arguments ?? {} });
				}
			}
			if (!blocks.length) blocks.push({ type: "text", text: "[incompatible content omitted]" });
			anthropicMessages.push({ role: "assistant", content: blocks });
		} else if (msg.role === "toolResult") {
			const text = typeof msg.content === "string" ? msg.content : messageContentToText(msg.content);
			anthropicMessages.push({
				role: "user",
				content: [{ type: "tool_result", tool_use_id: sanitizeToolId(msg.toolCallId, sanitizedIds), content: text || "", is_error: msg.isError }],
			});
		}
	}

	return { anthropicMessages, sanitizedIds };
}
