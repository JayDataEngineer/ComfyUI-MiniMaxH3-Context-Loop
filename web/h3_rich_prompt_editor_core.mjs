import {H3_ALL_SECTIONS} from "./h3_prompt_schema_core.mjs?v=0.7.6";

export function revealPromptCaret(editor) {
    const selection = editor.ownerDocument.defaultView.getSelection();
    if (!selection?.rangeCount || !editor.contains(selection.anchorNode)) return;
    const range = selection.getRangeAt(0).cloneRange();
    range.collapse(true);
    const caret = range.getBoundingClientRect();
    if (!caret.height) return;
    const box = editor.getBoundingClientRect();
    const top = box.top + editor.clientTop;
    const bottom = top + editor.clientHeight;
    if (caret.top < top) editor.scrollTop -= top - caret.top;
    else if (caret.bottom > bottom) editor.scrollTop += caret.bottom - bottom;
}

export function richGenerationMode(referenceMode) {
    if (["tagged", "scheduled", "native"].includes(referenceMode)) return "Ref2VA";
    if (referenceMode === "native_keyframes") return "I2VA/FL2VA";
    return "H3 chain scene";
}

function recordTokens(record) {
    const tokens = [record?.token, record?.label];
    const explicitTag = String(record?.tag ?? "")
        .trim().replace(/^[@#]+/, "");
    const tokenTag = /^@([A-Za-z][A-Za-z0-9_-]{0,63})$/
        .exec(String(record?.nativeToken ?? record?.token ?? "").trim())?.[1];
    const canonicalTag = explicitTag || tokenTag;
    if (canonicalTag) tokens.push(`#${canonicalTag}`);
    return tokens
        .map((value) => String(value ?? "").trim())
        .filter(Boolean);
}

export function referenceRecordMap(records) {
    const map = new Map();
    for (const record of Array.isArray(records) ? records : []) {
        for (const token of recordTokens(record)) map.set(token, record);
    }
    return map;
}

// Keep this alias grammar aligned with chain_nodes.py's reference compiler.
// In particular, sentence punctuation such as the period in "Use @hero."
// belongs to the surrounding prompt, not to the decorated reference token.
const RICH_SECTION_ALTERNATION = H3_ALL_SECTIONS.join("|");
const RICH_TOKEN_PATTERN = new RegExp(
    `^(${RICH_SECTION_ALTERNATION}):|`
    + "((?<![A-Za-z0-9_])#[A-Za-z][A-Za-z0-9_-]{0,63}(?:\\[[0-9]+(?:\\.[0-9]+)?s?\\]|(?!\\[))(?![A-Za-z0-9_-])"
    + "|(?<![A-Za-z0-9_])@[A-Za-z][A-Za-z0-9_-]{0,63}(?![A-Za-z0-9_-])"
    + "|<(?:Picture|Video|Audio|Subject)\\s+\\d+>|<\\/?d>|<scenetrans>|<cutoff>|<\\|(?:cutoff|lyrics_start|lyrics_end|caption_start|caption_end)\\|>"
    + "|(?<![A-Za-z0-9_])\\(S\\d+(?:,S\\d+)*\\))",
    "gim",
);

export function tokenizeRichPrompt(text, records = []) {
    const source = String(text ?? "");
    const recordMap = referenceRecordMap(records);
    const parts = [];
    let offset = 0;
    for (const match of source.matchAll(RICH_TOKEN_PATTERN)) {
        const index = match.index ?? 0;
        if (index > offset) parts.push({type: "text", text: source.slice(offset, index)});
        const token = match[0];
        const lower = token.toLowerCase();
        const section = lower.endsWith(":")
            ? H3_ALL_SECTIONS.find((item) => `${item}:` === lower) : null;
        const semanticMatch = /^#([a-z][a-z0-9_-]{0,63})(?:\[([0-9]+(?:\.[0-9]+)?)s?\])?$/i.exec(token);
        // A semantic-only asset has no native @tag token. Resolve bare #tag
        // and #tag[time] through the timestamp-independent registered key.
        const recordKey = semanticMatch ? `#${semanticMatch[1]}` : token;
        const record = recordMap.get(recordKey) ?? null;
        if (section) {
            parts.push({type:"section", kind:"section", text:token, section, unresolved:false});
        } else if (record || lower.startsWith("@") || semanticMatch || /^<(?:picture|video|audio)\s/i.test(lower)) {
            const namedKind = lower.startsWith("<picture") ? "picture"
                : lower.startsWith("<video") ? "video"
                    : lower.startsWith("<audio") ? "audio" : null;
            const semanticValid = !semanticMatch || record?.kind === "picture";
            parts.push({
                type: "reference",
                text: token,
                kind: record?.kind ?? namedKind ?? "unknown",
                record,
                unresolved: !record || !semanticValid,
                semantic: Boolean(semanticMatch),
                timestamp: semanticMatch?.[2] !== undefined
                    ? Number(semanticMatch[2]) : null,
            });
        } else if (lower.startsWith("<subject")) {
            parts.push({type: "subject", text: token});
        } else if (lower === "<scenetrans>" || lower === "<cutoff>" || lower === "<|cutoff|>") {
            parts.push({type:"flow", kind:"flow", text:token, unresolved:false});
        } else if (/^<\|(?:lyrics|caption)_/.test(lower)) {
            parts.push({type:"dialogue", text:token});
        } else if (lower.startsWith("(s")) {
            parts.push({type:"speaker", kind:"speaker", text:token, unresolved:false});
        } else {
            parts.push({type: "dialogue", text: token});
        }
        offset = index + token.length;
    }
    if (offset < source.length) parts.push({type: "text", text: source.slice(offset)});
    return parts;
}

function undoInputGroup(inputType) {
    const type = String(inputType ?? "");
    if (type === "insertText") return "typing";
    if (type === "deleteContentBackward") return "backspace";
    if (type === "deleteContentForward") return "delete";
    return "";
}

export function promptUndoDirection(event) {
    if (!event || event.altKey || !(event.ctrlKey || event.metaKey)) return null;
    const key = String(event.key ?? "").toLowerCase();
    if (key === "z") return event.shiftKey ? "redo" : "undo";
    if (key === "y" && !event.shiftKey) return "redo";
    return null;
}

/** Text-level undo survives contenteditable token decoration, which replaces
 * DOM children and therefore discards the browser's native undo manager. */
export class PromptUndoHistory {
    constructor(initialText = "", {limit = 100, coalesceMs = 750} = {}) {
        this.limit = Math.max(1, Math.trunc(Number(limit) || 100));
        this.coalesceMs = Math.max(0, Number(coalesceMs) || 0);
        this.reset(initialText);
    }

    reset(text = "") {
        this.current = String(text ?? "");
        this.undoStack = [];
        this.redoStack = [];
        this.lastGroup = "";
        this.lastTime = 0;
        return this.current;
    }

    align(text = "") {
        const value = String(text ?? "");
        if (value === this.current) return false;
        this.reset(value);
        return true;
    }

    record(text, {inputType = "", now = Date.now()} = {}) {
        const value = String(text ?? "");
        if (value === this.current) return false;
        const group = undoInputGroup(inputType);
        const timestamp = Number(now) || 0;
        const coalesced = Boolean(
            group && group === this.lastGroup
            && timestamp - this.lastTime >= 0
            && timestamp - this.lastTime <= this.coalesceMs
        );
        if (!coalesced) {
            this.undoStack.push(this.current);
            if (this.undoStack.length > this.limit) this.undoStack.shift();
        }
        this.current = value;
        this.redoStack = [];
        this.lastGroup = group;
        this.lastTime = timestamp;
        return true;
    }

    undo() {
        if (!this.undoStack.length) return null;
        this.redoStack.push(this.current);
        this.current = this.undoStack.pop();
        this.lastGroup = "";
        this.lastTime = 0;
        return this.current;
    }

    redo() {
        if (!this.redoStack.length) return null;
        this.undoStack.push(this.current);
        this.current = this.redoStack.pop();
        this.lastGroup = "";
        this.lastTime = 0;
        return this.current;
    }
}
