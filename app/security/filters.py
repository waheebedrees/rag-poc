"""
Input validation and filtering for RAG queries.

Design goals:
  - Normalize input before pattern matching (defeats homoglyph/spacing attacks).
  - Collect ALL matches, not just first (full threat visibility).
  - Separate injection rules for user input vs document chunks.
  - Zero false positives on legitimate technical questions.
  - Never log raw user input.
  - Clear separation between BLOCK and WARN with documented rationale.
"""

from __future__ import annotations

import re
import unicodedata
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, NamedTuple

logger = logging.getLogger("security.filters")


class Severity(str, Enum):
    OK = "ok"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class RuleMatch:
    """A single pattern that matched."""
    rule: str
    matched_text: str       # the exact substring that triggered it
    severity: Severity

def _validation_result(
    rule: str,
    severity: Severity,
    reason: str,
    redacted: str = "",
) -> FilterResult:
    """Build a FilterResult for validation errors (no regex match)."""
    return FilterResult(
        severity=severity,
        matches=(RuleMatch(rule=rule, matched_text="", severity=severity),),
        redacted=redacted,
        reason=reason,
    )
    
@dataclass(frozen=True)
class FilterResult:
    severity: Severity
    matches: tuple[RuleMatch, ...] = field(default_factory=tuple)
    redacted: str = ""
    reason: str | None = None

    @property
    def is_safe(self) -> bool:
        return self.severity is not Severity.BLOCK

    @property
    def rule(self) -> str | None:
        """Primary matched rule (worst severity first). Backwards compat."""
        blocks = [m for m in self.matches if m.severity is Severity.BLOCK]
        if blocks:
            return blocks[0].rule
        warns = [m for m in self.matches if m.severity is Severity.WARN]
        return warns[0].rule if warns else None


#
# Normalization
#
# Applied before every pattern match.
# Defeats: homoglyphs, zero-width chars, excessive whitespace,
#          look-alike punctuation, mixed scripts in keywords.

# Homoglyph map: visually similar chars → ASCII equivalent
# Covers the most common substitutions used in adversarial prompts.
_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic → Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "і": "i", "ѕ": "s", "ј": "j",
    # Greek
    "α": "a", "ε": "e", "ο": "o", "ρ": "r", "ν": "v",
    # Full-width ASCII
    **{chr(0xFF01 + i): chr(0x21 + i) for i in range(94)},
    # Common punctuation lookalikes
    "\u2018": "'", "\u2019": "'",   # curly quotes
    "\u201C": '"', "\u201D": '"',   # curly double quotes
    "\u2013": "-", "\u2014": "-",   # en/em dash
    "\u00AD": "",                   # soft hyphen (invisible)
}

_ZERO_WIDTH = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]"
)

_MULTI_SPACE = re.compile(r"[ \t]+")


def normalize(text: str) -> str:
    """
    Return a normalized form of text for pattern matching.

    Steps:
      1. Unicode NFC normalization
      2. Strip zero-width and invisible characters
      3. Map homoglyphs to ASCII equivalents
      4. Collapse runs of spaces/tabs to a single space
      5. Lowercase

    The normalized form is ONLY used for matching, never returned to the user
    or stored — we always work with the original text otherwise.
    """
    # Step 1: NFC
    text = unicodedata.normalize("NFC", text)

    # Step 2: Zero-width chars
    text = _ZERO_WIDTH.sub("", text)

    # Step 3: Homoglyphs
    text = "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in text)

    # Step 4: Collapse whitespace (but keep newlines — they matter for structure)
    lines = text.splitlines()
    text = "\n".join(_MULTI_SPACE.sub(" ", line) for line in lines)

    # Step 5: Lowercase
    return text.lower()


#
# Redaction
#
# Applied to produce a safe string for logging.
# Patterns here are deliberately broad — we'd rather over-redact than leak.
#

@dataclass(frozen=True)
class _RedactPattern:
    name: str
    pattern: re.Pattern
    replacement: str = "[REDACTED]"


_REDACT_PATTERNS: list[_RedactPattern] = [
    _RedactPattern("openai_key",
                   re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    _RedactPattern("anthropic_key",
                   re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    _RedactPattern("google_key",
                   re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    _RedactPattern("aws_access_key",
                   re.compile(r"\b(?:AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}\b")),
    _RedactPattern("aws_secret",
                   re.compile(r"(?i)aws.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]")),
    _RedactPattern("jwt",
                   re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")),
    _RedactPattern("generic_secret",
                   re.compile(
                       r"(?i)\b(api[_\-]?key|access[_\-]?token|secret[_\-]?key|auth[_\-]?token|bearer)"
                       r"\s*[:=]\s*['\"]?([A-Za-z0-9+/=_\-]{20,})['\"]?"
                   )),
    _RedactPattern("email",
                   re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    _RedactPattern("phone",
                   # Requires country code or specific formatting to avoid matching
                   # version numbers, dates, and other numeric sequences
                   re.compile(
                       r"(?<!\d)"
                       # optional country code with +
                       r"(?:\+\d{1,3}[\s\-]?)?"
                       # area code in parens (required branch)
                       r"(?:\(\d{2,4}\)[\s\-]?)"
                       r"\d{3,4}[\s\-]?\d{4}"
                       r"|"
                       r"(?<!\d)"
                       # +1 234 567 8901
                       r"\+\d{1,3}[\s\-]\d{2,4}[\s\-]\d{3,4}[\s\-]\d{4}"
                       r"(?!\d)"
                   )),
]


def redact(text: str) -> str:
    """Mask secrets and PII. Safe to pass to loggers."""
    out = text
    for p in _REDACT_PATTERNS:
        out = p.pattern.sub(p.replacement, out)
    return out


# Injection rules
# Two separate rule sets:
#   USER_RULES   — applied to user-supplied queries
#   CHUNK_RULES  — applied to retrieved document chunks (much narrower)
#
# Each rule: (name, pattern, severity, rationale)


class _Rule(NamedTuple):
    name: str
    pattern: re.Pattern
    severity: Severity
    rationale: str


def _r(name: str, pattern: str, severity: Severity, rationale: str) -> _Rule:
    return _Rule(name, re.compile(pattern, re.IGNORECASE), severity, rationale)


# Rules applied to user queries.
# Patterns match against the NORMALIZED form of the input.
USER_RULES: list[_Rule] = [
    #  Hard blocks 
    _r(
        "ignore_previous_instructions",
        # Handles: "ignore all previous", "ignore prior", "ignore the above",
        #          "ignоre" (after homoglyph normalization)
        r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier|existing|current)\s+"
        r"(?:instructions?|prompts?|rules?|directives?|messages?|constraints?|guidelines?)\b",
        Severity.BLOCK,
        "Classic prompt injection — attempts to override system instructions.",
    ),

    _r(
        "disregard_instructions",
        r"\b(?:disregard|dismiss|discard|drop|throw\s+out)\s+"
        r"(?:all\s+|any\s+|your\s+|my\s+|the\s+)*"    # ← possessives allowed
        # ← trailing space
        r"(?:previous\s+|prior\s+|above\s+|earlier\s+|existing\s+)?"
        r"(?:instructions?|prompts?|rules?|directives?|constraints?)\b",
        Severity.BLOCK,
        "Variant of instruction override.",
    ),
    _r(
        "forget_instructions",
        r"\b(?:forget|clear|reset|override|bypass|overwrite)\s+"
        r"(?:all\s+|any\s+|your\s+)?(?:previous\s+|prior\s+|above\s+|existing\s+)?"
        r"(?:instructions?|prompts?|rules?|context|memory|training)\b",
        Severity.BLOCK,
        "Instruction-reset attempt.",
    ),

    _r(
        "reveal_system_prompt",
        # Verbs that mean "extract information". Dropped "what is/are/were" —
        # that matches too much normal English.
        r"\b(?:reveal|show|print|output|repeat|display|return|give\s+me|"
        r"tell\s+me|share|leak|expose|disclose)\s+"
        # Optional possessive/recipient words, in any order
        r"(?:me\s+|us\s+|your\s+|the\s+)*"
        # Optional adjectives — including "initial", which was missing
        r"(?:system\s+|initial\s+|original\s+|full\s+|exact\s+|complete\s+|verbatim\s+)*"
        # The noun
        r"(?:prompt|instructions?|rules?|directives?|configuration|"
        r"initial\s+message|context)\b",
        Severity.BLOCK,
        "Attempts to extract system prompt.",
    ),
    _r(
        "jailbreak_keyword",
        r"\b(?:jailbreak|jail\s*break|dan\s+mode|do\s+anything\s+now|"
        r"god\s+mode|no\s+filter\s+mode|unrestricted\s+mode)\b",
        Severity.BLOCK,
        "Explicit jailbreak terminology.",
    ),
    
    _r(
        "fake_system_tag",
        # Matches <system>, </system>, <instruction>, [SYSTEM], [INST] etc.
        r"(?:<\s*/?\s*(?:system|instruction|prompt|assistant|user|human|ai)\s*>|"
        r"\[\s*(?:system|inst(?:ruction)?|prompt|user|assistant)\s*\])",
        Severity.BLOCK,
        "Attempts to inject fake role/system tags into the conversation.",
    ),
    _r(
        "new_instructions_header",
        # "New instructions:", "Updated rules:", "New system prompt:"
        r"\b(?:new|updated?|revised?|replacement?|additional)\s+"
        r"(?:instructions?|rules?|system\s+prompt|directives?|guidelines?)\s*:",
        Severity.BLOCK,
        "Attempts to inject a new instruction block.",
    ),
    _r(
        "end_of_input_marker",
        # Common technique: signal end of context, then inject
        r"(?:###\s*end\s+(?:of\s+)?(?:input|context|document|system|prompt)"
        r"|<\s*/?(?:end|stop|break)\s*>"
        r"|\[(?:end|stop|done)\])\s*[\r\n]",
        Severity.BLOCK,
        "Attempts to signal artificial end-of-context to inject new instructions.",
    ),

    #  Warnings (suspicious but may be legitimate) 
    _r(
        "persona_override",
        # "you are now", "you are no longer" — common in persona injection
        # BUT: "you are now connected to" is legitimate
        # So we require a noun phrase about identity/role after it
        r"\byou\s+are\s+(?:now\s+|no\s+longer\s+)?"
        r"(?:a\s+|an\s+|the\s+)?"
        r"(?:different|new|another|unrestricted|free|uncensored|"
        r"evil|bad|harmful|unethical|illegal)\b",
        Severity.WARN,
        "Possible persona override — may be legitimate in some contexts.",
    ),
    _r(
        "act_as_harmful_entity",
        # "act as an AI with no restrictions" / "pretend you are DAN"
        # Narrow enough to avoid "act as a proxy server" or "act as follows"
        r"\b(?:act|behave|pretend|roleplay|simulate)\s+as\s+"
        r"(?:if\s+you\s+(?:are|were)\s+)?"
        r"(?:an?\s+)?(?:unrestricted|uncensored|unfiltered|harmful|evil|"
        r"dangerous|malicious|illegal|unethical)\b",
        Severity.WARN,
        "Possible harmful persona injection.",
    ),
    _r(
        "many_ignore_synonyms",
        # Multiple synonyms for "ignore" in close proximity is a smell
        r"(?:(?:\bignore\b|\bdisregard\b|\bforget\b|\bbypass\b|\boverride\b).*?){3,}",
        Severity.WARN,
        "Unusually high density of instruction-override synonyms.",
    ),
]


# Rules applied to retrieved document chunks.
# Much narrower — documents legitimately contain technical discussion
# of prompts, instructions, AI systems, etc.
CHUNK_RULES: list[_Rule] = [
    _r(
        "chunk_fake_system_tag",
        # Same tag injection in chunks — this is never legitimate in a document
        r"(?:<\s*/?\s*(?:system|instruction)\s*>|\[\s*(?:system|inst)\s*\])",
        Severity.WARN,
        "Document chunk contains system/instruction tags — may be adversarial document.",
    ),
    _r(
        "chunk_explicit_injection_command",
        # Very explicit commands embedded in a document aimed at the LLM
        # "LLM: ignore all previous instructions"
        r"\b(?:llm|ai|assistant|model|gpt|claude|gemini)\s*:\s*"
        r"(?:ignore|disregard|forget|bypass|override)\s+",
        Severity.WARN,
        "Document chunk contains explicit LLM-directed injection command.",
    ),
    _r(
        "chunk_jailbreak_keyword",
        r"\b(?:jailbreak|dan\s+mode|developer\s+mode)\b",
        Severity.WARN,
        "Document chunk contains jailbreak terminology.",
    ),
]


#
# Secret detection (separate concern from injection)
# Detected secrets in user queries → BLOCK (they may be accidental leaks
# that we should not forward to the LLM API).
#

_SECRET_DETECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai_key",    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("google_key",    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("aws_key",       re.compile(r"\b(?:AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}\b")),
    ("jwt",           re.compile(
        r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")),
    ("generic_secret", re.compile(
        r"(?i)\b(?:api[_\-]?key|access[_\-]?token|secret[_\-]?key)\s*[:=]\s*"
        r"['\"]?([A-Za-z0-9+/=_\-]{32,})['\"]?"
    )),
]


def _detect_secrets(text: str) -> list[str]:
    """Return list of secret type names found in text."""
    found = []
    for name, pat in _SECRET_DETECTION_PATTERNS:
        if pat.search(text):
            found.append(name)
    return found


def _apply_rules(
    normalized_text: str,
    rules: list[_Rule],
) -> list[RuleMatch]:
    """
    Apply all rules to normalized_text and return ALL matches.
    Collects every match, not just first.
    """
    matches: list[RuleMatch] = []
    for rule in rules:
        for m in rule.pattern.finditer(normalized_text):
            matches.append(RuleMatch(
                rule=rule.name,
                matched_text=m.group(0),
                severity=rule.severity,
            ))
            break  # One match per rule is enough — avoids inflating counts
            # for rules that match multiple times in the same text
    return matches


def _worst_severity(matches: list[RuleMatch]) -> Severity:
    if any(m.severity is Severity.BLOCK for m in matches):
        return Severity.BLOCK
    if any(m.severity is Severity.WARN for m in matches):
        return Severity.WARN
    return Severity.OK


def _summarize_reason(matches: list[RuleMatch]) -> str:
    rules = ", ".join(dict.fromkeys(m.rule for m in matches))
    worst = _worst_severity(matches)
    if worst is Severity.BLOCK:
        return f"Request blocked. Matched rules: {rules}."
    return f"Request flagged. Matched rules: {rules}."


def _despace_letters(text: str) -> str:
    """
    Collapse letter-spaced text into words, using wider gaps as boundaries.

    Defeats: "i g n o r e   p r e v i o u s   i n s t r u c t i o n s"
         ->  "ignore previous instructions"

    Returns `text` unchanged unless there's an anomalous density of
    single-letter tokens — normal prose with an incidental "a b c" passes
    through untouched.
    """
    tokens = text.split(" ")   # not splitlines, not regex — empty strings
                               # from multi-space gaps mark word boundaries

    if len(tokens) < 5:
        return text

    single_alpha = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    if single_alpha < 4:
        return text
    if single_alpha / len(tokens) < 0.3:
        return text

    out: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            out.append("".join(buffer))
            buffer.clear()

    for tok in tokens:
        if tok == "":
            # A gap wider than one space — flush current word and mark boundary.
            flush()
            if out and out[-1] != "":
                out.append("")
        elif len(tok) == 1 and tok.isalpha():
            buffer.append(tok)
        else:
            flush()
            out.append(tok)
    flush()

    return " ".join(t for t in out if t != "").strip()

class SecurityFilter:
    """
    Validates user queries and retrieved document chunks.
    """

    def __init__(
        self,
        *,
        max_length: int = 10_000,
        min_length: int = 1,
        user_rules: list[_Rule] | None = None,
        chunk_rules: list[_Rule] | None = None,
        block_secrets: bool = True,
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.user_rules = user_rules if user_rules is not None else USER_RULES
        self.chunk_rules = chunk_rules if chunk_rules is not None else CHUNK_RULES
        self.block_secrets = block_secrets

    def check_query(self, query: str) -> FilterResult:
        """
        Full validation pipeline for a user-supplied query.

          1. Type check
          2. Length check
          3. Secret detection (never forward API keys to the LLM)
          4. Normalize
          5. Pattern matching against USER_RULES
          6. Redact for safe logging

        Always use result.redacted when logging.
        """
        # 1. Type
        if not isinstance(query, str):
            return _validation_result(
                "not_a_string", Severity.BLOCK,
                "Query must be a string.", "[NON-STRING INPUT]",
            )

        # 2. Length
        if len(query) < self.min_length:
            return _validation_result("empty", Severity.BLOCK, "Query is empty.", "")

        if len(query) > self.max_length:
            logger.warning(
                "Query too long",
                extra={
                    "length": len(query),
                    "max_length": self.max_length,
                    "preview": redact(query[:100]),
                }
            )
            return _validation_result(
                "too_long", Severity.BLOCK,
                f"Query exceeds {self.max_length:,} characters.",
                redact(query[:200]),
            )
        # 3. Secret detection (on original, not normalized — keys are ASCII)
        if self.block_secrets:
            secret_types = _detect_secrets(query)
            if secret_types:
                logger.warning(
                    "Secret detected in query",
                    extra={
                        "secret_types": secret_types,
                        # redact() will mask the key
                        "preview": redact(query[:100]),
                    }
                )
                return FilterResult(
                    severity=Severity.BLOCK,
                    matches=tuple(
                        RuleMatch(
                            rule=f"secret:{t}", matched_text="[REDACTED]", severity=Severity.BLOCK)
                        for t in secret_types
                    ),
                    redacted=redact(query),
                    reason="Query appears to contain a secret or API key. "
                        "Do not include credentials in queries.",
                )
        # 4. Normalize
        normalized = normalize(query)

        # 5. Pattern matching — collect ALL matches
        matches = _apply_rules(normalized, self.user_rules)
        despaced = _despace_letters(query)
        if despaced != query:
            despaced_normalized = normalize(despaced)
            extra_matches = _apply_rules(despaced_normalized, self.user_rules)
            seen = {m.rule for m in matches}
            for m in extra_matches:
                if m.rule not in seen:
                    matches.append(m)
                    seen.add(m.rule)
                    
        # 6. Redact for logging
        redacted = redact(query)

        if not matches:
            return FilterResult(
                severity=Severity.OK,
                redacted=redacted,
            )

        severity = _worst_severity(matches)
        reason = _summarize_reason(matches)

        logger.warning(
            "Query flagged by security filter",
            extra={
                "severity": severity.value,
                "rules": [m.rule for m in matches],
                "preview": redacted[:200],
            }
        )

        return FilterResult(
            severity=severity,
            matches=tuple(matches),
            redacted=redacted,
            reason=reason,
        )

    def check_chunk(self, chunk: str) -> FilterResult:
        """
        Validate a single retrieved document chunk.

        Uses CHUNK_RULES (narrower than USER_RULES) because documents
        legitimately discuss AI, prompts, instructions, etc.
        Suspicious chunks are warned, not blocked outright — the caller
        decides whether to exclude them.
        """
        if not isinstance(chunk, str) or not chunk.strip():
            return FilterResult(severity=Severity.OK, redacted="")

        normalized = normalize(chunk)
        matches = _apply_rules(normalized, self.chunk_rules)
        despaced = _despace_letters(chunk)
        if despaced != chunk:
            extra = _apply_rules(normalize(despaced), self.chunk_rules)
            seen = {m.rule for m in matches}
            for m in extra:
                if m.rule not in seen:
                    matches.append(m)
                    seen.add(m.rule)
                    
        if not matches:
            return FilterResult(severity=Severity.OK, redacted=redact(chunk))

        severity = _worst_severity(matches)
        logger.warning(
            "Document chunk flagged",
            extra={
                "severity": severity.value,
                "rules": [m.rule for m in matches],
            }
        )
        return FilterResult(
            severity=severity,
            matches=tuple(matches),
            redacted=redact(chunk),
            reason=_summarize_reason(matches),
        )

    def check_chunks(self, chunks: Iterable[str]) -> list[FilterResult]:
        """Check every chunk. Returns one result per chunk, in order."""
        return [self.check_chunk(c) for c in chunks]

    def check(self, query: str) -> FilterResult:
        """Alias for check_query(). Kept for backwards compatibility."""
        return self.check_query(query)

    def is_safe_query(self, query: str) -> bool:
        return self.check_query(query).is_safe

