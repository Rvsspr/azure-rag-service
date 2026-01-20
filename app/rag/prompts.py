from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

"""
app/rag/prompts.py

Utilities to build prompts/messages for a retrieval-augmented generation (RAG) workflow.

This module provides small, well-documented helpers to:
- Represent retrieved document contexts
- Assemble a system prompt
- Combine contexts and a user question into a chat-style messages payload
suitable for use with Azure OpenAI chat completions or similar APIs.
"""



@dataclass
class DocumentContext:
    """
    Represents a single retrieved document/context to be injected into the prompt.

    Attributes:
        id: an identifier or source string for citation (e.g., URL or filename)
        content: the textual content of the retrieved chunk
        metadata: optional extra metadata (e.g., chunk index, vector score)
    """
    id: str
    content: str
    metadata: Optional[Dict] = None


# Default system prompt instructing the model about its role.
DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant that answers questions using only the provided supporting "
    "documents. Use the documents to form an accurate answer and cite the document "
    "ids in square brackets (e.g., [doc1]). If the answer is not contained in the "
    "documents, say you don't know and do not hallucinate."
)

# Template used to wrap each context before insertion.
_CONTEXT_WRAP_TEMPLATE = "[{id}]\n{content}"


def _format_context(ctx: DocumentContext) -> str:
    """Format a single DocumentContext for inclusion in the prompt."""
    return _CONTEXT_WRAP_TEMPLATE.format(id=ctx.id, content=ctx.content.strip())


def _estimate_char_limit(texts: Sequence[str]) -> int:
    """
    Simple estimator helper — here we just use character length.
    Kept separate in case more advanced token estimation is added later.
    """
    return sum(len(t) for t in texts)


def assemble_context_block(
    contexts: Iterable[DocumentContext],
    *,
    max_chars: int = 3000,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    Convert an iterable of DocumentContext into a single block suitable for the prompt,
    truncating the set of contexts to remain under max_chars (simple greedy packing).

    Args:
        contexts: sequence of DocumentContext to include
        max_chars: maximum total characters for the returned block
        separator: string between context items

    Returns:
        A string containing formatted contexts concatenated by separator.
    """
    formatted: List[str] = []
    total = 0
    for ctx in contexts:
        piece = _format_context(ctx)
        piece_len = len(piece)
        # if single piece exceeds max_chars, include a truncated version of it
        if piece_len > max_chars and not formatted:
            truncated = piece[: max_chars - 100] + "\n\n...[truncated]"
            formatted.append(truncated)
            total = len(truncated)
            break
        if total + piece_len + len(separator) > max_chars:
            break
        formatted.append(piece)
        total += piece_len + len(separator)
    return separator.join(formatted)


def build_messages(
    question: str,
    contexts: Iterable[DocumentContext],
    *,
    system_prompt: Optional[str] = None,
    max_context_chars: int = 3000,
) -> List[Dict[str, str]]:
    """
    Build a list of chat messages suitable for a chat-completion API.

    The returned list has two messages:
    - system: instructions describing behavior
    - user: the question plus the provided context block

    Args:
        question: the user question to answer
        contexts: retrieved contexts/documents to provide as evidence
        system_prompt: optional override of the default system prompt
        max_context_chars: max characters of contexts to include (simple heuristic)

    Returns:
        A list of message dicts: [{'role': 'system', 'content': ...}, {'role': 'user', 'content': ...}]
    """
    sys = system_prompt or DEFAULT_SYSTEM_PROMPT
    context_block = assemble_context_block(contexts, max_chars=max_context_chars)
    user_parts = [
        "Use only the following documents to answer the question. Cite documents by their ids.",
        "",
        "DOCUMENTS:",
        context_block or "[no documents provided]",
        "",
        "QUESTION:",
        question.strip(),
        "",
        "If the answer cannot be found in the documents, respond with: \"I don't know.\"",
    ]
    user_content = "\n".join(user_parts)
    return [{"role": "system", "content": sys}, {"role": "user", "content": user_content}]


# Convenience: a small example generator for tests or quick usage.
def example_messages_for_debug() -> List[Dict[str, str]]:
    """
    Return a minimal example messages payload for local debugging.
    """
    docs = [
        DocumentContext(id="doc1", content="Azure provides cloud computing services including VMs and managed databases."),
        DocumentContext(id="doc2", content="RAG stands for retrieval-augmented generation."),
    ]
    return build_messages("What is RAG and who provides cloud VMs?", docs)


__all__ = [
    "DocumentContext",
    "DEFAULT_SYSTEM_PROMPT",
    "assemble_context_block",
    "build_messages",
    "example_messages_for_debug",
]