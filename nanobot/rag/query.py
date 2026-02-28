"""Query preprocessing for RAG."""

from typing import Set

# Common research term abbreviations
QUERY_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    # English abbreviations
    "nn": ("neural network", "neural networks"),
    "dl": ("deep learning",),
    "ml": ("machine learning",),
    "nlp": ("natural language processing",),
    "llm": ("large language model", "large language models"),
    "cv": ("computer vision",),
    "rl": ("reinforcement learning",),
    "gan": ("generative adversarial network",),
    "cnn": ("convolutional neural network",),
    "rnn": ("recurrent neural network",),
    "lstm": ("long short-term memory",),
    "transformer": ("transformer", "transformer architecture", "attention mechanism"),
    "bert": ("bert", "bidirectional encoder representations from transformers"),
    "gpt": ("gpt", "generative pre-trained transformer"),
    "vae": ("variational autoencoder",),
    "sft": ("supervised fine-tuning",),
    "rlhf": ("reinforcement learning from human feedback",),
    "dpo": ("direct preference optimization",),
    "rag": ("retrieval-augmented generation",),
    "ocr": ("optical character recognition",),
    "api": ("application programming interface",),
    "sdk": ("software development kit",),
    # Chinese terms (for mixed language queries)
    "注意力": ("attention", "attention mechanism"),
    "嵌入": ("embedding", "representation learning"),
    "微调": ("fine-tuning", "fine tuning"),
    "大模型": ("large language model", "llm"),
    "卷积": ("convolution", "convolutional"),
    "循环": ("recurrent", "recurrent neural network"),
    "自编码": ("autoencoder", "variational autoencoder"),
    "对抗": ("adversarial", "generative adversarial network"),
    "强化学习": ("reinforcement learning",),
    "自然语言处理": ("natural language processing", "nlp"),
    "计算机视觉": ("computer vision", "cv"),
}


def expand_query(query: str) -> str:
    """
    Expand abbreviations in query, adding synonyms.

    Args:
        query: Original query string

    Returns:
        Expanded query with synonyms added
    """
    query_lower = query.lower()
    expansions: list[str] = [query]  # Keep original query

    for abbr, expansions_list in QUERY_ABBREVIATIONS.items():
        # Check if abbreviation is in the query (as a whole word)
        import re
        # Use word boundary matching for English abbreviations
        if abbr.isalpha():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, query_lower):
                expansions.extend(expansions_list)
        else:
            # For Chinese characters, use simple substring matching
            if abbr in query_lower:
                expansions.extend(expansions_list)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: list[str] = []
    for exp in expansions:
        if exp not in seen:
            seen.add(exp)
            result.append(exp)

    return " ".join(result)


class QueryExpander:
    """Query expander with config toggle."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def expand(self, query: str) -> str:
        """
        Expand query if enabled.

        Args:
            query: Original query

        Returns:
            Expanded query or original query
        """
        if not self.enabled:
            return query
        return expand_query(query)
