# **LLM Text Splitter**

A lightweight, rule-based text splitter designed for preparing long documents for Large Language Model (LLM) context windows. It intelligently breaks down text into manageable chunks, prioritizing meaningful structural breaks (like paragraphs or lines) before resorting to arbitrary character limits.

## Features

*   **Context-Aware Splitting:** Prioritizes splitting by natural document structures (paragraphs, lines) to maintain semantic coherence.
*   **Arbitrary Fallback:** If natural units are too large, it gracefully falls back to character-based splitting.
*   **Configurable Overlap:** Allows for character overlap between arbitrarily split sub-chunks to preserve context.
*   **Multiple Strategies:** Supports 'paragraphs', 'lines', and 'characters' splitting strategies.
*   **Lightweight:** No external NLP dependencies, relying solely on Python's standard library.

## Installation

You can install `llm-text-splitter` using pip:

```bash
pip install llm-text-splitter
```

## Usage
Here's how to use the LLMTextSplitter in your Python projects:

```python
from llm_text_splitter import LLMTextSplitter

long_document = """
## Introduction to Large Language Models

Large Language Models (LLMs) are a type of artificial intelligence model that have been trained on vast amounts of text data. They are capable of understanding and generating human-like text, making them incredibly versatile for a wide range of applications, from content creation to customer service.

## How LLMs Work

LLMs operate on a transformer architecture, which allows them to process words in relation to all other words in a sequence, rather than one by one. This enables them to grasp context and nuances in language far better than previous models. Training involves predicting the next word in a sentence, which helps them learn grammar, facts, and even some reasoning abilities.

This is a very long paragraph that might exceed the chunk limit on its own. It discusses the various applications of LLMs, including but not limited to, summarization, translation, code generation, creative writing, and question answering. The ability of these models to adapt to different tasks with minimal fine-tuning is what makes them so powerful and revolutionary in the field of AI. Their impact is being felt across industries, transforming how businesses interact with data and customers.

## Challenges and Future

Despite their power, LLMs face challenges such as computational cost, ethical considerations (bias, misinformation), and the need for significant data. Future developments aim to address these, focusing on efficiency, interpretability, and safety.

Another short paragraph.
"""

# --- Example 1: Splitting by Paragraphs (default strategy) ---
# Initialize splitter with a typical LLM context window size (e.g., 500 characters for demo)
# and a small overlap for arbitrary splits.
splitter_para = LLMTextSplitter(max_chunk_chars=500, overlap_chars=50)

print("--- Example 1: Splitting by Paragraphs (default strategy) ---")
print("This strategy attempts to keep paragraphs intact. If a paragraph is too long,")
print("it will then try to split it by lines, and finally by characters.\n")

chunks_by_paragraph = splitter_para.split_document(long_document, strategy="paragraphs")
for i, chunk in enumerate(chunks_by_paragraph):
    print(f"Chunk {i+1} (Length: {len(chunk)} chars):\n---\n{chunk}\n---")
    if i < len(chunks_by_paragraph) - 1:
        print("\n" + "="*40 + "\n") # Separator for readability

print("\n\n" + "#"*60 + "\n\n") # Major section separator

# --- Example 2: Splitting by Lines ---
splitter_lines = LLMTextSplitter(max_chunk_chars=200, overlap_chars=20)

print("--- Example 2: Splitting by Lines ---")
print("This strategy prioritizes splitting by newlines. If a line is too long,")
print("it will then split it arbitrarily by characters.\n")

chunks_by_line = splitter_lines.split_document(long_document, strategy="lines")
for i, chunk in enumerate(chunks_by_line):
    print(f"Chunk {i+1} (Length: {len(chunk)} chars):\n---\n{chunk}\n---")
    if i < len(chunks_by_line) - 1:
        print("\n" + "="*40 + "\n")

print("\n\n" + "#"*60 + "\n\n") # Major section separator

# --- Example 3: Splitting by Characters (arbitrary splitting) ---
splitter_chars = LLMTextSplitter(max_chunk_chars=100, overlap_chars=10)

print("--- Example 3: Splitting by Characters (arbitrary splitting) ---")
print("This strategy performs purely character-based splitting with overlap.")
print("Useful when no natural breaks are desired or available.\n")

chunks_by_char = splitter_chars.split_document(long_document, strategy="characters")
for i, chunk in enumerate(chunks_by_char):
    print(f"Chunk {i+1} (Length: {len(chunk)} chars):\n---\n{chunk}\n---")
    if i < len(chunks_by_char) - 1:
        print("\n" + "="*40 + "\n")

print("\n\n" + "#"*60 + "\n\n") # Major section separator

# --- Example 4: Handling a very long single line/paragraph with forced arbitrary splits ---
print("--- Example 4: Handling a very long single line/paragraph with forced arbitrary splits ---")
print("This demonstrates how the splitter breaks down content that lacks natural separators,")
print("ensuring chunks respect `max_chunk_chars` while using `overlap_chars`.\n")

very_long_line = "This is an extremely long sentence that will definitely exceed the chunk limit. It's designed to show how the splitter handles text that cannot be broken down by natural separators like paragraphs or short lines. The goal is to ensure no part of the text is lost, even if it means splitting mid-sentence. This is crucial for maintaining data integrity when feeding large documents into LLMs with strict context window constraints." * 3
splitter_long = LLMTextSplitter(max_chunk_chars=150, overlap_chars=20)
chunks_long_line = splitter_long.split_document(very_long_line, strategy="characters") # 'characters' best illustrates arbitrary split
for i, chunk in enumerate(chunks_long_line):
    print(f"Chunk {i+1} (Length: {len(chunk)} chars):\n---\n{chunk}\n---")
    if i < len(chunks_long_line) - 1:
        print("\n" + "="*40 + "\n")
```

