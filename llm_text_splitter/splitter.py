import math
from typing import List, Callable, Optional

class LLMTextSplitter:
    """
    A lightweight, rule-based text splitter designed for preparing long documents
    for Large Language Model (LLM) context windows. It prioritizes keeping
    "meaningful chunks" together based on common document structures (e.g.,
    paragraphs, lines) and then splits further if needed, applying overlap
    only within arbitrarily split sub-chunks.

    Attributes:
        max_chunk_chars (int): The maximum number of characters allowed in a single chunk.
                               This should typically align with your LLM's context window.
        overlap_chars (int): The number of characters to overlap between arbitrary sub-chunks
                             when a larger unit (like a paragraph or line) needs to be broken down.
                             This helps maintain context across splits.
    """

    def __init__(self, max_chunk_chars: int = 1000, overlap_chars: int = 0):
        if not isinstance(max_chunk_chars, int) or max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be a positive integer.")
        if not isinstance(overlap_chars, int) or overlap_chars < 0:
            raise ValueError("overlap_chars must be a non-negative integer.")
        if overlap_chars >= max_chunk_chars:
            raise ValueError("overlap_chars must be less than max_chunk_chars to ensure progress.")

        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def _split_arbitrary_chunk(self, text: str) -> List[str]:
        """
        Splits a single, very long string arbitrarily based on max_chunk_chars and overlap.
        This is used when a natural unit (like a line or paragraph) exceeds the max_chunk_chars.
        """
        if len(text) <= self.max_chunk_chars:
            return [text]

        chunks = []
        # Calculate step size: max_chunk_chars minus overlap. Ensure it's at least 1.
        step = max(1, self.max_chunk_chars - self.overlap_chars)

        for i in range(0, len(text), step):
            # Determine the end of the current chunk
            chunk_end = min(i + self.max_chunk_chars, len(text))
            
            # Extract the chunk. The 'i' already handles the stepping,
            # so the start of the chunk is 'i'.
            chunk = text[i:chunk_end]
            chunks.append(chunk)
            
        return chunks

    def _split_by_delimiter(self, text: str, delimiter: str) -> List[str]:
        """
        Splits text by a given delimiter (e.g., '\n', '\n\n').
        Then, it further processes these segments to ensure they fit within max_chunk_chars,
        using arbitrary splitting with overlap if necessary.
        """
        if not text:
            return []

        # Split the text by the primary delimiter
        segments = [s.strip() for s in text.split(delimiter) if s.strip()]
        
        chunks: List[str] = []
        current_chunk_buffer: List[str] = []
        current_buffer_len: int = 0

        for segment in segments:
            segment_len = len(segment)

            # If the segment itself is too long, split it arbitrarily
            if segment_len > self.max_chunk_chars:
                if current_chunk_buffer:
                    chunks.append(delimiter.join(current_chunk_buffer))
                    current_chunk_buffer = []
                    current_buffer_len = 0
                chunks.extend(self._split_arbitrary_chunk(segment))
            # If adding the segment to the current buffer would exceed max_chunk_chars
            elif current_buffer_len + segment_len + len(delimiter) > self.max_chunk_chars:
                if current_chunk_buffer: # Finalize the current buffer
                    chunks.append(delimiter.join(current_chunk_buffer))
                current_chunk_buffer = [segment] # Start a new buffer with this segment
                current_buffer_len = segment_len
            # Otherwise, add the segment to the current buffer
            else:
                current_chunk_buffer.append(segment)
                current_buffer_len += segment_len + len(delimiter) # Account for delimiter when joining

        # Add any remaining content in the buffer
        if current_chunk_buffer:
            chunks.append(delimiter.join(current_chunk_buffer))
            
        # Clean up any empty chunks that might result from splitting
        return [chunk for chunk in chunks if chunk.strip()]

    def split_document(self, text: str, strategy: str = "paragraphs") -> List[str]:
        """
        Splits a document based on a chosen strategy.

        Args:
            text (str): The input document text to split.
            strategy (str): The primary splitting strategy.
                            'paragraphs': Splits by double newlines ('\\n\\n') first,
                                          then by single newlines ('\\n') if paragraphs are too long,
                                          then arbitrarily.
                            'lines': Splits by single newlines ('\\n') first, then arbitrarily.
                            'characters': Splits directly by characters, applying overlap.

        Returns:
            List[str]: A list of text chunks, each not exceeding max_chunk_chars.

        Raises:
            ValueError: If an unsupported splitting strategy is provided.
        """
        if not isinstance(text, str):
            raise TypeError("Input 'text' must be a string.")
        
        if not text.strip():
            return [] # Return empty list for empty or whitespace-only input

        if strategy == "paragraphs":
            # First, try to split by paragraphs
            chunks = self._split_by_delimiter(text, '\n\n')
            
            # Now, iterate through these paragraph-level chunks.
            # If any are still too long (e.g., a single very long paragraph),
            # split them further by lines.
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > self.max_chunk_chars:
                    final_chunks.extend(self._split_by_delimiter(chunk, '\n'))
                else:
                    final_chunks.append(chunk)
            return final_chunks
            
        elif strategy == "lines":
            return self._split_by_delimiter(text, '\n')
            
        elif strategy == "characters":
            return self._split_arbitrary_chunk(text)
            
        else:
            raise ValueError(f"Unsupported splitting strategy: '{strategy}'. "
                             "Choose from 'paragraphs', 'lines', or 'characters'.")
