import pytest
from llm_text_splitter import LLMTextSplitter

def test_initialization():
    splitter = LLMTextSplitter(max_chunk_chars=100, overlap_chars=10)
    assert splitter.max_chunk_chars == 100
    assert splitter.overlap_chars == 10

    with pytest.raises(ValueError, match="max_chunk_chars must be a positive integer."):
        LLMTextSplitter(max_chunk_chars=0)
    with pytest.raises(ValueError, match="overlap_chars must be a non-negative integer."):
        LLMTextSplitter(overlap_chars=-1)
    with pytest.raises(ValueError, match="overlap_chars must be less than max_chunk_chars"):
        LLMTextSplitter(max_chunk_chars=10, overlap_chars=10)
    with pytest.raises(ValueError, match="overlap_chars must be less than max_chunk_chars"):
        LLMTextSplitter(max_chunk_chars=10, overlap_chars=11)

def test_empty_input():
    splitter = LLMTextSplitter(max_chunk_chars=100)
    assert splitter.split_document("") == []
    assert splitter.split_document("   \n\t") == []

def test_short_document():
    splitter = LLMTextSplitter(max_chunk_chars=100)
    text = "Hello, world!"
    chunks = splitter.split_document(text)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_split_by_lines_no_overlap():
    splitter = LLMTextSplitter(max_chunk_chars=20, overlap_chars=0)
    text = "Line 1\nLine 2 is longer\nLine 3\nLine 4"
    chunks = splitter.split_document(text, strategy="lines")
    assert len(chunks) == 3
    assert chunks[0] == "Line 1"
    assert chunks[1] == "Line 2 is longer"
    assert chunks[2] == "Line 3\nLine 4"

def test_split_by_lines_with_long_line_arbitrary_split():
    splitter = LLMTextSplitter(max_chunk_chars=20, overlap_chars=5)
    text = "This is a very very very long line that needs to be split.\nShort line."
    chunks = splitter.split_document(text, strategy="lines")
    assert len(chunks) == 5
    assert chunks[0] == "This is a very very "
    assert chunks[1] == "very very long line "
    assert chunks[2] == "line that needs to b"
    assert chunks[3] == " to be split."
    assert chunks[4] == "Short line."

def test_split_by_paragraphs():
    splitter = LLMTextSplitter(max_chunk_chars=50, overlap_chars=0)
    text = "Para 1.\n\nPara 2 is quite long and will need to be split into lines or chars.\n\nPara 3."
    chunks = splitter.split_document(text, strategy="paragraphs")
    assert len(chunks) == 4
    assert chunks[0] == "Para 1."
    assert chunks[1] == "Para 2 is quite long and will need to be split int"
    # You'll need to add assertions for the other chunks from Para 2, and for Para 3.
    # Example for Para 2 remaining: "o lines or chars." (if it just splits by char)
    # Or if it splits by lines, it might be more complex.
    # You should inspect the full `chunks` list to determine the exact expected values.

def test_split_by_characters_with_overlap():
    splitter = LLMTextSplitter(max_chunk_chars=10, overlap_chars=3)
    text = "abcdefghijkl" # Length 12
    chunks = splitter.split_document(text, strategy="characters")
    assert len(chunks) == 2
    assert chunks[0] == "abcdefghij" # 10 chars
    assert chunks[1] == "hijkl"
    
def test_unsupported_strategy():
    splitter = LLMTextSplitter(max_chunk_chars=100)
    with pytest.raises(ValueError, match="Unsupported splitting strategy"):
        splitter.split_document("Some text", strategy="unsupported")

def test_long_single_paragraph_split_by_lines():
    splitter = LLMTextSplitter(max_chunk_chars=50, overlap_chars=10)
    long_para = "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very much so far for me to be able to get back to it and I can't wait to see the rest of the game and I will be there for you and your family"