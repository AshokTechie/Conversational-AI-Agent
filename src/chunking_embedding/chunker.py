import re
from typing import List, Dict, Any
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class DocumentChunkerNode:
    """
    Decoupled Node 2 (Part A): Chunking Node.
    Recursively splits raw text pages into semantically cohesive chunks
    while retaining page-number provenance and document metadata.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        """
        Recursively splits text using hierarchical separators.
        """
        final_chunks = []
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # Find best separator
        separator = self.separators[-1]
        for s in self.separators:
            if s == "":
                separator = ""
                break
            if s in text:
                separator = s
                break

        splits = text.split(separator) if separator else list(text)

        current_chunk = []
        current_len = 0

        for part in splits:
            part_len = len(part) + len(separator)
            if current_len + part_len > self.chunk_size:
                if current_chunk:
                    joined = separator.join(current_chunk).strip()
                    if joined:
                        final_chunks.append(joined)
                    # Handle overlap by keeping the trailing items
                    overlap_tokens = []
                    overlap_len = 0
                    for prev in reversed(current_chunk):
                        if overlap_len + len(prev) <= self.chunk_overlap:
                            overlap_tokens.insert(0, prev)
                            overlap_len += len(prev) + len(separator)
                        else:
                            break
                    current_chunk = overlap_tokens
                    current_len = overlap_len
            current_chunk.append(part)
            current_len += part_len

        if current_chunk:
            joined = separator.join(current_chunk).strip()
            if joined:
                final_chunks.append(joined)

        return final_chunks

    def chunk_documents(self, raw_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks a list of ingested pages into vector-ready chunk dictionaries.
        """
        chunked_output = []
        global_chunk_idx = 0

        for page in raw_pages:
            text = page.get("text", "")
            page_num = page.get("page_number", 1)
            doc_id = page.get("doc_id", "unknown")
            source = page.get("source", "unknown")

            sub_chunks = self._split_text(text)
            for idx, chunk_text in enumerate(sub_chunks):
                chunked_output.append({
                    "chunk_id": f"{doc_id}_p{page_num}_c{idx}",
                    "global_index": global_chunk_idx,
                    "text": chunk_text,
                    "page_number": page_num,
                    "doc_id": doc_id,
                    "source": source,
                    "char_count": len(chunk_text),
                    "metadata": {
                        "chunk_index_in_page": idx,
                        "page_number": page_num,
                        "source": source
                    }
                })
                global_chunk_idx += 1

        return chunked_output
