import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import re

def has_image_links(content: str) -> bool:
    """Check if content contains any remaining image links (excluding code blocks)."""
    code_block_pattern = r'```[\s\S]*?```|`[^`]*`'
    content_no_code = re.sub(code_block_pattern, '', content)

    image_patterns = [
        r'!\[\[.*?\]\]',
        r'!\[.*?\]\(.*?\)',
    ]
    return any(re.search(pattern, content_no_code) for pattern in image_patterns)


EXPORT_PATH = Path("obsidian_export.json")

# Files that are pure navigation/index — not worth embedding
SKIP_FILES = ["todos.md", "AI Syllabus.md"]

headers_splits = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_splits,
    strip_headers=False  # Keep headers inside chunk text so context isn't lost
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n```", "\n\n", "\n", " ", ""]
    # \n``` first — tries to split AT code block boundaries before splitting inside them
)

with EXPORT_PATH.open(encoding="utf-8") as f:
    data = json.load(f)

all_chunks = []

for item in data:
    filename = item.get("filename", "<no filename>")
    content = item.get("contents", "")
    last_updated = item.get("last updated date&time", "")
    filepath = item.get("filepath", "")

    content = re.sub(r'!\[\[.*?\]\]', '', content)   # Remove image embeds ![[image.png]]
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content) # Remove markdown images ![alt](url)
    content = re.sub(r'\[\[.*?\]\]', '', content)    # Remove wiki links [[link]]

    if has_image_links(content):
        print(f"Warning: Image links still present in {filename}")

    # Skip empty content
    if not content or not content.strip():
        continue

    # Skip low-value index/navigation files
    if any(skip.lower() in filename.lower() for skip in SKIP_FILES):
        print(f"Skipping index file: {filename}")
        continue

    # Skip files with very little real content (under 100 words)
    if len(content.split()) < 100:
        print(f"Skipping thin file ({len(content.split())} words): {filename}")
        continue

    md_header_chunks = markdown_splitter.split_text(content)
    final_chunks = text_splitter.split_documents(md_header_chunks)

    for idx, chunk in enumerate(final_chunks):
        meta = chunk.metadata

        h1 = meta.get("Header1", "")
        h2 = meta.get("Header2", "")
        h3 = meta.get("Header3", "")

        # Build heading breadcrumb
        heading_parts = [h for h in [h1, h2, h3] if h]
        heading_context = " > ".join(heading_parts) if heading_parts else filename

        # Prepend heading context to the text — LLM reads this during retrieval
        enriched_text = f"[Source: {filename}]\n[Topic: {heading_context}]\n\n{chunk.page_content}"

        rag_chunk = {
            "chunk_id": f"{filename}_{idx}",
            "text": enriched_text,
            "metadata": {
                "filename": filename,
                "source": filepath,
                "last_updated": last_updated,
                "h1": h1,
                "h2": h2,
                "h3": h3,
            }
        }
        all_chunks.append(rag_chunk)

with open("rag_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=4, ensure_ascii=False)

print(f"\nSuccess! Created {len(all_chunks)} RAG-optimized chunks.")