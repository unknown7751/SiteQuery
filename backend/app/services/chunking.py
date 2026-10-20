"""Structure-aware chunking for specs, RFIs and minutes."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.config import Settings

# Tried in order: the splitter only falls to the next (finer) separator when a piece is
# still larger than `chunk_size`, so clauses and tables stay whole whenever they fit.
# All are regexes; lookaheads keep the marker ("A.", "-", "3.2.1") attached to its own text.
_SEPARATORS: list[str] = [
    r"\n#{1,6} ",  # markdown headings
    r"\n(?=\s*(?:PART|SECTION|ARTICLE)\s+[\dA-Z])",  # spec divisions: "PART 2 - PRODUCTS"
    r"\n\s*\n",  # paragraph / table-block boundary
    # bullets, "1.", "3.2.1", "a)", "A." clause markers at line start
    r"\n(?=\s*(?:[-*•●▪◦]\s|\d+(?:\.\d+)*[.)]?\s|\(?[A-Za-z]\)\s|[A-Z]\.\s))",
    r"\n",
    r"(?<=[.;])\s+",  # sentence / sub-clause end
    r"\s+",
    r"",
]


def build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
        is_separator_regex=True,
        keep_separator="start",
    )
