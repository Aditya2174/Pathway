# import unicodedata
# import pathway as pw
# from pathway.optional_import import optional_imports

# @pw.udf
# def null_splitter(txt: str) -> list[tuple[str, dict]]:
#     """A splitter which returns its argument as one long text ith null metadata.

#     Args:
#         txt: text to be split

#     Returns:
#         list of pairs: chunk text and metadata.

#     The null splitter always return a list of length one containing the full text and empty metadata.
#     """
#     return [(txt, {})]


# def _normalize_unicode(text: str):
#     """Normalize Unicode characters."""
#     return unicodedata.normalize("NFKC", text)


# class BaseTokenSplitter(pw.UDF):
#     """Base class for token-based text splitting."""

#     CHARS_PER_TOKEN = 3
#     PUNCTUATION = [".", "?", "!", "\n"]

#     def __init__(self, encoding_name: str = "cl100k_base"):
#         with optional_imports("xpack-llm"):
#             import tiktoken  # noqa:F401

#         super().__init__()
#         self.encoding_name = encoding_name
#         self.tokenizer = tiktoken.get_encoding(encoding_name)

#     def _tokenize(self, text: str):
#         """Tokenize and normalize the text."""
#         text = _normalize_unicode(text)
#         return self.tokenizer.encode_ordinary(text)

#     def __call__(self, text: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
#         """Split given strings into smaller chunks.

#         Args:
#             text (ColumnExpression[str]): Column with texts to be split.
#             **kwargs: override for defaults set in the constructor.
#         """
#         return super().__call__(text, **kwargs)


# class DefaultTokenCountSplitter(BaseTokenSplitter):
#     """
#     Splits text into chunks based on min and max token limits.
#     """

#     def __init__(self, min_tokens: int = 50, max_tokens: int = 500, encoding_name: str = "cl100k_base"):
#         super().__init__(encoding_name)
#         self.min_tokens = min_tokens
#         self.max_tokens = max_tokens

#     def __wrapped__(self, txt: str) -> list[tuple[str, dict]]:
#         tokens = self._tokenize(txt)
#         output = []
#         i = 0

#         while i < len(tokens):
#             chunk_tokens = tokens[i : i + self.max_tokens]
#             chunk = self.tokenizer.decode(chunk_tokens)
#             last_punctuation = max([chunk.rfind(p) for p in self.PUNCTUATION], default=-1)

#             if last_punctuation != -1 and last_punctuation > self.CHARS_PER_TOKEN * self.min_tokens:
#                 chunk = chunk[: last_punctuation + 1]

#             i += len(self._tokenize(chunk))
#             output.append((chunk, {}))

#         return output
# class SlidingWindowSplitter(BaseTokenSplitter):
#     """
#     Splits text into overlapping chunks with a sliding window.
#     """

#     def __init__(self, max_tokens: int = 500, overlap_tokens: int = 20, encoding_name: str = "cl100k_base"):
#         super().__init__(encoding_name)
#         self.max_tokens = max_tokens
#         self.overlap_tokens = overlap_tokens

#     def __wrapped__(self, txt: str) -> list[tuple[str, dict]]:
#         tokens = self._tokenize(txt)
#         output = []
#         i = 0

#         while i < len(tokens):
#             end = min(i + self.max_tokens, len(tokens))
#             chunk_tokens = tokens[i:end]
#             chunk = self.tokenizer.decode(chunk_tokens)
#             output.append((chunk, {}))
#             i += self.max_tokens - self.overlap_tokens

#         return output

# class SmallToBigSplitter(BaseTokenSplitter):
#     """
#     Splits text into small and large chunks for small-to-big chunking.
#     """

#     def __init__(self, small_chunk_size: int = 175, large_chunk_size: int = 512, overlap_tokens: int = 20, encoding_name: str = "cl100k_base"):
#         super().__init__(encoding_name)
#         self.small_chunk_size = small_chunk_size
#         self.large_chunk_size = large_chunk_size
#         self.overlap_tokens = overlap_tokens

#     def __wrapped__(self, txt: str) -> list[tuple[str, dict]]:
#         tokens = self._tokenize(txt)
#         output = []

#         # Small chunks
#         i = 0
#         while i < len(tokens):
#             end = min(i + self.small_chunk_size, len(tokens))
#             small_chunk_tokens = tokens[i:end]
#             small_chunk = self.tokenizer.decode(small_chunk_tokens)
#             output.append((small_chunk, {"chunk_size": "small"}))
#             i += self.small_chunk_size - self.overlap_tokens

#         # Large chunks
#         i = 0
#         while i < len(tokens):
#             end = min(i + self.large_chunk_size, len(tokens))
#             large_chunk_tokens = tokens[i:end]
#             large_chunk = self.tokenizer.decode(large_chunk_tokens)
#             output.append((large_chunk, {"chunk_size": "large"}))
#             i += self.large_chunk_size - self.overlap_tokens

#         return outputimport unicodedata


import pathway as pw
from pathway.optional_import import optional_imports
from nltk.tokenize import sent_tokenize

@pw.udf
def null_splitter(txt: str) -> list[tuple[str, dict]]:
    """A splitter which returns its argument as one long text with null metadata.

    Args:
        txt: text to be split

    Returns:
        list of pairs: chunk text and metadata.

    The null splitter always returns a list of length one containing the full text and empty metadata.
    """
    return [(txt, {})]


def _normalize_unicode(text: str):
    import unicodedata
    """Normalize Unicode characters."""
    return unicodedata.normalize("NFKC", text)


class BaseSentenceSplitter(pw.UDF):
    """Base class for sentence-based text splitting using NLTK."""

    PUNCTUATION = [".", "?", "!", "\n"]

    def __init__(self):
        super().__init__()

    def _split_sentences(self, text: str):
        """Split and normalize the text into sentences."""
        text = _normalize_unicode(text)
        return sent_tokenize(text)

    def __call__(self, text: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """Split given strings into smaller chunks.

        Args:
            text (ColumnExpression[str]): Column with texts to be split.
            **kwargs: override for defaults set in the constructor.
        """
        return super().__call__(text, **kwargs)


class DefaultSentenceCountSplitter(BaseSentenceSplitter):
    """
    Splits text into chunks based on min and max sentence limits.
    """

    def __init__(self, min_sentences: int = 3, max_sentences: int = 10):
        super().__init__()
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def __wrapped__(self, txt: str) -> list[tuple[str, dict]]:
        sentences = self._split_sentences(txt)
        output = []
        i = 0

        while i < len(sentences):
            chunk = " ".join(sentences[i:i + self.max_sentences])
            i += self.max_sentences

            # Find last punctuation within chunk to split meaningfully
            last_punctuation = max([chunk.rfind(p) for p in self.PUNCTUATION], default=-1)
            if last_punctuation != -1 and len(chunk.split()) > self.min_sentences:
                chunk = chunk[: last_punctuation + 1]

            output.append((chunk, {}))

        return output


class SlidingWindowSentenceSplitter(BaseSentenceSplitter):
    """
    Splits text into overlapping chunks with a sliding window of sentences.
    """

    def __init__(self, max_sentences: int = 10, overlap_sentences: int = 2):
        super().__init__()
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

    def __wrapped__(self, txt: str) -> list[tuple[str, dict]]:
        sentences = self._split_sentences(txt)
        output = []
        i = 0

        while i < len(sentences):
            end = min(i + self.max_sentences, len(sentences))
            chunk = " ".join(sentences[i:end])
            output.append((chunk, {}))
            i += self.max_sentences - self.overlap_sentences

        return output


class SmallToBigSentenceSplitter(BaseSentenceSplitter):
    """
    Splits text into small and large chunks for small-to-big chunking.
    """

    def __init__(self, small_chunk_size: int = 3, large_chunk_size: int = 10, overlap_sentences: int = 2):
        super().__init__()
        self.small_chunk_size = small_chunk_size
        self.large_chunk_size = large_chunk_size
        self.overlap_sentences = overlap_sentences

    def __wrapped__(self, txt: str) -> list[tuple[str, dict]]:
        sentences = self._split_sentences(txt)
        output = []

        # Small chunks
        i = 0
        while i < len(sentences):
            end = min(i + self.small_chunk_size, len(sentences))
            small_chunk = " ".join(sentences[i:end])
            output.append((small_chunk, {"chunk_size": "small"}))
            i += self.small_chunk_size - self.overlap_sentences

        # Large chunks
        i = 0
        while i < len(sentences):
            end = min(i + self.large_chunk_size, len(sentences))
            large_chunk = " ".join(sentences[i:end])
            output.append((large_chunk, {"chunk_size": "large"}))
            i += self.large_chunk_size - self.overlap_sentences

        return output
