import fitz
from spacy.lang.en import English
import pandas as pd
import re
import os
import math


def text_formatter(text: str) -> str:
    """Minor formatting on text"""
    cleaned_text = text.replace("\n", " ").strip()

    # more

    return cleaned_text


def open_read_pdf(pdf_path: str) -> list[dict]:
    pages_and_text = []

    file_name = os.path.basename(pdf_path)

    if pdf_path.endswith(".pdf"):
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            text = text_formatter(text=text)
            pages_and_text.append(
                {
                    "file_name": file_name,
                    "page_number": i,
                    "page_char_count": len(text),
                    "page_word_count": len(text.split(" ")),
                    "page_sent_count_raw": len(text.split(". ")),
                    "page_token_count": (len(text) / 4),
                    "text": text,
                }
            )
    elif pdf_path.endswith(".txt"):
        with open(pdf_path, "r") as file:
            text = file.read()
            text = text_formatter(text=text)
            words = text.split(" ")
            page_size = 100  # Number of words per page
            num_pages = math.ceil(len(words) / page_size)

            for i in range(num_pages):
                start_index = i * page_size
                end_index = min((i + 1) * page_size, len(words))
                page_words = words[start_index:end_index]
                page_text = " ".join(page_words)

                pages_and_text.append(
                    {
                        "file_name": file_name,
                        "page_number": i,
                        "page_char_count": len(page_text),
                        "page_word_count": len(page_words),
                        "page_sent_count_raw": len(page_text.split(". ")),
                        "page_token_count": (len(page_text) / 4),
                        "text": page_text,
                    }
                )

    return pages_and_text


def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [
        input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)
    ]


def preprocess(path):
    pages = open_read_pdf(path)

    nlp = English()

    nlp.add_pipe("sentencizer")

    for item in pages:
        item["sentences"] = list(nlp(item["text"]).sents)

        item["sentences"] = [str(s) for s in item["sentences"]]

        item["page_sentence_count"] = len(item["sentences"])

    chunk_size = 5

    for item in pages:
        item["sentence_chunks"] = split_list(item["sentences"], slice_size=chunk_size)

        item["num_chunks"] = len(item["sentence_chunks"])

    chunks = []

    for item in pages:
        for sentence_chunk in item["sentence_chunks"]:
            cd = {}
            cd["page_number"] = item["page_number"]

            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(
                r"\.([A-Z])", r". \1", joined_sentence_chunk
            )  # ".A" -> ". A" for any full-stop/capital letter combo

            cd["sentence_chunk"] = f"{path}:{joined_sentence_chunk}"

            cd["chunk_char_count"] = len(joined_sentence_chunk)
            cd["chunk_word_count"] = len(
                [word for word in joined_sentence_chunk.split(" ")]
            )
            cd["chunk_token_count"] = (
                len(joined_sentence_chunk) / 4
            )  # 1 token = ~4 characters

            cd['file_name'] = path 
            chunks.append(cd)

    df = pd.DataFrame(chunks)

    min_token_len = 30
    pages_and_chunks_over_min_token_len = df[
        df["chunk_token_count"] > min_token_len
    ].to_dict(orient="records")

    return pages_and_chunks_over_min_token_len
