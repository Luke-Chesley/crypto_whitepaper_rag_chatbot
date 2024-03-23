from tqdm import tqdm
import torch


def embed(model, pages_and_chunks, device):
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
    text_chunk_embeddings = []

    with tqdm(total=len(text_chunks), desc="Embedding", unit="chunk") as pbar:
        for i in range(0, len(text_chunks), 32):
            batch = text_chunks[i : i + 32]
            batch_embeddings = model.encode(
                batch, batch_size=32, convert_to_tensor=True, device=device
            )
            text_chunk_embeddings.extend(batch_embeddings.tolist())
            pbar.update(len(batch))

    text_chunk_embeddings = torch.tensor(text_chunk_embeddings)
    return text_chunk_embeddings