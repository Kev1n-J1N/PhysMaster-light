import os
import logging
import warnings
from typing import List, Tuple, Any, Optional

import nltk
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===== 配置 =====
SPACY_MODEL = "en_core_web_sm"
PERCENTILE_THRESHOLD = 85
OVERLAP_SIZE = 1
EMBED_BATCH_SIZE = 12
EMBED_MODEL_PATH = "/data/public_model/Qwen3-Embedding-0.6B/"  # 换成你本地的模型路径
TOP_K = 5
MAX_SENTS_PER_CHUNK = 6  # 给稀疏检索准备的一个简单切块长度


# ===== 工具函数 =====
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        logging.info("下载 NLTK punkt ...")
        nltk.download("punkt")


def load_spacy_model(model_name=SPACY_MODEL):
    try:
        return spacy.load(model_name)
    except OSError:
        logging.info(f"spaCy 模型 {model_name} 不存在，正在下载...")
        os.system(f"python -m spacy download {model_name}")
        return spacy.load(model_name)


def read_pdf_to_text(pdf_path: str) -> str:
    import PyPDF2
    parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
    return "\n".join(parts)


def _split_paragraphs(text: str) -> List[str]:
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    processed = []
    current = ""
    for p in raw_paragraphs:
        is_heading = p.startswith("#")
        if is_heading and current.strip():
            processed.append(current.strip())
            current = p
        else:
            current = f"{current}\n\n{p}".strip() if current else p
    if current:
        processed.append(current.strip())
    return processed


def split_doc_into_sentences(text: str, nlp=None) -> Tuple[List[List[str]], List[str]]:
    """
    把文档切成段，再切成句子。
    返回：每段的句子列表 + 全部句子平铺列表
    """
    if not text or not isinstance(text, str):
        return [], []

    paragraphs = _split_paragraphs(text)
    sentences_by_paragraph: List[List[str]] = []
    all_sentences: List[str] = []

    for paragraph in paragraphs:
        if nlp:
            # 优先用 nltk，spaCy 兜底
            try:
                sents = nltk.sent_tokenize(paragraph)
            except Exception:
                doc = nlp(paragraph)
                sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        else:
            sents = nltk.sent_tokenize(paragraph)
        sents = [s for s in sents if s]
        sentences_by_paragraph.append(sents)
        all_sentences.extend(sents)

    return sentences_by_paragraph, all_sentences


# ====== 稠密切块（你的原思路） ======
def _chunk_sentences_from_embeddings(
    sentences: List[str],
    embeddings: np.ndarray,
    percentile_threshold: int,
    overlap_size: int = 1,
) -> List[str]:
    if len(sentences) < 2:
        return [" ".join(sentences).strip()] if sentences else []

    similarities = []
    for i in range(len(embeddings) - 1):
        emb1 = embeddings[i]
        emb2 = embeddings[i + 1]
        denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim = float(np.dot(emb1, emb2) / denom) if denom != 0 else 0.0
        similarities.append(sim)

    distances = 1 - np.asarray(similarities)
    threshold = np.percentile(distances, percentile_threshold)

    split_indices = [i for i, dist in enumerate(distances) if dist > threshold]

    chunks = []
    start_index = 0
    for end_index in split_indices:
        chunk_sentences = sentences[start_index: end_index + 1]
        # 防止第一个块太短
        if len(chunk_sentences) == 1 and start_index == 0:
            if len(chunk_sentences[0].split()) <= 3:
                continue
        chunks.append(" ".join(chunk_sentences).strip())
        start_index = max(0, end_index + 1 - overlap_size)

    if start_index < len(sentences):
        chunks.append(" ".join(sentences[start_index:]).strip())

    return [c for c in chunks if c]


def encode_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def build_dense_chunks_and_embeddings(pdf_path: str, model: SentenceTransformer, nlp) -> Tuple[List[str], np.ndarray]:
    """
    稠密路线：语义分段 + 再对 chunk 做 embedding
    """
    text = read_pdf_to_text(pdf_path)
    if not text.strip():
        logging.warning("PDF 没读到有效文本")
        return [], np.empty((0, model.get_sentence_embedding_dimension()))

    sents_by_para, sents_flat = split_doc_into_sentences(text, nlp=nlp)
    if not sents_flat:
        logging.warning("没切出句子")
        return [], np.empty((0, model.get_sentence_embedding_dimension()))

    # 所有句子一次性 embedding
    all_sent_emb = encode_texts(model, sents_flat)

    chunks: List[str] = []
    cursor = 0
    for para_sents in sents_by_para:
        if not para_sents:
            continue
        cnt = len(para_sents)
        para_emb = all_sent_emb[cursor: cursor + cnt]
        cursor += cnt

        para_chunks = _chunk_sentences_from_embeddings(
            para_sents,
            para_emb,
            percentile_threshold=PERCENTILE_THRESHOLD,
            overlap_size=OVERLAP_SIZE,
        )
        chunks.extend(para_chunks)

    if not chunks:
        return [], np.empty((0, model.get_sentence_embedding_dimension()))

    chunk_embs = encode_texts(model, chunks)
    return chunks, chunk_embs


# ====== 稀疏切块（简单按句数） ======
def _chunk_sentences_by_length(
    sentences_by_paragraph: List[List[str]],
    max_sents_per_chunk: int = MAX_SENTS_PER_CHUNK,
    overlap_size: int = OVERLAP_SIZE,
) -> List[str]:
    chunks: List[str] = []
    for para_sents in sentences_by_paragraph:
        if not para_sents:
            continue

        start = 0
        while start < len(para_sents):
            end = min(start + max_sents_per_chunk, len(para_sents))
            chunk_sents = para_sents[start:end]
            chunk_text = " ".join(chunk_sents).strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end == len(para_sents):
                break
            start = end - overlap_size if end - overlap_size > start else end
    return chunks


def build_sparse_chunks_and_index(pdf_path: str) -> Tuple[List[str], Optional[TfidfVectorizer], Any]:
    """
    稀疏路线：简单切块 + TF-IDF 索引
    """
    text = read_pdf_to_text(pdf_path)
    if not text.strip():
        logging.warning("PDF 没读到有效文本")
        return [], None, None

    sents_by_para, _ = split_doc_into_sentences(text)
    chunks = _chunk_sentences_by_length(sents_by_para)
    if not chunks:
        logging.warning("没有切出 chunk")
        return [], None, None

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return chunks, vectorizer, tfidf_matrix


# ====== 两个检索函数 ======
def search_dense_chunks(query: str, pdf_path: str, top_k: int = TOP_K):
    """
    用稠密检索：适合语义相近但措辞不同的场景
    """
    download_nltk_data()
    nlp = load_spacy_model()

    logging.info(f"加载 embedding 模型: {EMBED_MODEL_PATH}")
    model = SentenceTransformer(
        EMBED_MODEL_PATH,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    try:
        model.half()
    except Exception:
        pass

    chunks, chunk_embs = build_dense_chunks_and_embeddings(pdf_path, model, nlp)
    if len(chunks) == 0:
        logging.warning("没有可检索的 chunk（dense）")
        return []

    query_emb = encode_texts(model, [query])[0]

    chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_emb)
    scores = (chunk_embs @ query_emb) / (chunk_norms[:, 0] * query_norm + 1e-10)

    top_k = min(top_k, len(chunks))
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        if scores[idx] > 0.4:  # 你原来的经验阈值
            results.append({
                "chunk_index": int(idx),
                "score": float(scores[idx]),
                "text": chunks[idx],
            })
    return results


def search_sparse_chunks(query: str, pdf_path: str, top_k: int = TOP_K):
    """
    用稀疏检索：适合关键词、术语能对得上的场景
    """
    download_nltk_data()

    chunks, vectorizer, tfidf_matrix = build_sparse_chunks_and_index(pdf_path)
    if not chunks or vectorizer is None:
        logging.warning("没有可检索的 chunk（sparse）")
        return []

    query_vec = vectorizer.transform([query])
    scores = linear_kernel(query_vec, tfidf_matrix).ravel()

    top_k = min(top_k, len(chunks))
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        if scores[idx] > 0.0:  # 稀疏 0 就说明没字对上
            results.append({
                "chunk_index": int(idx),
                "score": float(scores[idx]),
                "text": chunks[idx],
            })
    return results


# ====== demo ======
if __name__ == "__main__":
    pdf_path = "/data/public_data/arxiv_database/arxiv/pdf/2503/2503.10410v3.pdf"
    query = "uses foreground digital assets to render onto a 2D image for simulation"

    print("====== 稀疏检索 ======")
    sparse_hits = search_sparse_chunks(query, pdf_path, top_k=5)
    for h in sparse_hits:
        print("=" * 50)
        print(f"[sparse] score: {h['score']:.4f}")
        print(h["text"][:400], "...\n")

    print("====== 稠密检索 ======")
    dense_hits = search_dense_chunks(query, pdf_path, top_k=5)
    for h in dense_hits:
        print("=" * 50)
        print(f"[dense] score: {h['score']:.4f}")
        print(h["text"][:400], "...\n")
