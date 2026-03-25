from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent

CFG = {
    "dirs": {
        "source": CURRENT_DIR / "source",
        "out": CURRENT_DIR / "out",
        "knowledge": CURRENT_DIR / "knowledge",
        "index": CURRENT_DIR / "index",
    },
    "embedding": {
        "model": "BAAI/bge-small-en-v1.5",
        "normalize_embeddings": True,
        "batch_size_cuda": 256,
        "batch_size_cpu": 32,
        "show_progress_bar": True,
        "use_fp16_on_cuda": True,
    },
    "chunking": {
        "max_chars": 1200,
        "overlap_chars": 240,
        "min_chars": 180,
    },
    "conversion": {
        "enabled": True,
        "skip_existing_conversion": True,
        "keep_intermediate_outputs": True,
        "mineru_mode": "hybrid_auto",
    },
    "ingest": {
        "skip_existing_sources": True,
    },
    "chunk_id_fmt": "{}:ch{}:sec{}:{:04d}",
    "default_chap_sec": ("0", "0"),
    "target_file": "",
    "supported_exts": [".pdf", ".md", ".txt"],
}


class PriorStore:
    def __init__(self, cfg: Dict | None = None):
        self.cfg = cfg or CFG
        self.dirs = {name: Path(path) for name, path in self.cfg["dirs"].items()}
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        requested_device = os.environ.get("PHY_PRIOR_DEVICE", "").strip().lower()
        if requested_device:
            self.device = requested_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_cfg = self.cfg["embedding"]
        print(f"[*] Initializing embedding model {self.embedding_cfg['model']} on {self.device}...")
        self.emb_model = SentenceTransformer(self.embedding_cfg["model"], device=self.device)
        if self.device.startswith("cuda") and self.embedding_cfg.get("use_fp16_on_cuda", False):
            try:
                self.emb_model.half()
                print("[*] Enabled fp16 for embedding model on CUDA.")
            except Exception as e:
                print(f"[!] Failed to enable fp16, continue with default precision: {e}")

        self.knowledge_path = self.dirs["knowledge"] / "chunks.jsonl"
        self.faiss_path = self.dirs["index"] / "index.faiss"
        self.id_map_path = self.dirs["index"] / "id_map.jsonl"
        self.index_meta_path = self.dirs["index"] / "index_meta.json"
        self.chunks_data: List[Dict] = []
        self.new_chunks_data: List[Dict] = []

    def _embedding_batch_size(self) -> int:
        env_batch = os.environ.get("PHY_PRIOR_EMBED_BATCH")
        if env_batch:
            return max(1, int(env_batch))
        if self.device.startswith("cuda"):
            return int(self.embedding_cfg.get("batch_size_cuda", 256))
        return int(self.embedding_cfg.get("batch_size_cpu", 32))

    def _safe_unlink(self, path: Path):
        if path.exists():
            path.unlink()

    def _load_existing_id_map(self) -> Dict[int, str]:
        if not self.id_map_path.exists():
            return {}
        id_map: Dict[int, str] = {}
        with self.id_map_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if "index_id" in item and "chunk_id" in item:
                    id_map[int(item["index_id"])] = item["chunk_id"]
        return id_map

    def _load_index_meta(self) -> Dict:
        if not self.index_meta_path.exists():
            return {}
        with self.index_meta_path.open("r", encoding="utf-8") as f:
            return json.load(f) or {}

    def _get_id_prefix_from_filename(self, filename: str) -> str:
        name = Path(filename).stem.lower()
        author = re.search(r"[a-z]+", name).group() if re.search(r"[a-z]+", name) else "phys"
        year = re.search(r"\d{4}", name).group() if re.search(r"\d{4}", name) else "2024"
        return f"{author}{year}"

    def _extract_paper_meta_concise(self, elements: List[Dict]) -> Tuple[str, List[str]]:
        title = "Unknown Title"
        authors = ["Unknown Author"]
        title_idx = -1
        for i, element in enumerate(elements):
            if element.get("text_level") == 1:
                title = element.get("text", "").strip()
                title_idx = i
                break
        if title_idx != -1:
            for idx in range(title_idx + 1, min(title_idx + 5, len(elements))):
                raw_text = elements[idx].get("text", "").strip()
                if not raw_text:
                    continue
                clean_text = re.sub(r"[^a-zA-Z\s,]", "", raw_text)
                names = re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", clean_text)
                if names:
                    authors = names
                    break
        return title, authors

    def _extract_keywords_pure(self, text: str) -> List[str]:
        phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
        words = re.findall(r"\b[a-z]{7,}\b", text.lower())
        stopwords = {"following", "between", "results", "through", "using", "shown"}
        common = [word for word, _ in Counter(words).most_common(10) if word not in stopwords]
        return list(dict.fromkeys(phrases + common))[:5]

    def _split_text_with_overlap(self, text: str) -> List[str]:
        clean_text = re.sub(r"\s+", " ", text or "").strip()
        if not clean_text:
            return []
        max_chars = int(self.cfg["chunking"]["max_chars"])
        overlap_chars = int(self.cfg["chunking"]["overlap_chars"])
        min_chars = int(self.cfg["chunking"]["min_chars"])
        if len(clean_text) <= max_chars:
            return [clean_text] if len(clean_text) >= min_chars else []

        sentences = re.split(r"(?<=[\.\!\?\。])\s+", clean_text)
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                if len(current) >= min_chars:
                    chunks.append(current)
                tail = current[-overlap_chars:] if overlap_chars > 0 else ""
                current = f"{tail} {sentence}".strip()
                if len(current) > max_chars:
                    current = current[:max_chars]
            else:
                current = candidate
        if current and len(current) >= min_chars:
            chunks.append(current)
        return chunks

    def _identify_header_strict(self, text: str) -> Tuple[str | None, str, str]:
        clean_text = text.strip()
        sec_match = re.match(r"^(\d+(?:\.\d+)+)", clean_text)
        if sec_match:
            value = sec_match.group(1)
            return "sec", value, value.split(".")[0]
        ch_match = re.match(r"^(?:Chapter\s+)?(\d+)(?!\.)", clean_text, re.I)
        if ch_match:
            value = ch_match.group(1)
            return "ch", value, value
        return None, "0", "0"

    def _load_plain_text_elements(self, path: Path, ext: str) -> List[Dict]:
        elements = []
        with path.open("r", encoding="utf-8") as f:
            for line in f.readlines():
                text = line.strip()
                if not text:
                    continue
                element = {"text": text, "type": "paragraph", "page_idx": 0}
                if ext == ".md" and text.startswith("#"):
                    level = len(re.match(r"^#+", text).group(0))
                    element["text_level"] = level
                    element["type"] = "title"
                    element["text"] = text.lstrip("#").strip()
                elements.append(element)
        if ext == ".txt" and elements:
            elements[0]["text_level"] = 1
            elements[0]["type"] = "title"
        return elements

    def _run_mineru_conversion(self, source_file: Path) -> Path | None:
        if not self.cfg["conversion"]["enabled"]:
            return None

        base_name = source_file.stem
        mode = self.cfg["conversion"]["mineru_mode"]
        json_path = self.dirs["out"] / base_name / mode / f"{base_name}_content_list.json"
        if json_path.exists() and self.cfg["conversion"].get("skip_existing_conversion", True):
            print(f" [SKIP] Existing conversion found: {source_file.name}")
            return json_path

        cmd = ["mineru", "-p", str(source_file.resolve()), "-o", str(self.dirs["out"].resolve())]
        print("\n" + "=" * 50)
        print(f"[*] MinerU conversion: {source_file.name}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[+] {source_file.name} finished MinerU conversion.")
        except subprocess.CalledProcessError as e:
            print(f"[!] MinerU error ({e.returncode}), skipping this file.")
            return None
        except Exception as e:
            print(f"[!] Unknown MinerU error: {e}")
            return None

        if not json_path.exists():
            print(f"[!] Converted JSON not found: {json_path}")
            return None
        return json_path

    def _load_elements_for_file(self, file_path: Path) -> tuple[List[Dict], bool] | tuple[None, None]:
        ext = file_path.suffix.lower()
        is_plain_text = ext in [".md", ".txt"]
        if ext == ".pdf":
            json_path = self._run_mineru_conversion(file_path)
            if json_path is None:
                return None, None
            print(f"[*] Chunking and metadata extraction: {file_path.name}")
            with json_path.open("r", encoding="utf-8") as f:
                return json.load(f), False

        print(f"[*] Chunking and metadata extraction: {file_path.name}")
        elements = self._load_plain_text_elements(file_path, ext)
        if not elements:
            print(f"[!] Warning: Empty text file: {file_path.name}")
            return None, None
        return elements, is_plain_text

    def _gather_source_files(self, target_path: str = "") -> List[Path]:
        target = (target_path or self.cfg.get("target_file", "")).strip()
        if target:
            path = Path(target).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() not in self.cfg["supported_exts"]:
                raise FileNotFoundError(f"Invalid target file: {path}")
            return [path]
        source_files = sorted(
            path for path in self.dirs["source"].iterdir()
            if path.is_file() and path.suffix.lower() in self.cfg["supported_exts"]
        )
        if not self.cfg.get("ingest", {}).get("skip_existing_sources", True):
            return source_files

        existing_source_ids = {
            chunk.get("source", {}).get("source_id")
            for chunk in self.chunks_data
            if isinstance(chunk, dict)
        }
        filtered_files = [path for path in source_files if path.name not in existing_source_ids]
        skipped = len(source_files) - len(filtered_files)
        if skipped:
            print(f"[*] Skip {skipped} existing source files already ingested.")
        return filtered_files

    def _build_chunks_from_elements(self, elements: List[Dict], fname: str, is_plain_text: bool) -> List[Dict]:
        real_title, real_authors = self._extract_paper_meta_concise(elements)
        id_prefix = self._get_id_prefix_from_filename(fname)
        year_match = re.search(r"\d{4}", id_prefix)
        year_val = year_match.group() if year_match else "2024"

        c_ch, c_sec = self.cfg["default_chap_sec"]
        file_temp_chunks: List[Dict] = []
        is_skipping_toc = False
        is_in_numbered_section = True if is_plain_text else False
        one_heading_count = 0

        for element in elements:
            text = element.get("text", "").strip()
            if not text:
                continue
            page_idx = int(element.get("page_idx", 0)) + 1

            if not is_plain_text:
                if text.lower() == "contents":
                    is_skipping_toc = True
                    is_in_numbered_section = False
                    continue
                if "text_level" in element or element.get("type") == "title":
                    h_type, h_val, inferred_ch = self._identify_header_strict(text)
                    if h_type:
                        if is_skipping_toc:
                            if re.match(r"^1(\.|\s|$)", text):
                                one_heading_count += 1
                                if one_heading_count == 2:
                                    is_skipping_toc = False
                                    is_in_numbered_section = True
                        else:
                            is_in_numbered_section = True
                        if h_type == "ch":
                            c_ch, c_sec = h_val, "0"
                        else:
                            c_sec = h_val
                            if c_ch == "0":
                                c_ch = inferred_ch
                    else:
                        is_in_numbered_section = False
                    continue

            if is_skipping_toc or not is_in_numbered_section:
                continue
            if element.get("type") not in ["text", "paragraph", "list", "equation"] or len(text) <= 50:
                continue

            eq_id = ""
            tag_match = re.search(r"\\tag\s*\{([^}]+)\}", text)
            if tag_match:
                eq_id = f"({tag_match.group(1)})"

            for split_text in self._split_text_with_overlap(text):
                seq_idx = len(file_temp_chunks) + 1
                chunk_id = self.cfg["chunk_id_fmt"].format(id_prefix, c_ch, c_sec, seq_idx)
                chapter_str = f"Ch.{c_ch}" if c_sec == "0" else f"Ch.{c_sec}"
                chunk = {
                    "chunk_id": chunk_id,
                    "text": split_text,
                    "source": {
                        "source_id": fname,
                        "title": real_title,
                        "authors": real_authors,
                        "year": int(year_val),
                        "edition": "Original",
                    },
                    "locator": {
                        "chapter": c_ch,
                        "section": c_sec,
                        "page_start": page_idx,
                        "page_end": page_idx,
                        "equation_id": eq_id,
                    },
                    "citation": f"{real_authors[0]} et al. ({year_val}), {chapter_str}, p.{page_idx}" + (f", Eq. {eq_id}" if eq_id else ""),
                    "keywords": self._extract_keywords_pure(split_text),
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                }
                file_temp_chunks.append(chunk)

        for idx in range(len(file_temp_chunks)):
            if idx > 0:
                file_temp_chunks[idx]["prev_chunk_id"] = file_temp_chunks[idx - 1]["chunk_id"]
            if idx < len(file_temp_chunks) - 1:
                file_temp_chunks[idx]["next_chunk_id"] = file_temp_chunks[idx + 1]["chunk_id"]
        return file_temp_chunks

    def process(self, target_path: str = "", reset_existing: bool = False):
        self.new_chunks_data = []
        if reset_existing:
            self.chunks_data = []
            for path in [self.knowledge_path, self.faiss_path, self.id_map_path, self.index_meta_path]:
                self._safe_unlink(path)
        elif self.knowledge_path.exists():
            with self.knowledge_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.chunks_data.append(json.loads(line))
                    except Exception:
                        continue

        existing_chunk_ids = {chunk.get("chunk_id") for chunk in self.chunks_data if isinstance(chunk, dict)}
        source_files = self._gather_source_files(target_path)
        if not source_files:
            print("[!] Error: no supported files found.")
            return

        all_final_chunks: List[Dict] = []
        for source_file in tqdm(source_files, desc="Processing source files"):
            loaded = self._load_elements_for_file(source_file)
            if loaded == (None, None):
                continue
            elements, is_plain_text = loaded
            file_chunks = self._build_chunks_from_elements(elements, source_file.name, bool(is_plain_text))
            all_final_chunks.extend(file_chunks)

        new_chunks = [chunk for chunk in all_final_chunks if chunk.get("chunk_id") not in existing_chunk_ids]
        if new_chunks:
            with self.knowledge_path.open("a", encoding="utf-8") as f:
                for chunk in new_chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        self.chunks_data.extend(new_chunks)
        self.new_chunks_data = new_chunks
        print(
            "[*] Chunking and metadata extraction complete. "
            f"Generated {len(new_chunks)} new chunks (total {len(self.chunks_data)})."
        )

        if not self.cfg["conversion"].get("keep_intermediate_outputs", True):
            print("[*] keep_intermediate_outputs=false, intermediate files remain untouched for now.")

    def _write_id_map(self, chunks: List[Dict], start_index: int = 0, mode: str = "w"):
        with self.id_map_path.open(mode, encoding="utf-8") as f:
            for offset, chunk in enumerate(chunks):
                f.write(
                    json.dumps(
                        {"chunk_id": chunk["chunk_id"], "index_id": start_index + offset},
                        ensure_ascii=False,
                    ) + "\n"
                )

    def _write_index_meta(self, num_vectors: int, batch_size: int, update_mode: str):
        with self.index_meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "index_type": "IndexFlatIP",
                    "embedding_model": self.embedding_cfg["model"],
                    "normalized_embeddings": bool(self.embedding_cfg.get("normalize_embeddings", True)),
                    "embedding_batch_size": batch_size,
                    "device": self.device,
                    "chunking": self.cfg["chunking"],
                    "num_vectors": num_vectors,
                    "update_mode": update_mode,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _can_incrementally_update_index(self) -> bool:
        if not self.faiss_path.exists() or not self.id_map_path.exists():
            return False
        meta = self._load_index_meta()
        if meta.get("index_type") != "IndexFlatIP":
            return False
        if meta.get("embedding_model") != self.embedding_cfg["model"]:
            return False
        existing_id_map = self._load_existing_id_map()
        if not existing_id_map:
            return False
        existing_count = len(existing_id_map)
        existing_chunk_count = len(self.chunks_data) - len(self.new_chunks_data)
        if existing_count != existing_chunk_count:
            return False
        if meta.get("num_vectors") not in (None, existing_count):
            return False
        return True

    def _encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        embeddings = self.emb_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=bool(self.embedding_cfg.get("show_progress_bar", True)),
            normalize_embeddings=bool(self.embedding_cfg.get("normalize_embeddings", True)),
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype="float32")

    def build_index(self, incremental: bool = True):
        if not self.chunks_data:
            print("[!] No chunks available, skip FAISS index build.")
            return

        batch_size = self._embedding_batch_size()
        if incremental and self.faiss_path.exists() and not self.new_chunks_data:
            print("[*] No new chunks detected, keep existing FAISS index.")
            return

        if incremental and self.new_chunks_data and self._can_incrementally_update_index():
            print("[*] Incrementally updating FAISS index...")
            texts = [chunk["text"] for chunk in self.new_chunks_data]
            print(f"[*] Embedding {len(texts)} new chunks with batch_size={batch_size} on {self.device}...")
            embeddings = self._encode_texts(texts, batch_size)
            index = faiss.read_index(str(self.faiss_path))
            start_index = index.ntotal
            index.add(embeddings)
            faiss.write_index(index, str(self.faiss_path))
            self._write_id_map(self.new_chunks_data, start_index=start_index, mode="a")
            self._write_index_meta(index.ntotal, batch_size, update_mode="incremental")
            print(f"[*] FAISS index incrementally updated: {self.faiss_path} (+{len(self.new_chunks_data)} vectors)")
        else:
            print("[*] Building FAISS index from scratch...")
            texts = [chunk["text"] for chunk in self.chunks_data]
            print(f"[*] Embedding {len(texts)} chunks with batch_size={batch_size} on {self.device}...")
            embeddings = self._encode_texts(texts, batch_size)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            faiss.write_index(index, str(self.faiss_path))
            self._write_id_map(self.chunks_data, start_index=0, mode="w")
            self._write_index_meta(len(self.chunks_data), batch_size, update_mode="rebuild")
            print(f"[*] FAISS index built: {self.faiss_path}")

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    store = PriorStore()
    store.process(reset_existing=True)
    store.build_index()
