import os
import json
import re
import subprocess
import numpy as np
import faiss
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import Counter
from sentence_transformers import SentenceTransformer

# ===================== 核心配置 =====================
CFG = {
    "dirs": {
        "source": "./prior/source", 
        "out": "./prior/out",        # MinerU 转换结果根目录
        "knowledge": "./prior/knowledge", 
        "index": "./prior/index",
    },
    "embedding": {"model": "all-MiniLM-L6-v2"},
    "chunk_id_fmt": "{}:ch{}:sec{}:{:04d}",
    "default_chap_sec": ("0", "0"),
    # 可选：指定单个文件路径（为空则遍历 source 目录）
    "target_file": "",
    "supported_exts": [".pdf", ".md", ".txt"]
}

# ===================== 核心处理类 =====================
class PriorStore:
    def __init__(self):
        for d in CFG["dirs"].values(): os.makedirs(d, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"[*] 初始化 Embedding 模型 ({self.device})...")
        self.emb_model = SentenceTransformer(CFG["embedding"]["model"], device=self.device)
        
        self.knowledge_path = os.path.join(CFG["dirs"]["knowledge"], "chunks.jsonl")
        self.faiss_path = os.path.join(CFG["dirs"]["index"], "index.faiss")
        self.id_map_path = os.path.join(CFG["dirs"]["index"], "id_map.jsonl")
        self.chunks_data = []

    # ------------------- 特征提取逻辑 -------------------

    def _get_id_prefix_from_filename(self, filename: str) -> str:
        """从文件名提取标识，如 'bonnerot_2022.pdf' -> 'bonnerot2022'"""
        name = os.path.splitext(filename.lower())[0]
        author = re.search(r'[a-z]+', name).group() if re.search(r'[a-z]+', name) else "phys"
        year = re.search(r'\d{4}', name).group() if re.search(r'\d{4}', name) else "2024"
        return f"{author}{year}"

    def _extract_paper_meta_concise(self, elements: List[Dict]) -> Tuple[str, List[str]]:
        """提取论文标题和作者列表"""
        title = "Unknown Title"
        authors = ["Unknown Author"]
        title_idx = -1
        # 1. 第一个文本层级为 1 的是标题
        for i, el in enumerate(elements):
            if el.get("text_level") == 1:
                title = el.get("text", "").strip()
                title_idx = i
                break
        # 2. 标题下第一个符合结构的非空行是作者
        if title_idx != -1:
            for j in range(title_idx + 1, min(title_idx + 5, len(elements))):
                raw_text = elements[j].get("text", "").strip()
                if raw_text:
                    clean_text = re.sub(r'[^a-zA-Z\s,]', '', raw_text) # 只要英文
                    names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', clean_text)
                    if names: authors = names; break
        return title, authors

    def _extract_keywords_pure(self, text: str) -> List[str]:
        """纯动态关键词提取：专有名词短语 + 高频学术长词"""
        phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        words = re.findall(r'\b[a-z]{7,}\b', text.lower())
        stopwords = {"following", "between", "results", "through", "using", "shown"}
        common = [w for w, c in Counter(words).most_common(10) if w not in stopwords]
        return list(dict.fromkeys(phrases + common))[:5]

    def _identify_header_strict(self, text: str) -> Tuple[str, str, str]:
        """识别数字开头的标题层级"""
        clean_text = text.strip()
        sec_match = re.match(r"^(\d+(?:\.\d+)+)", clean_text)
        if sec_match:
            val = sec_match.group(1)
            return "sec", val, val.split(".")[0]
        ch_match = re.match(r"^(?:Chapter\s+)?(\d+)(?!\.)", clean_text, re.I)
        if ch_match:
            val = ch_match.group(1)
            return "ch", val, val
        return None, "0", "0"

    def _load_plain_text_elements(self, path: str, ext: str) -> List[Dict]:
        """将 .md/.txt 转为简化 elements 列表"""
        elements = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            text = line.strip()
            if not text:
                continue
            el = {"text": text, "type": "paragraph", "page_idx": 0}
            if ext == ".md" and text.startswith("#"):
                level = len(re.match(r"^#+", text).group(0))
                el["text_level"] = level
                el["type"] = "title"
                el["text"] = text.lstrip("#").strip()
            elements.append(el)
        if ext == ".txt" and elements:
            elements[0]["text_level"] = 1
            elements[0]["type"] = "title"
        return elements

    # ------------------- 主解析流程 -------------------

    def process(self, target_path: str = ""):
        target_path = (target_path or CFG.get("target_file", "")).strip()
        if os.path.exists(self.knowledge_path):
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.chunks_data.append(json.loads(line))
                    except Exception:
                        continue
        existing_chunk_ids = {c.get("chunk_id") for c in self.chunks_data if isinstance(c, dict)}
        if target_path:
            target_path = os.path.abspath(target_path)
            ext = os.path.splitext(target_path)[1].lower()
            if not os.path.isfile(target_path) or ext not in CFG["supported_exts"]:
                print(f"[!] Error: invalid file path: {target_path}")
                return
            pdf_files = [os.path.basename(target_path)]
            source_dir = os.path.dirname(target_path)
        else:
            pdf_files = [
                f for f in os.listdir(CFG["dirs"]["source"])
                if os.path.splitext(f)[1].lower() in CFG["supported_exts"]
            ]
            source_dir = CFG["dirs"]["source"]
        all_final_chunks = []

        if not pdf_files:
            print("[!] Error: no supported files found.")
            return

        for fname in pdf_files:
            ext = os.path.splitext(fname)[1].lower()
            is_plain_text = ext in [".md", ".txt"]

            # 1. 路径预处理（使用绝对路径并增加双引号保护）
            abs_source = os.path.abspath(os.path.join(source_dir, fname))
            abs_out_root = os.path.abspath(CFG["dirs"]["out"])
            
            if ext == ".pdf":
                base_name = fname.replace(".pdf", "")
                # 预期生成的 json 文件路径
                json_path = os.path.join(CFG["dirs"]["out"], base_name, "hybrid_auto", f"{base_name}_content_list.json")
                
                # 2. 增量判断
                if not os.path.exists(json_path):
                    print(f"\n" + "="*50)
                    print(f"[*] MinerU conversion: {fname}")
                    
                    # 调用 MinerU 进行转换
                    cmd = f'mineru -p "{abs_source}" -o "{abs_out_root}"'
                    
                    try:
                        # MinerU 的日志直接显示在控制台
                        subprocess.run(cmd, shell=True, check=True)
                        print(f"[+] {fname} Finished MinerU conversion.")
                    except subprocess.CalledProcessError as e:
                        print(f" [!] MinerU error ({e.returncode}), skipping this file.")
                        continue
                    except Exception as e:
                        print(f" [!] Unknown error: {e}")
                        continue
                else:
                    print(f" [SKIP] There are already json files, no need to convert: {fname}")

                # 3. 二次确认文件是否存在
                if not os.path.exists(json_path):
                    print(f" [!] Warning: The converted JSON not found: {json_path}")
                    continue
                
                # 4. 解析 JSON 并生成 Chunk
                print(f"[*] Chunking and metadata extraction: {fname}")
                with open(json_path, "r", encoding="utf-8") as f:
                    elements = json.load(f)
            else:
                print(f"[*] Chunking and metadata extraction: {fname}")
                elements = self._load_plain_text_elements(abs_source, ext)
                if not elements:
                    print(f" [!] Warning: Empty text file: {fname}")
                    continue
            
            # 3. 提取元数据
            real_title, real_authors = self._extract_paper_meta_concise(elements)
            id_prefix = self._get_id_prefix_from_filename(fname)
            year_val = re.search(r'\d{4}', id_prefix).group() if re.search(r'\d{4}', id_prefix) else "2024"

            # 4. 状态机扫描 (目录过滤 + 章节识别)
            c_ch, c_sec = CFG["default_chap_sec"]
            file_temp_chunks = []
            is_skipping_toc = False
            is_in_numbered_section = True if is_plain_text else False
            one_heading_count = 0 

            for el in elements:
                text = el.get("text", "").strip()
                if not text: continue
                
                # 获取物理页码
                c_pg = el.get("page_idx", 0) + 1

                if not is_plain_text:
                    # A. 目录拦截逻辑
                    if text.lower() == "contents":
                        is_skipping_toc = True
                        is_in_numbered_section = False
                        continue
                    
                    # B. 识别标题元素 (text_level 或 type:title)
                    if "text_level" in el or el.get("type") == "title":
                        h_type, h_val, inf_ch = self._identify_header_strict(text)
                        
                        if h_type: # 数字开头的有效标题
                            if is_skipping_toc:
                                if re.match(r"^1(\.|\s|$)", text):
                                    one_heading_count += 1
                                    if one_heading_count == 2: # 正文开始
                                        is_skipping_toc = False
                                        is_in_numbered_section = True
                            else:
                                is_in_numbered_section = True
                            
                            # 更新层级
                            if h_type == "ch": c_ch, c_sec = h_val, "0"
                            elif h_type == "sec":
                                c_sec = h_val
                                if c_ch == "0": c_ch = inf_ch
                        else:
                            # 非数字标题拦截内容（如 References）
                            is_in_numbered_section = False
                        continue

                # C. 核心知识提取
                if not is_skipping_toc and is_in_numbered_section:
                    if el.get("type") in ["text", "paragraph", "list", "equation"] and len(text) > 50:
                        
                        # 特殊：从 \tag {数字} 中提取公式 ID
                        eq_id = ""
                        tag_match = re.search(r"\\tag\s*\{([^}]+)\}", text)
                        if tag_match:
                            eq_id = f"({tag_match.group(1)})" # 包装为 (1) 格式
                        
                        seq_idx = len(file_temp_chunks) + 1
                        cid = CFG["chunk_id_fmt"].format(id_prefix, c_ch, c_sec, seq_idx)
                        
                        ch_str=""  # 章节字符串
                        if c_sec == "0":
                            ch_str = f"Ch.{c_ch}"
                        else:
                            ch_str = f"Ch.{c_sec}"

                        chunk = {
                            "chunk_id": cid,
                            "text": text,
                            "source": {
                                "source_id": fname,
                                "title": real_title,
                                "authors": real_authors,
                                "year": int(year_val),
                                "edition": "Original"
                            },
                            "locator": {
                                "chapter": c_ch,
                                "section": c_sec,
                                "page_start": c_pg,
                                "page_end": c_pg,
                                "equation_id": eq_id
                            },
                            "citation": f"{real_authors[0]} et al. ({year_val}), {ch_str}, p.{c_pg}" + (f", Eq. {eq_id}" if eq_id else ""),
                            "keywords": self._extract_keywords_pure(text),
                            "prev_chunk_id": None, "next_chunk_id": None
                        }
                        file_temp_chunks.append(chunk)

            # 5. 链接邻居
            for i in range(len(file_temp_chunks)):
                if i > 0: file_temp_chunks[i]["prev_chunk_id"] = file_temp_chunks[i-1]["chunk_id"]
                if i < len(file_temp_chunks)-1: file_temp_chunks[i]["next_chunk_id"] = file_temp_chunks[i+1]["chunk_id"]
            
            all_final_chunks.extend(file_temp_chunks)

        new_chunks = [c for c in all_final_chunks if c.get("chunk_id") not in existing_chunk_ids]
        if new_chunks:
            # 追加写入 JSONL (强制 UTF-8)
            with open(self.knowledge_path, "a", encoding="utf-8") as f:
                for c in new_chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
        self.chunks_data.extend(new_chunks)
        print(
            "[*] Chunking and metadata extraction complete. "
            f"Generated {len(new_chunks)} new chunks (total {len(self.chunks_data)}). "
            "Directory filtering successful."
        )

    def build_index(self):
        """构建 FAISS 索引 (逻辑同前)"""
        if not self.chunks_data: return
        print("[*] Building FAISS index...")
        texts = [c["text"] for c in self.chunks_data]
        embs = self.emb_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(np.array(embs).astype("float32"))
        faiss.write_index(index, self.faiss_path)
        with open(self.id_map_path, "w", encoding="utf-8") as f:
            for i, c in enumerate(self.chunks_data):
                f.write(json.dumps({"chunk_id": c["chunk_id"], "index_id": i}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ps = PriorStore()
    ps.process()
    ps.build_index()
