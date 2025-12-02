import fitz  # PyMuPDF
import re
import json
from typing import Dict, List, Tuple


def is_title(text: str) -> Tuple[bool, str, str]:
    match_digit = re.match(r'^(\d+(\.\d+)*)(\s+)([^\d][\s\S]*)$', text)
    if match_digit:
        return True, match_digit.group(1), match_digit.group(4).strip()

    match_letter = re.match(r'^([A-Z](\.\d+)*)(\s+)([\s\S]+)$', text)
    if match_letter:
        return True, match_letter.group(1), match_letter.group(4).strip()

    return False, "", ""


def extract_title_author_abstract(doc, font_threshold=1.0, top_line_limit=10):
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    lines_info = []

    for block in blocks:
        for line in block.get("lines", []):
            line_text = ""
            font_sizes = []
            for span in line.get("spans", []):
                text = span["text"].strip()
                if text:
                    line_text += text + " "
                    font_sizes.append(span["size"])
            if line_text.strip() and font_sizes:
                avg_font_size = sum(font_sizes) / len(font_sizes)
                lines_info.append((avg_font_size, line_text.strip()))

    # Title
    title_lines = []
    title_end_index = 0
    for i, (font_size, text) in enumerate(lines_info[:top_line_limit]):
        if i == 0:
            title_lines.append((font_size, text))
        else:
            prev_font_size = title_lines[-1][0]
            if abs(prev_font_size - font_size) < font_threshold:
                title_lines.append((font_size, text))
            else:
                title_end_index = i
                break
    title = " ".join([text for _, text in title_lines])

    # Author Info
    author_lines = []
    abstract_index = None
    for i in range(title_end_index, len(lines_info)):
        _, text = lines_info[i]
        if re.match(r"(?i)^abstract$", text):
            abstract_index = i
            break
        author_lines.append(text.strip())
    author_info = " ".join(author_lines)

    # Abstract
    abstract_lines = []
    if abstract_index:
        for _, line_text in lines_info[abstract_index + 1:]:
            if re.match(r'^(1|I+)[\s\.]', line_text) or re.match(r'(?i)^introduction$', line_text):
                break
            abstract_lines.append(line_text.strip())
    abstract = " ".join(abstract_lines)

    return title.strip(), author_info.strip(), abstract.strip()


def extract_headings_by_font(doc) -> List[Tuple[str, str, int, float]]:
    font_sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        font_sizes.append(span["size"])
    body_font_avg = sum(font_sizes) / len(font_sizes)
    font_threshold = 1.2

    headings = []
    last_number = None
    last_font = None

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            block_text = ""
            block_fonts = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if text:
                        block_text += text + " "
                        block_fonts.append(span["size"])
            text = block_text.strip()
            if not text or not block_fonts:
                continue

            avg_font = sum(block_fonts) / len(block_fonts)

            # ✅ 过滤小字体内容，避免正文干扰
            if avg_font < body_font_avg:
                continue

            match, number, title = is_title(text)
            if match:
                if last_number == number and abs((last_font or 0) - avg_font) < 0.5:
                    continue
                last_number = number
                last_font = avg_font
                headings.append((number, title, page_num, avg_font))
            else:
                if (avg_font > body_font_avg * font_threshold and 
                    len(text.split()) <= 6 and 
                    text[0].isupper() and 
                    not re.match(r'^[\d\s\-\+\=\,\(\)]+$', text)):
                    headings.append(("", text, page_num, avg_font))
    return headings



def font_level_mapping(headings: List[Tuple[str, str, int, float]]) -> Dict[float, int]:
    fonts = sorted(set(round(font, 2) for _, _, _, font in headings), reverse=True)
    return {font: idx for idx, font in enumerate(fonts)}


def pdf_reader(pdf_path: str) -> Dict[str, str]:
    doc = fitz.open(pdf_path)
    sections = {}
    current_content = []
    current_key = None

    title, author_info, abstract = extract_title_author_abstract(doc)
    sections["Title"] = title
    sections["Author Info"] = author_info
    sections["Abstract"] = abstract

    headings = extract_headings_by_font(doc)
    font_to_level = font_level_mapping(headings)

    heading_map = {}
    current_hierarchy = {}

    for number, title_part, _, font in headings:
        level = font_to_level[round(font, 2)]
        key_text = f"{number} {title_part}".strip() if number else title_part
        current_hierarchy[level] = key_text
        for l in list(current_hierarchy.keys()):
            if l > level:
                del current_hierarchy[l]
        sorted_levels = sorted(current_hierarchy.items())
        compound_key = " ".join([v for _, v in sorted_levels])
        heading_map[compound_key] = {"level": level, "title": title_part}

    current_hierarchy = {}
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks", sort=True)
        for block in blocks:
            x0, y0, x1, y1, text, _, block_type = block
            text = text.strip()
            if not text or bool(block_type):  # ✅ 移除 y0/y1 限制
                continue
            if page_num == 0 and y0 < 250:
                continue
            
            match, number, title_part = is_title(text)
            
            if match or any(text == h[1] for h in headings if h[0] == ""):
                font_size = 0
                for b in page.get_text("dict")["blocks"]:
                    for line in b.get("lines", []):
                        if text in "".join(span["text"] for span in line["spans"]):
                            font_size = sum(span["size"] for span in line["spans"]) / len(line["spans"])
                            break
                level = font_to_level.get(round(font_size, 2), 0)
                key_text = f"{number} {title_part}".strip() if match else text
                current_hierarchy[level] = key_text
                for l in list(current_hierarchy.keys()):
                    if l > level:
                        del current_hierarchy[l]
                sorted_levels = sorted(current_hierarchy.items())
                compound_key = " ".join([v for _, v in sorted_levels])
                if current_key and current_content:
                    sections[current_key] = "\n".join(current_content)
                current_key = compound_key
                current_content = []
                continue

            if current_key:
                cleaned = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
                current_content.append(cleaned)
    if current_key and current_content:
        sections[current_key] = "\n".join(current_content)

    return sections


def save_sections_to_json(sections: Dict[str, str], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)


# 示例入口
if __name__ == "__main__":
    pdf_path = "/home/ma-user/modelarts/work/duyuwen/pasa/create_dataset/pdf_temp/2501.04519.pdf"
    output_path = "./output.json"

    data = pdf_reader(pdf_path)
    save_sections_to_json(data, output_path)
    print(f"✅ 提取完成，已保存到 {output_path}")
