import os
import re
import sys
import json
import time
import shutil
import requests
import tarfile
import concurrent.futures
from pathlib import Path
from search_utils.call_mtr import CallMetricsTracker
from func_timeout import func_set_timeout
with open(f"{os.path.dirname(os.path.abspath(__file__))}/../../config/config.json", "r") as f:
    config = json.load(f)
TIMEOUT = config["timeout"]

metrics_tracker = CallMetricsTracker()

current_path = os.path.dirname(os.path.abspath(__file__))
sys_path = f'{current_path}/../prompts/parse_bib.txt'
with open(sys_path, "r", encoding="utf-8") as f:
    sys_prompt = f.read().replace('\n', ' ')

# MODEL: 本地模型调用，解析ref
from model_vllm import Agent
with open(f'{current_path}/../../config/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
qwen_agent = Agent(model_name=config["tool_model"])

def safe_json_loads(res):
    """
    安全地解析JSON字符串，处理各种常见错误
    """
    if not res or not isinstance(res, str):
        return {}
    
    # 处理Markdown代码块格式
    if res.startswith('```json'):
        res = res[7:]  # 移除开头的```json
    if res.endswith('```'):
        res = res[:-3]  # 移除结尾的```
    
    # 移除前后空白
    res = res.strip()
    
    # 如果为空，返回空字典
    if not res:
        return {}
    
    try:
        return json.loads(res)
    except json.JSONDecodeError as e:
        # 处理转义字符错误
        if "Invalid \\escape" in str(e):
            # 将所有反斜杠都转义为双反斜杠
            fixed_res = re.sub(r'\\', r'\\\\', res)
            try:
                return json.loads(fixed_res)
            except json.JSONDecodeError:
                pass
        
        # 最后尝试：提取JSON对象
        try:
            # 查找最外层的花括号
            brace_count = 0
            start = -1
            for i, char in enumerate(res):
                if char == '{':
                    if brace_count == 0:
                        start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        json_str = res[start:i+1]
                        return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass
        
        # 如果所有方法都失败，返回空字典
        print(f'JSON解析失败，返回空字典: {e}, res: {res[:200]}...')
        return {}
    
def proxy_request_get(url, stream=False, verify=True, params=None, headers=None, timeout=None):
    entry = 'http://{}:{}@27144f3588738de8.arq.na.ipidea.online:2336'.format(
        "dpzhongtai-zone-custom-region-us", "Aa12345678")
    return requests.get(url, stream=stream, verify=verify, proxies={
        'http': entry,
        'https': entry,
    }, params=params, headers=headers, timeout=timeout)

@func_set_timeout(TIMEOUT)
def fetch_eprint(arxiv_id, save_dir, dl_timeout=30):
    """下载arxiv论文压缩包"""
    hit = 0
    if os.path.exists(f"{save_dir}/{arxiv_id}.tar.gz") and config["use_local_db"]:
        print(f"\033[94m {save_dir}/{arxiv_id}.tar.gz 已存在，跳过下载\033[0m")
        hit = 1
        return f"{save_dir}/{arxiv_id}.tar.gz",hit
    url = f'https://arxiv.org/e-print/{arxiv_id}'
    resp = requests.get(url, stream=True, timeout=dl_timeout)
    if 'html' in resp.headers.get('Content-Type', ''):
        print('被 arXiv 拦截，需要验证码，未能下载源码。')
        return None,0
    # 优先从Content-Disposition获取原始文件名
    fname = None
    cd = resp.headers.get('Content-Disposition')
    if cd:
        m = re.search('filename="?([^";]+)"?', cd)
        if m:
            fname = m.group(1)
    if not fname:
        # 兜底：用原有逻辑
        fname = f'{arxiv_id}.tar.gz'
    if 'tar.gz' not in fname:
        print(f"{arxiv_id}下载不是tex格式，而是：{fname}")
        return None,hit
    fname = f'{arxiv_id}.tar.gz'
    local_path = os.path.join(save_dir, fname)
    if os.path.exists(local_path):
        print(f'{local_path} 已存在，跳过下载')
        return local_path,hit
    if resp.status_code == 200:
        start = time.time()
        if config["save_data"]:
            with open(local_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=4096):
                    if time.time() - start > dl_timeout:
                        print(f"tex下载超时，自动退出: {arxiv_id}")
                        return None,hit
                    f.write(chunk)
        return local_path,hit
    else:
        print(f'arxiv 源码包下载失败: {arxiv_id}, 状态码: {resp.status_code}')
        return None,hit

def extract_tar_gz(tar_path, extract_dir):
    if os.path.exists(extract_dir):
        print(f'{extract_dir} 已存在，跳过解压')
        return
    os.makedirs(extract_dir, exist_ok=True)
    
    # 尝试tarfile支持的格式
    tar_modes = ['r:gz', 'r:bz2', 'r:xz', 'r:']
    for mode in tar_modes:
        try:
            with tarfile.open(tar_path, mode) as tar:
                tar.extractall(path=extract_dir)
            break
        except tarfile.ReadError:
            continue
    else:
        # 尝试zip格式
        try:
            import zipfile
            with zipfile.ZipFile(tar_path, 'r') as zip_ref:
                zip_ref.extractall(path=extract_dir)
        except (zipfile.BadZipFile, ImportError):
            # 尝试gzip单文件
            if tar_path.endswith('.gz'):
                import gzip
                out_path = os.path.join(extract_dir, os.path.basename(tar_path)[:-3])
                with gzip.open(tar_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            else:
                raise Exception(f"无法识别的压缩格式: {tar_path}")
    
    # 删除原文件
    for i in range(3):
        try:
            os.remove(tar_path)
            return
        except PermissionError as e:
            print(f"第{i+1}次尝试删除失败: {e}")
            time.sleep(0.5)
    print(f"最终删除失败: {tar_path}")

def find_tex(folder):
    """在文件夹根目录查找主tex文件"""
    # 1. 先找到所有包含begin标志的tex文件
    tex_candidates = []
    for file in os.listdir(folder):
        if file.endswith('.tex'):
            path = os.path.join(folder, file)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if '\\begin{document}' in content:
                        tex_candidates.append(path)
            except UnicodeDecodeError:
                print(f"警告: {path} 无法解码，已跳过")
    
    # 1.1 如果没找到tex文件，返回最大的文件
    if not tex_candidates:
        all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if all_files:
            # 按文件大小排序，取最大的
            all_files.sort(key=lambda x: os.path.getsize(os.path.join(folder, x)), reverse=True)
            first_file = os.path.join(folder, all_files[0])
            print(f"未找到.tex文件，尝试解析最大的文件: {first_file}")
            return first_file
        else:
            raise FileNotFoundError('文件夹为空，未找到任何文件')

    if len(tex_candidates) == 1:
        return tex_candidates[0]
    
    # 2. 查找根目录下的bbl文件，寻找与其同名的tex
    bbl_files = [file for file in os.listdir(folder) if file.endswith('.bbl')]
    for bbl_file in bbl_files:
        bbl_name = os.path.splitext(os.path.basename(bbl_file))[0]
        for tex_path in tex_candidates:
            tex_name = os.path.splitext(os.path.basename(tex_path))[0]
            if tex_name == bbl_name:
                return tex_path
    
    # 3. 优先返回文件名包含'main'的
    for tex_path in tex_candidates:
        if 'main' in os.path.basename(tex_path).lower():
            return tex_path
    
    # 4. 按文件大小从大到小排序，返回最大的
    tex_candidates.sort(key=lambda x: os.path.getsize(x), reverse=True)
    if tex_candidates:
        return tex_candidates[0]
    # 5. 返回第一个
    return tex_candidates[0]

def read_tex(tex_path, visited=None, root_dir=None):
    """递归读取tex文件内容和bib内容"""
    if visited is None:
        visited = set()
    if root_dir is None:
        root_dir = os.path.dirname(tex_path)
    if tex_path in visited:
        return '', ''
    visited.add(tex_path)
    base_dir = os.path.dirname(tex_path)
    try:
        with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print(f"警告: {tex_path} 无法解码，已跳过")
        return '', ''
    content = ''
    bib_content = ''
    # 展开\input、\include、\subfile
    input_pattern = re.compile(r'\\(input|include|subfile)\s*(\{[^}]+\}|[^\s\{][^\s]*)')
    # 展开\import{dir/}{file}
    import_pattern = re.compile(r'\\import\s*\{([^}]+)\}\s*\{([^}]+)\}')
    # 展开\subimport{dir/}{file}
    subimport_pattern = re.compile(r'\\subimport\s*\{([^}]+)\}\s*\{([^}]+)\}')
    for line in lines:
        if line.strip().startswith('%'): # 跳过注释行
            continue
        # 先处理import
        match = import_pattern.search(line)
        if match:
            dir_part = match.group(1)
            file_part = match.group(2)
            if not file_part.endswith('.tex'):
                file_part += '.tex'
            subpath = os.path.join(root_dir, dir_part, file_part)
            if os.path.exists(subpath):
                sub_content, sub_bib = read_tex(subpath, visited, root_dir)
                content += sub_content
                bib_content += sub_bib
            else:
                content += f'% 未找到{os.path.join(dir_part, file_part)}\n'
            continue
        # 再处理subimport
        match = subimport_pattern.search(line)
        if match:
            dir_part = match.group(1)
            file_part = match.group(2)
            if not file_part.endswith('.tex'):
                file_part += '.tex'
            subpath = os.path.join(base_dir, dir_part, file_part)
            if os.path.exists(subpath):
                sub_content, sub_bib = read_tex(subpath, visited, root_dir)
                content += sub_content
                bib_content += sub_bib
            else:
                content += f'% 未找到{os.path.join(dir_part, file_part)}\n'
            continue
        # 处理input/include/subfile
        match = input_pattern.search(line)
        if match:
            subfile = match.group(2)
            if subfile.startswith('{') and subfile.endswith('}'):
                subfile = subfile[1:-1]
            subfile = subfile.strip()
            if not subfile.endswith('.tex'):
                subfile += '.tex'
            subpath = os.path.join(base_dir, subfile)
            if os.path.exists(subpath):
                sub_content, sub_bib = read_tex(subpath, visited, root_dir)
                content += sub_content
                bib_content += sub_bib
            else:
                content += f'% 未找到{subfile}\n'
        else:
            content += line
    # 提取thebibliography环境
    bib_match = re.search(r'\\begin{thebibliography}.*?\\end{thebibliography}', content, re.DOTALL)
    if bib_match:
        bib_content = bib_match.group(0)
        content = content[:bib_match.start()] + content[bib_match.end():] # 去掉正文中的thebibliography部分
    # 截断\appendix或\section{Appendix}及其后内容
    m1 = re.search(r'\\appendix', content)
    m2 = re.search(r'\\section\{[ ]*Appendix[ }]', content, re.IGNORECASE)
    cut = len(content)
    if m1:
        cut = min(cut, m1.start())
    if m2:
        cut = min(cut, m2.start())
    return content[:cut], bib_content

def extract_brace_content(s, start):
    # 输入字符串s和{的起始位置，返回完整内容和右括号位置
    assert s[start] == '{'
    depth = 0
    content = []
    for i in range(start, len(s)):
        if s[i] == '{':
            if depth > 0:
                content.append(s[i])
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return ''.join(content), i + 1
            else:
                content.append(s[i])
        else:
            if depth > 0:
                content.append(s[i])
    return ''.join(content), len(s)

def normalize_text(text):
    # 去除标点符号和特殊字符，保留字母数字和空格
    text = re.sub(r'[^\w\s]', '', text)
    # 标准化空白符
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def simplify_title(text):
    # 提取title
    text = text.replace('\n', ' ')
    while True:
        m = re.search(r'\\href\s*\{', text)
        if not m:
            break
        start = m.end() - 1
        _, next_pos = extract_brace_content(text, start)  # 提取第一个参数
        while next_pos < len(text) and text[next_pos].isspace():  # 跳过空白字符
            next_pos += 1
        if next_pos < len(text) and text[next_pos] == '{':  # 提取第二个参数
            content, end_pos = extract_brace_content(text, next_pos)
            text = text[:m.start()] + content + text[end_pos:]
        else:
            break
    while True:
        m = re.search(r'\\[a-zA-Z]+\s*\{', text)
        if not m:
            break
        start = m.end() - 1
        content, end_pos = extract_brace_content(text, start)  # 提取命令内容
        text = text[:m.start()] + content + text[end_pos:]
    text = re.sub(r'\\[a-zA-Z]+\s*', '', text)  # 去除无参数命令
    return normalize_text(text)

def parse_sections(content, bib_dict):
    """提取section/subsection，只保留bib_dict中存在的引用"""
    cite_pattern = re.compile(r'\\cite[tp]?\{([^}]+)\}')
    pattern = re.compile(r'\\(section|subsection)\{')
    matches = list(pattern.finditer(content))
    # 记录所有块的起止和类型
    blocks = []
    for idx, m in enumerate(matches):
        kind = m.group(1)
        brace_start = m.end()-1
        title, _ = extract_brace_content(content, brace_start)
        block_start = m.start()
        block_end = matches[idx+1].start() if idx+1 < len(matches) else len(content)
        blocks.append({
            'kind': kind,
            'title_raw': title,
            'title': simplify_title(title),
            'start': block_start,
            'end': block_end,
            'content': content[block_start:block_end]
        })
    # 组织成section->subsection结构
    sections = []
    i = 0
    while i < len(blocks):
        if blocks[i]['kind'] == 'section':
            sec = blocks[i]
            sec_dict = {
                'title': sec['title'],
                'content': sec['content'],
                'subsections': [],
                'citations': []
            }
            # 收集属于该section的subsection
            j = i+1
            while j < len(blocks) and blocks[j]['kind'] == 'subsection':
                raw_citations = [k.strip() for cite in cite_pattern.findall(blocks[j]['content']) for k in cite.split(',')]
                citations = [k for k in raw_citations if k in bib_dict]
                sec_dict['subsections'].append({
                    'title': blocks[j]['title'],
                    'content': blocks[j]['content'],
                    'citations': list(set(citations))
                    # 'citations': raw_citations
                })
                j += 1
            # 提取section正文引用
            if sec_dict['subsections']:
                # 有subsection，只提取section到第一个subsection之间的正文引用
                first_sub = blocks[i+1] if (i+1)<len(blocks) and blocks[i+1]['kind']=='subsection' else None
                if first_sub:
                    before_sub_content = sec['content'][:first_sub['start']-sec['start']]
                    raw_citations = [k.strip() for cite in cite_pattern.findall(before_sub_content) for k in cite.split(',')]
                    citations = [k for k in raw_citations if k in bib_dict]
                    sec_dict['citations'] = list(set(citations))
                    # sec_dict['citations'] = raw_citations
            else:
                # 没有subsection，提取整个section正文引用
                raw_citations = [k.strip() for cite in cite_pattern.findall(sec['content']) for k in cite.split(',')]
                citations = [k for k in raw_citations if k in bib_dict]
                sec_dict['citations'] = list(set(citations))
                # sec_dict['citations'] = raw_citations
            sections.append(sec_dict)
            i = j
        else:
            i += 1
    # 格式化输出
    result = {}
    section_num = 1
    for sec in sections:
        sec_title = f"{section_num}. {sec['title']}"
        # 只保留有引用的section
        if sec['citations']:
            titles = [bib_dict[k] for k in sec['citations'] if k in bib_dict]
            if titles:
                result[sec_title] = titles
            # result[sec_title] = sec['citations']
        # 处理subsection
        if sec['subsections']:
            subsection_num = 1
            for sub in sec['subsections']:
                if sub['citations']:
                    sub_title = f"{sec_title} {section_num}.{subsection_num}. {sub['title']}"
                    titles = [bib_dict[k] for k in sub['citations'] if k in bib_dict]
                    if titles:
                        result[sub_title] = titles
                    # result[sub_title] = sub['citations']
                subsection_num += 1
        section_num += 1
    return result

def parse_tex_folder_content(folder, max_chars=300):
    """解析tex文件夹并返回章节内容（前max_chars字）"""
    t0 = time.time()
    tex_path = find_tex(folder)
    t1 = time.time()
    metrics_tracker.add_misc_time('cnt_find_tex_time', t1 - t0)
    content, bib_content = read_tex(tex_path)
    t2 = time.time()
    metrics_tracker.add_misc_time('cnt_read_tex_time', t2 - t1)
    sections_content = parse_sections_content(content, max_chars)
    t3 = time.time()
    metrics_tracker.add_misc_time('cnt_parse_sections_time', t3 - t2)
    shutil.rmtree(folder)
    return sections_content

def parse_sections_content(content, max_chars=300):
    """提取section/subsection的内容文本（前max_chars字），使用与parse_sections完全相同的逻辑"""
    import re
    
    # 提取section和subsection
    pattern = re.compile(r'\\(section|subsection)\{')
    matches = list(pattern.finditer(content))
    
    if not matches:
        return {}
    
    # 记录所有块的起止和类型
    blocks = []
    for idx, m in enumerate(matches):
        kind = m.group(1)
        brace_start = m.end()-1
        title, _ = extract_brace_content(content, brace_start)
        block_start = m.start()
        block_end = matches[idx+1].start() if idx+1 < len(matches) else len(content)
        
        # 提取内容并清理
        raw_content = content[block_start:block_end]
        
        # 深度清理LaTeX内容
        clean_content = raw_content
        
        # 移除注释行
        clean_content = re.sub(r'%.*$', '', clean_content, flags=re.MULTILINE)
        
        # 移除常见的LaTeX环境
        clean_content = re.sub(r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}', '', clean_content, flags=re.DOTALL)
        
        # 移除LaTeX命令（带参数）
        clean_content = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', clean_content)
        clean_content = re.sub(r'\\[a-zA-Z]+\[[^\]]*\]', '', clean_content)
        
        # 移除简单的LaTeX命令
        clean_content = re.sub(r'\\[a-zA-Z]+', '', clean_content)
        
        # 移除剩余的大括号内容
        clean_content = re.sub(r'\{[^}]*\}', '', clean_content)
        
        # 移除方括号内容
        clean_content = re.sub(r'\[[^\]]*\]', '', clean_content)
        
        # 移除特殊字符
        clean_content = re.sub(r'[~&$#^_]', '', clean_content)
        
        # 移除多余的标点符号
        clean_content = re.sub(r'[{}\\]', '', clean_content)
        
        # 规范化空白字符
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        # 移除开头的section/subsection声明残留
        clean_content = re.sub(r'^(section|subsection)\s*', '', clean_content)
        
        # 获取前max_chars字符
        preview_content = clean_content[:max_chars] + "..." if len(clean_content) > max_chars else clean_content
        
        blocks.append({
            'kind': kind,
            'title': simplify_title(title),
            'content': preview_content
        })
    
    # 组织成section->subsection结构，使用与parse_sections完全相同的编号逻辑
    sections_content = {}
    i = 0
    section_num = 1
    while i < len(blocks):
        if blocks[i]['kind'] == 'section':
            sec = blocks[i]
            sec_title = f"{section_num}. {sec['title']}"
            sec_content = sec['content']
            
            # 收集属于该section的subsection
            j = i+1
            subsections = []
            subsection_num = 1
            while j < len(blocks) and blocks[j]['kind'] == 'subsection':
                subsections.append({
                    'title': blocks[j]['title'],
                    'content': blocks[j]['content']
                })
                j += 1
                subsection_num += 1
            
            # 添加顶级章节
            sections_content[sec_title] = sec_content
            
            # 添加子章节，使用与parse_sections完全相同的格式
            if subsections:
                subsection_num = 1
                for sub in subsections:
                    sub_title = f"{sec_title} {section_num}.{subsection_num}. {sub['title']}"
                    sub_content = sub['content']
                    # 确保子章节内容长度不超过max_chars
                    if len(sub_content) > max_chars:
                        sub_content = sub_content[:max_chars] + "..."
                    sections_content[sub_title] = sub_content
                    subsection_num += 1
            i = j
            section_num += 1
        else:
            i += 1
    
    return sections_content


def find_bbl(tex_path):
    """查找bbl文件"""
    base = os.path.splitext(os.path.basename(tex_path))[0]
    bbl_path = os.path.join(os.path.dirname(tex_path), base + '.bbl')
    if not os.path.exists(bbl_path):
        # 兜底：目录下唯一bbl
        bbls = [f for f in os.listdir(os.path.dirname(tex_path)) if f.endswith('.bbl')]
        if bbls:
            bbl_path = os.path.join(os.path.dirname(tex_path), bbls[0])
    return bbl_path

def parse_bib(folder):
    """解析文件夹下所有bib文件，返回{key: title}映射"""
    bib_dict = {}
    bib_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.bib'):
                bib_files.append(os.path.join(root, file))
    entry_pattern = re.compile(r'@\w+\s*\{\s*([^,]+),', re.IGNORECASE)
    title_pattern = re.compile(r'title\s*=\s*([\{\"])(.*)', re.IGNORECASE)
    for bib_path in bib_files:
        try:
            with open(bib_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            print(f"警告: {bib_path} 无法解码，已跳过")
            continue
        key = None
        title = None
        in_entry = False
        in_title = False
        title_delim = None
        title_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if not in_entry:
                m = entry_pattern.match(line)
                if m:
                    key = m.group(1).strip()
                    in_entry = True
                    title = None
                    in_title = False
                    title_lines = []
                    title_delim = None
                continue
            # 处理title字段
            if in_entry and not in_title:
                m = title_pattern.match(line)
                if m:
                    title_delim = m.group(1)
                    rest = m.group(2).strip()
                    if rest.endswith(',' or '}'):
                        # 单行title
                        rest = rest.rstrip(',')
                        rest = rest.rstrip('}')
                        title = rest
                        in_title = False
                    else:
                        # 多行title
                        title_lines = [rest]
                        in_title = True
                    continue
            elif in_title:
                # 继续收集title内容
                title_lines.append(line)
                # 判断是否结束
                if title_delim == '{':
                    if '}' in line:
                        # 可能有多余内容
                        idx = line.find('}')
                        title_lines[-1] = line[:idx]
                        title = ' '.join(title_lines)
                        in_title = False
                elif title_delim == '"':
                    if '"' in line:
                        idx = line.find('"')
                        title_lines[-1] = line[:idx]
                        title = ' '.join(title_lines)
                        in_title = False
                continue
            # 条目结束
            if in_entry and line.endswith('}'):  # 条目结束
                if key and title:
                    # 清理title
                    title_clean = simplify_title(title)
                    bib_dict[key] = title_clean
                in_entry = False
                key = None
                title = None
                in_title = False
                title_lines = []
                title_delim = None
    return bib_dict

def extract_titles_from_bib(bib_path):
    """【未使用】用bibtexparser解析bib"""
    # import bibtexparser
    # titles = []
    # with open(bib_path, 'r', encoding='utf-8') as bibtex_file:
    #     bib_database = bibtexparser.load(bibtex_file)
    #     for entry in bib_database.entries:
    #         if 'title' in entry:
    #             # 去除多余空格和换行
    #             title = entry['title'].replace('\n', ' ').strip()
    #             titles.append(title)
    # return titles

def parse_bbl(bbl_path, bbl_content, bib_folder):
    """手动解析bbl-bib文件，LLM兜底，返回{key: title}映射"""
    bib_dict = {}
    content = bbl_content
    if bbl_path and os.path.exists(bbl_path):
        try:
            with open(bbl_path, 'r', encoding='utf-8', errors='ignore') as f:
                content += f.read()
        except UnicodeDecodeError:
            print(f"警告: {bbl_path} 无法解码，已跳过")
    bibitems = []
    if content: # 解析bbl内容
        content = re.sub(r'%\s*\n\s*', '', content)  # 移除%符号
        content = re.sub(r'\\\s*\n\s*', ' ', content)  # 处理续行符
        bibitem_pattern = re.compile(r'\\bibitem\s*(?:\[[^\]]*\]\s*)?\{([^}]+)\}')
        bibitems = list(bibitem_pattern.finditer(content))
        for idx, m in enumerate(bibitems):
            key = m.group(1)
            if key in bib_dict:
                continue
            start = m.end()
            end = bibitems[idx+1].start() if idx+1 < len(bibitems) else len(content)
            entry = content[start:end]
            showtitle = re.search(r'\\showarticletitle\{([^}]*)\}', entry) # 优先提取showarticletitle
            title = ''
            if showtitle:
                title = simplify_title(showtitle.group(1))
            else:
                # 取第一个newblock后的内容
                newblock_parts = re.split(r'\\newblock', entry)
                if len(newblock_parts)>1:
                    title = simplify_title(newblock_parts[1].strip())
            if title:
                bib_dict[key] = title
    # 解析bib内容
    if bib_folder is None and bbl_path:
        bib_folder = os.path.dirname(bbl_path)
    if bib_folder:
        bib_dict_bib = parse_bib(bib_folder)
        bib_dict.update(bib_dict_bib)
    # 让LLM解析bbl
    if not bib_dict and bibitems:
        group_contents = []
        for i in range(len(bibitems)):
            start = bibitems[i].start()
            if i + 1 < len(bibitems):
                end = bibitems[i + 1].start()
            else:
                end = len(content)
            group_content = sys_prompt + content[start:end]
            group_contents.append(group_content)
        results = qwen_agent.batch_infer_safe(group_contents, batch_size=len(group_contents))
        normalized_content = normalize_text(content)
        for res in results:
            try:
                if not res:
                    continue
                group_dict = safe_json_loads(res)
                if isinstance(group_dict, dict):
                    # 对键值对进行内容匹配验证
                    for key, value in group_dict.items():
                        normalized_value = normalize_text(value)
                        if normalized_value in normalized_content and key in content:
                            bib_dict[key] = simplify_title(value)
            except Exception as e:
                print(f'LLM解析失败: {e}, res: {res}')
    return bib_dict

def parse_tex_folder(folder):
    """解析tex文件夹"""
    t0 = time.time()
    tex_path = find_tex(folder)
    t1 = time.time()
    metrics_tracker.add_misc_time('find_tex_time', t1 - t0)

    content, bib_content = read_tex(tex_path)
    t2 = time.time()
    metrics_tracker.add_misc_time('read_tex_time', t2 - t1)

    bbl_path = find_bbl(tex_path)
    t3 = time.time()
    metrics_tracker.add_misc_time('find_bbl_time', t3 - t2)

    bib_dict = parse_bbl(bbl_path, bib_content, folder)
    t4 = time.time()
    metrics_tracker.add_misc_time('parse_bbl_time', t4 - t3)

    sections = parse_sections(content, bib_dict)
    t5 = time.time()
    metrics_tracker.add_misc_time('parse_sections_time', t5 - t4)

    shutil.rmtree(folder)
    return sections

if __name__ == '__main__':
    exit(0)
    res = call_llm(r"""
             \begin{thebibliography}{10}\itemsep=-1pt

\bibitem{ba2016layer}
Jimmy~Lei Ba, Jamie~Ryan Kiros, and Geoffrey~E. Hinton.
\newblock Layer normalization, 2016.

\bibitem{beyer2020we}
Lucas Beyer, Olivier~J H{\'e}naff, Alexander Kolesnikov, Xiaohua Zhai, and
  A{\"a}ron van~den Oord.
\newblock Are we done with imagenet?
\newblock {\em arXiv preprint arXiv:2006.07159}, 2020.

\bibitem{carion2020end}
Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander
  Kirillov, and Sergey Zagoruyko.
\newblock End-to-end object detection with transformers.
\newblock In {\em European Conference on Computer Vision}, pages 213--229.
  Springer, 2020.

\bibitem{chen2020pre}
Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei
  Ma, Chunjing Xu, Chao Xu, and Wen Gao.
\newblock Pre-trained image processing transformer.
\newblock {\em arXiv preprint arXiv:2012.00364}, 2020.

\bibitem{chollet2017xception}
Fran{\c{c}}ois Chollet.
\newblock Xception: Deep learning with depthwise separable convolutions.
\newblock In {\em Proceedings of the IEEE conference on computer vision and
  pattern recognition}, pages 1251--1258, 2017.

\bibitem{chu2021really}
Xiangxiang Chu, Bo Zhang, Zhi Tian, Xiaolin Wei, and Huaxia Xia.
\newblock Do we really need explicit position encodings for vision
  transformers?
\newblock {\em arXiv preprint arXiv:2102.10882}, 2021.

\bibitem{xiyang2020danas}
Xiyang Dai, Dongdong Chen, Mengchen Liu, Yinpeng Chen, and Lu YUan.
\newblock Da-nas: Data adapted pruning for efficient neural architecture
  search.
\newblock In {\em European Conference on Computer Vision}, 2020.

\bibitem{dai2020up}
Zhigang Dai, Bolun Cai, Yugeng Lin, and Junying Chen.
\newblock Up-detr: Unsupervised pre-training for object detection with
  transformers.
\newblock {\em arXiv preprint arXiv:2011.09094}, 2020.""")
    print(res)
