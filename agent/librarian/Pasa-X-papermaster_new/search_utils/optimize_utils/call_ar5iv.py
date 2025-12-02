import re
import bs4
import warnings
import requests


############### 工具函数 ###############
def generate_full_toc(soup):
    toc = []
    stack = [(0, toc)]
    
    # Mapping of heading tags to their levels
    heading_tags = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5}
    
    for tag in soup.find_all(heading_tags.keys()):
        level = heading_tags[tag.name]
        title = tag.get_text()
        
        # Ensure the stack has the correct level
        while stack and stack[-1][0] >= level:
            stack.pop()
        
        current_level = stack[-1][1]

        # Find the nearest enclosing section with an id
        section = tag.find_parent('section', id=True)
        section_id = section.get('id') if section else None
        
        # Create the new entry
        new_entry = {'title': title, 'id': section_id, 'subsections': []}
        
        current_level.append(new_entry)
        stack.append((level, new_entry['subsections']))

    return toc

def parse_text(local_text, tag):
    ignore_tags = ['a', 'figure', 'center', 'caption', 'td', 'h1', 'h2', 'h3', 'h4']
    # latexmlc
    ignore_tags += ['sup']
    max_math_length = 300000

    for child in tag.children:
        child_type = type(child)
        if child_type == bs4.element.NavigableString:
                txt = child.get_text()
                local_text.append(txt)

        elif child_type == bs4.element.Comment:
            continue
        elif child_type == bs4.element.Tag:

                if child.name in ignore_tags or (child.has_attr('class') and child['class'][0] == 'navigation'):
                    continue
                elif child.name == 'cite':
                    # add hrefs
                    hrefs = [a.get('href').strip('#') for a in child.find_all('a', class_='ltx_ref')]
                    local_text.append('~\cite{' + ', '.join(hrefs) + '}')
                elif child.name == 'img' and child.has_attr('alt'):
                    math_txt = child.get('alt')
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.has_attr('class') and (child['class'][0] == 'ltx_Math' or child['class'][0] == 'ltx_equation'):
                    math_txt = child.get_text()
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.name == 'section':
                    return
                else:
                    parse_text(local_text, child)
        else:
            raise RuntimeError('Unhandled type')

def clean_text(text):
    delete_items = ['=-1', '\t', u'\xa0', '[]', '()', 'mathbb', 'mathcal', 'bm', 'mathrm', 'mathit', 'mathbf', 'mathbfcal', 'textbf', 'textsc', 'langle', 'rangle', 'mathbin']
    for item in delete_items:
        text = text.replace(item, '')
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[[,]+]', '', text)
    text = re.sub(r'\.(?!\d)', '. ', text)
    text = re.sub('bib. bib', 'bib.bib', text)
    return text

def remove_stop_word_sections_and_extract_text(toc, soup, stop_words=['references', 'acknowledgments', 'about this document', 'apopendix']):
    def has_stop_word(title, stop_words):
        return any(stop_word.lower() in title.lower() for stop_word in stop_words)
    
    def extract_text(entry, soup):
        section_id = entry['id']
        if section_id: # section_id
            section = soup.find(id=section_id)
            if section is not None:
                local_text = []
                parse_text(local_text, section)
                if local_text:
                    processed_text = clean_text(''.join(local_text))
                    entry['text'] = processed_text
        return 0 
    
    def filter_and_update_toc(entries):
        filtered_entries = []
        for entry in entries:
            if not has_stop_word(entry['title'], stop_words):
                # Get clean text
                extract_text(entry, soup)                
                entry['subsections'] = filter_and_update_toc(entry['subsections'])
                filtered_entries.append(entry)
        return filtered_entries
    
    return filter_and_update_toc(toc)

def get_subsection(sections):
    res = {}
    for section in sections:
        if "text" in section and section["text"].strip() != "":
            res[section["title"].strip()] = section["text"].strip()
        subsections = get_subsection(section["subsections"])
        for k, v in subsections.items():
            res[k] = v
    return res

def get_1st_section(sections):
    res = {}
    for section in sections:
        subsections = get_subsection(section["subsections"])
        if "text" in section and section["text"].strip() != "" or len(subsections) > 0:
            if "text" in section and section["text"].strip() != "":
                res[section["title"].strip()] = section["text"].strip()
            else:
                res[section["title"].strip()] = ""
            for k, v in subsections.items():
                res[section["title"].strip()] += v.strip()
    res_new = {}
    for k, v in res.items():
        if "appendix" not in k.lower():
            res_new[" ".join(k.split("\n")).strip()] = v
    return res_new

def get_2nd_section(sections):
    res = {}
    for section in sections:
        subsections = get_1st_section(section["subsections"])
        if "text" in section and section["text"].strip() != "":
            if "text" in section and section["text"].strip() != "":
                res[section["title"].strip()] = section["text"].strip()
        for k, v in subsections.items():
            res[section["title"].strip() + " " + k.strip()] = v.strip()
    res_new = {}
    for k, v in res.items():
        if "appendix" not in k.lower():
            res_new[" ".join(k.split("\n")).strip()] = v
    return res_new

def parse_metadata(metas):
    # 先看第0项的第一个双引号
    first = metas[0]
    match = re.search(r'“([^”]+)”', first)
    if match:
        title = match.group(1)
    elif len(metas) > 1:
        # 如果失败，且数组长度大于1，就取第1项
        title = metas[1]
    else:
        # 兜底
        title = first

    # 格式化结果
    title = title.replace('\n', ' ')
    title = title.strip(' "“”\'，,.;。')
    return {"title": title}

def create_dict_for_citation(ul_element):
    citation_dict, futures, id_attrs = {}, [], []
    for li in ul_element.find_all("li", recursive=False):
        id_attr = li['id']
        metas = [x.text.strip() for x in li.find_all('span', class_='ltx_bibblock')]
        id_attrs.append(id_attr)
        futures.append(parse_metadata(metas))
    results = list(zip(id_attrs, futures))
    citation_dict = dict(results)
    return citation_dict

def parse_html(html_file):
    soup = bs4.BeautifulSoup(html_file, "lxml")
    # parse title
    title = soup.head.title.get_text().replace("\n", " ")
    # parse abstract
    abstract = soup.find(class_='ltx_abstract').get_text()
    # parse citation
    citation = soup.find(class_='ltx_biblist')
    citation_dict = create_dict_for_citation(citation)
    # generate the full toc without text
    sections = generate_full_toc(soup)
    # remove the sections need to skip and extract the text of the rest sections
    sections = remove_stop_word_sections_and_extract_text(sections, soup)
    document = {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "references": citation_dict,
    }
    return document 

############### 搜索ar5iv函数 ###############
def  search_section_by_arxiv_id(entry_id, cite=r"~\\cite\{(.*?)\}"):
    """根据aid搜索ar5iv，返回论文完整信息，未找到返回None"""
    arxiv_id = entry_id.split('v')[0] if 'v' in entry_id else entry_id
    
    url = f'https://ar5iv.labs.arxiv.org/html/{arxiv_id}'
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            html_content = response.text
            if not 'https://ar5iv.labs.arxiv.org/html' in html_content:
                warnings.warn(f'Invalid ar5iv HTML document: {url}')
                return None
            else:
                try:
                    document = parse_html(html_content)
                except:
                    warnings.warn(f'Wrong format HTML document: {url}')
                    return None
                try:
                    sections = get_2nd_section(document["sections"][0]["subsections"])
                except:
                    warnings.warn(f'Get subsections error')
                    return None
                sections2title = {}
                for k, v in sections.items():
                    k = " ".join(k.split("\n"))
                    sections2title[k] = set()
                    bibs = re.findall(cite, v, re.DOTALL)
                    for bib in bibs:
                        bib = bib.split(",")
                        for b in bib:
                            if b not in document["references"]:
                                continue
                            sections2title[k].add(document["references"][b]["title"]) # !!! The title here may be incorrect, you can use an LLM to parse the write title from document["references"][b]["meta_string"] !!!
                    if len(sections2title[k]) == 0:
                        del sections2title[k]
                    else:
                        sections2title[k] = list(sections2title[k])
                
                # 去掉title中的aid部分
                title = document["title"]
                title = re.sub(r'\[\d+\.\d+v?\d*\]\s*', '', title).strip()
                return {
                    "arxiv_id": arxiv_id,
                    # "title": title,
                    # "abstract": document["abstract"],
                    "sections": sections2title,
                    # "references": {f"ref{i+1}": v["title"] for i, v in enumerate(document["references"].values())}
                }
        else:
            warnings.warn(f"Failed to retrieve content. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        warnings.warn(f"An error occurred: {e}")
        return None


if __name__ == '__main__':
    pass