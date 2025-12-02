import os
import requests
def arxiv_download(url: str, dir: str = '.'):
    """
    下载url的arxiv文献并保存到指定目录下

    :param url: arxiv文献的URL
    :param dir: 下载文献的保存目录
    :return: 下载文献的文件路径
    """
    paper_id = url.split('/')[-1]
    filename = f'{paper_id}.pdf'
    direct_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
    if ".pdf" not in dir:
        save_path = dir + "/" + filename
    else:
        save_path = dir
    try:
        print(f"开始下载: {url} 到 {save_path}")
        response = requests.get(direct_url)
        print(f"HTTP: {response.status_code}")
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f'下载成功，保存路径：{save_path}')
        return save_path
    except requests.RequestException as e:
        print(f'下载失败，错误信息：{e}')
        return ""

def openreview_download(url: str, dir: str = '.'):
    """
    下载url的openreview文献并保存到指定目录下

    :param url: openreview的URL
    :param dir: 下载文献的保存目录
    :return: 下载文献的文件路径
    """
    if ".pdf" not in dir:
        file_path = dir + "/" + "paper.pdf"
    else:
        file_path = dir
    try:
        headers = {
            "Accept": "application/pdf"
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                pdf_file.write(chunk)
        
        print(f"下载成功，保存路径：{file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"下载失败，错误信息：{e}")
        return ""

def paper_downloader(url: str, dir: str = '.'):
    if not os.path.exists(os.path.dirname(dir)):
        os.makedirs(os.path.dirname(dir))
    path = ""
    if "arxiv" in url:
        path = arxiv_download(url, dir)
    elif "openreview" in url:
        path = openreview_download(url, dir)
    else:
        path = ""
    return path

if __name__ == '__main__':
    url = "https://arxiv.org/abs/2501.04519"
    url2 = "https://openreview.net/pdf?id=tDAu3FPJn9"
    arxiv_download(url)
    openreview_download(url2)