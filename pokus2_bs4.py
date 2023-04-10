import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urlsplit

def download_file1(url, save_path):
    response = requests.get(url, stream=True)
    content_type = response.headers.get('content-type')
    if (content_type == 'text/html; charset=utf-8'):

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                

def download_file2(url, save_path):
    response = requests.get(url, stream=True)
    content_type = response.headers.get('content-type')
    file_extension = content_type.split('/')[-1]
    file_name = os.path.basename(url)
    file_name = f"{os.path.splitext(file_name)[0]}.{file_extension}"
    file_save_path = os.path.join(save_path, file_name)
    with open(file_save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    content_type = response.headers.get('content-type')
    print(content_type)
    file_extension = content_type.split('/')[-1]
    file_name = os.path.basename(url)
    file_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
    url_path = urlsplit(url).path
    file_path = os.path.dirname(url_path)
    full_save_path = os.path.join(save_path, file_path)
    os.makedirs(full_save_path, exist_ok=True)
    file_save_path = os.path.join(full_save_path, file_name)
    with open(file_save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
def create_directory_structure(parsed_url):
    path = os.path.join('output_offline', parsed_url.netloc, *parsed_url.path.split('/')[1:])
    os.makedirs(path, exist_ok=True)
    return path

def scrape_assets(soup, base_url, save_directory):
    assets = {'img': 'src', 'link': 'href', 'script': 'src'}

    for tag, attribute in assets.items():
        for element in soup.find_all(tag):
            url = element.get(attribute)
            if url and not urlparse(url).netloc:
                asset_url = urljoin(base_url, url)
                save_path = os.path.join(save_directory, url.lstrip('/'))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                download_file(asset_url, save_path)
                element[attribute] = os.path.relpath(save_path, save_directory)

def scrape_web_page(url, levels_deep, visited=None):
    if visited is None:
        visited = set()

    if levels_deep < 0 or url in visited:
        return

    visited.add(url)
    response = requests.get(url)

    if response.status_code == 200:
        parsed_url = urlparse(url)
        save_directory = create_directory_structure(parsed_url)
        save_path = os.path.join(save_directory, 'index.html' if parsed_url.path == '/' else os.path.basename(parsed_url.path))

        soup = BeautifulSoup(response.text, 'html.parser')
        scrape_assets(soup, url, save_directory)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(str(soup))

        for link in soup.find_all('a'):
            href = link.get('href')
            if href and not urlparse(href).scheme:
                next_url = urljoin(url, href)
                scrape_web_page(next_url, levels_deep - 1, visited)

if __name__ == "__main__":

    os.chdir('text webscraping')
    url_site = 'https://cw.fel.cvut.cz/b221/courses/b4b33alg/cviceni'
    url_site = 'https://cw.fel.cvut.cz/wiki/courses/b0m36qua/start'
    output_offline = 'output_offline'

    base_url = 'https://cw.fel.cvut.cz/wiki/courses/b0m36qua/start'
    levels_deep = int("1")
    scrape_web_page(base_url, levels_deep)