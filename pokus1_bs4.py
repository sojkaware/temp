import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urlunparse

os.chdir('text webscraping')
url_site = 'https://cw.fel.cvut.cz/b221/courses/b4b33alg/cviceni'
url_site_root = 'https://cw.fel.cvut.cz/b221/courses/b4b33alg'

url_site = 'https://cw.fel.cvut.cz/wiki/courses/b0m36qua/start'


output_offline = 'output_offline'

if not os.path.exists(output_offline):
    os.makedirs(output_offline)

def is_valid_url2(url, parent_url, root_url):
    if url.startswith(root_url):
        parsed_url = urlparse(url)
        path = parsed_url.path
        # check if the URL ends with a file extension
        if '.' in path and path.rindex('.') > path.rindex('/'):
            return True
    if urlparse(url).netloc == '' and urlparse(parent_url).netloc == urlparse(root_url).netloc:
        return True
    return False

def is_valid_url(url):
    
    if url.__contains__('b0m36qua'):
            return True
    return False

def download_file(url, output_dir):
    local_filename = os.path.join(output_dir, urlparse(url).path.lstrip('/').replace('/', os.path.sep))
    local_dir = os.path.dirname(local_filename)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    if not os.path.exists(local_filename):
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                # Process the response here
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except requests.exceptions.HTTPError as e:
            # Handle the exception here
            print(f'Request failed: {e}')
            return ''
    return local_filename


def process_url(url, output_dir):
    r = requests.get(url)
    r.raise_for_status()

    soup = BeautifulSoup(r.content, 'html.parser')
    tags = soup.find_all(['img', 'a'])

    for tag in tags:
        attribute = 'href' if tag.name == 'a' else 'src'
        url_candidate = tag.get(attribute)
        absolute_url = urljoin(url, url_candidate)
        if url_candidate and is_valid_url(absolute_url):
            
            local_file = download_file(absolute_url, output_dir)
            if not local_file == '':
                tag[attribute] = os.path.relpath(local_file, output_dir)

    output_file = os.path.join(output_dir, urlparse(url).path.lstrip('/').replace('/', os.path.sep))
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(soup))

    return output_file


index_file = process_url(url_site, output_offline)
print(f"For offline browsing, open {index_file}")
