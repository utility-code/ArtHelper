from fastai.vision.all import *
from fastai.vision.widgets import *
from fastcore.all import *
import time
from fastdownload import download_url
from tqdm import tqdm
import argparse as ap

ags = ap.ArgumentParser()
ags.add_argument("-q", help="queries")
aps = ags.parse_args()


root_dir = "/media/hdd/Datasets/Landscapes"
path=Path(root_dir)/"sorted"
searches = 'Animals', 'Mech'
down_path = Path(root_dir)/"sorted"/"tmp"

searches = aps.q
print(searches)


def search_images(term, max_images=300):
    url = 'https://duckduckgo.com/'
    res = urlread(url,data={'q':term})
    searchObj = re.search(r'vqd=([\d-]+)\&', res)
    requestUrl = url + 'i.js'
    params = dict(l='us-en', o='json', q=term, vqd=searchObj.group(1), f=',,,', p='1', v7exp='a')
    urls,data = set(),{'next':1}
    while len(urls)<max_images and 'next' in data:
        data = urljson(requestUrl,data=params)
        urls.update(L(data['results']).itemgot('image'))
        requestUrl = url + data['next']
        time.sleep(0.2)
    return L(urls)[:max_images]

for o in searches:
    dest = (down_path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    print(f'{o} photo')
    resize_images(down_path/o, dest=path/o)

failed = verify_images(get_image_files(down_path))
failed.map(Path.unlink)
print(len(failed))