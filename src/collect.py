from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import os, time, sys

KEY = 'c2bfc8c970341f6b6d320b3597b6c1ac'
SECRET = '87bfbc5a551d1a46'
WAIT_TIME = 1

keyword = sys.argv[1]
outdir = '/home/miyata/test/img/download/' + keyword

if not os.path.exists(outdir):
    os.makedirs(outdir)

flickr = FlickrAPI(KEY, SECRET, format='parsed-json')
result = flickr.photos.search(
    text = keyword,
    per_page = 1000,
    media = 'photos',
    sort = 'relevance',
    save_search = 1,
    extras = 'url_q, license'
)
photos = result['photos']

for i,photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = outdir + '/' f"{keyword}_{i:0>3}.jpg"
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(WAIT_TIME)