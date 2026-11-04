from urllib.request import urlopen
import re

url='https://svi2.cdc.gov/data-downloads/assets/main-DVz7aho5.js'
text=urlopen(url,timeout=60).read().decode()
for pattern in ['2018','csv','county','DownloadUrl','download','https:']:
    hits=list(re.finditer(re.escape(pattern),text))
    print(pattern,len(hits))
    for hit in hits[-8:]:print(text[max(0,hit.start()-120):hit.end()+150])
