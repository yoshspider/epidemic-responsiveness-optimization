from urllib.request import urlopen
import re

url='https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html'
text=urlopen(url,timeout=60).read().decode()
print('\n'.join(re.findall(r'<script[^>]+src=["\']([^"\']+)',text)))
for line in text.splitlines():
    if any(value in line.lower() for value in ['2018','svi.cdc','iframe','csv']):
        print(line.strip()[:320])
