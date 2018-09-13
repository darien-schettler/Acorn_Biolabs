import urllib.request
import urllib.error
from bs4 import BeautifulSoup

# Specify the variable URL
var_page = 'https://you.23andme.com/reports/ghr.vte/print/'

# query the website and return the html to the variable ‘page’
page = urllib.request.urlopen(var_page)
print(page)