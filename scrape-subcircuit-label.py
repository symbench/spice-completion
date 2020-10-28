# This is a suboptim
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def search_digikey(term):
    opts = Options()
    opts.add_argument('--headless')
    driver = webdriver.Chrome(options=opts)
    driver.get('https://www.digikey.com/en/products/')
    search_box = driver.find_element_by_class_name('search-textbox')
    search_box.send_keys(term)
    btn = driver.find_element_by_class_name('search-button')
    btn.click()
    time.sleep(0.5)
    cells = driver.find_elements_by_class_name('MuiTableCell-root')
    result_elements = [cell for cell in cells if 'Items)' in cell.text]
    categories = [ cell.find_element_by_tag_name('a').text for cell in result_elements ]
    if len(categories) == 0:
        items = [ element for element in driver.find_elements_by_tag_name('li') if 'Items)' in element.text ]
        categories = [ item.find_element_by_tag_name('a').text for item in items ]

    driver.quit()
    return categories

subcircuit = sys.argv[1]
categories = search_digikey(subcircuit)
if len(categories) > 0:
    category = categories[0].replace(' ', '_')
    print(category)
