from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
import pandas as pd
import re 
from unidecode import unidecode
from threading import Thread

def get_lyrics(artist):
    df = pd.DataFrame(columns = ['song_name', 'lyrics'])

    options = Options()
    options.headless = True

    driver = webdriver.Chrome()
    driver.get('https://www.metrolyrics.com/{}-lyrics.html'.format(artist))
    sleep(2)

    elements = driver.find_elements_by_class_name('hasvidtable')

    for i in range(len(elements)):
        print(i)
        elements = driver.find_elements_by_class_name('hasvidtable')
        row = [elements[i].text]
        elements[i].click()
        sleep(2)

        verses = driver.find_elements_by_class_name('verse')
        lyrics = ''
        for verse in verses:
            text = verse.text.strip()
            text = re.sub(r"\[.*\]\n", "", unidecode(text))
            if(lyrics == ''):
                lyrics = lyrics + text.replace('\n', '|-|')
            else:
                lyrics = lyrics + '|-|' + text.replace('\n', '|-|')
        print(lyrics)
        row.append(lyrics)
        df.loc[i] = row
        driver.execute_script("window.history.go(-1)")
        sleep(2)
    open('data/{}-lyrics.csv'.format(artist), 'w+')
    df.to_csv('data/{}-lyrics.csv'.format(artist))

artists = [
    # 'cardi-b',
    'travis-scott'
    # 'nicki-minaj',
    # '50-cent',
    # 'wu-tang-clan'
]

threads = []
for artist in artists:
    t = Thread(target = get_lyrics, args = (artist,))
    threads.append(t)
    t.start()

for thread in threads:
    thread.join()


