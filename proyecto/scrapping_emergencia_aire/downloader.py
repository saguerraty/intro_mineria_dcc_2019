#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:35:46 2019

@author: juanpablosuazo
"""

import ssl
from urllib.error import HTTPError
import requests
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains

ssl._create_default_https_context = ssl._create_unverified_context


for i in range(5, 10):
    driver = webdriver.Chrome('/Users/juanpablosuazo/yo/pega/whitestack/tutorials-master/libraries/chromedriver')

    driver.get("http://airesantiago.gob.cl/consolidados-de-calidad-del-aire-ano-201" + str(i) + "/")
    actions = ActionChains(driver)

    reportes = driver.find_elements_by_partial_link_text('Pron')
    urls = []

    for rep in reportes:
            urls.append(rep.get_attribute('href'))

    driver.close()


    for url in urls:
        try:
            urlName = str(url).split('/')
            urlN = urlName[len(urlName)-1]

            r = requests.get(url)

            with open('/Users/juanpablosuazo/UChile/semestre10/mineriaDeDatos/hitos/hito2/reportes/201' + str(i) + '/' + urlN, "wb") as code:
                code.write(r.content)

        except HTTPError:
            print(url)



