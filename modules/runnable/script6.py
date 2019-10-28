import urllib.request

def getHtml(url):
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    html = response.read()
    return html

def saveHtml(file_name,file_content):
    with open(file_name.replace('/','_')+'.html','wb') as f:
        f.write(file_content)


url = 'https://www.virustotal.com/gui/file/968c37e74571c6f3bf8f2749c9e1d0ea6999eb503de2a9a6cc78c68530559c6d/detection.html'
html = getHtml(url)
saveHtml("C:/Users/Asichurter/Desktop/dl_malwares/test",html)
print("结束")