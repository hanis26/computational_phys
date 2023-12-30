f=open('mobydick.txt','r',encoding='utf-8')

for line in f:
    if "*** START OF THE PROJECT GUTENBERG EBOOK MOBY-DICK; OR THE WHALE ***":
        break

wordcount={}

for line in f:
    if "*** END OF THE PROJECT GUTENBERG EBOOK MOBY-DICK; OR THE WHALE ***":
        break

line = line.strip().lower()
line=line.replace('--','').replace('\s','').replace('&','and')
for c in '!?":;,().*[]':  #when you iterate over a string, it takes it character by character 
    line = line.replace(c,'')
words=line.split(' ')
for word in words:
    if not word:
        continue
try:
    wordcount[word] +=1
except KeyError:
    wordcount[word]=1


print(wordcount)
