f=open('mobydick.txt','r',encoding='utf-8')
testfile=open('testfile.txt','w',encoding='utf-8')

for line in f:
    testfile.write(line)
    if "*** START OF THE PROJECT GUTENBERG EBOOK MOBY-DICK; OR THE WHALE ***" in line:
        break
wordcount={}
for line in f:
    if "*** END OF THE PROJECT GUTENBERG EBOOK MOBY-DICK; OR THE WHALE ***" in line:
        break
    line=line.strip().lower()
    line=line.replace("&",'and').replace("--",' ').replace('\'','')
    for c in '.,-\"\'?!:;()*[]_':
        line.replace(c,'')
    line = line.replace('“','')
    words=line.split(' ')
    for word in words:
        #testfile.write(word+'\n')
        if not word:
            continue
        try:
            wordcount[word]+=1
        except ValueError:
            wordcount[word]=1

wc=[]
for k,v in wordcount.items():
    wc.append((v,k))

for j in range(len(wc)):
    testfile.write(str(wc[j][1])+"  "+str(wc[j][0])+"\n")
    



f.close()
testfile.close()