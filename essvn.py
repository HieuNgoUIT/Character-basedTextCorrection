vnese = ['a', 'á', 'à', 'ả', 'ã', 'ạ', \
         'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', \
         'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', \
         'e', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', \
         'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', \
         'i', 'í', 'ì', 'ỉ', 'ĩ', 'ị', \
         'o', 'ó', 'ò', 'ỏ', 'õ', 'ọ', \
         'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', \
         'u', 'ú', 'ù', 'ủ', 'ũ', 'ụ', \
         'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', \
         'y']              
en = ['a', 'i', 'u', 'e', 'o', 'A', 'I', 'U', 'E', 'O']

def prepare_sequence(words):
    if words == None: return ""
    vowels = vnese 
    my_list = []
    word = words.split("_")           
    for w in word:
        w = w.lower()
        l = len(w)
        try :
            result = w[0]
        except:
            return ""
        for character in range(1,l):
            if w[character] in vowels and w[character-1] not in vowels  \
            or w[character] not in vowels and w[character-1] in vowels:
                my_list.append(list([result]))
                result = w[character]
            else:
                result = result + w[character]
        my_list.append(list([result]))
    return my_list


import os
entries = os.listdir("./corpus.viwiki/mysmalltest")
#print(entries)
data = ""
for entry in entries:
    path = "./corpus.viwiki/mysmalltest/" + entry
    with open(path) as f:
        currentdata = f.read()
    data = data + currentdata
print("Loading data from file .........")

#print(data)
# with open("./corpus.viwiki/viwiki/chính trị.txt") as f:
#     data = f.read()

#print(data)

from pyvi import ViTokenizer

wordsstr = ViTokenizer.tokenize(data)
words = wordsstr.split(" ")
#print(words) list 1d

result = []
for i in words:
    print(i)
    try:
        result = result + prepare_sequence(i)
    except:
        pass
    #print(result)
#print(result)
import numpy as np 
kq = np.array(result)
kq = kq.reshape(len(kq),1)
# print(kq[:,:])
#print(kq.shape)
#print(kq[:5,:]) 

from gensim.models import Word2Vec
w2v = Word2Vec(sentences=kq, size=100, window=4, iter=50)
w2v.save("word2vec.model")
# print(w2v.wv.vocab)

# #temp = w2v.wv.distance("chính_trị","Chính_trị")
# #print(temp)

# sumVector = np.zeros((1,100))
# WandV = {}
# #print(w2v.wv['p'])
# #print(type(w2v.wv['p']))
# for w in words: 
#     try:
#         WandV[w] = w2v.wv[w]
#     except:
#         WandV[w] = None
# import pandas as pd
# df = pd.DataFrame()

# for i in WandV.values():
#     #print(pd.DataFrame(i))
#     df = df.append(pd.Series(i), ignore_index=True)
# from sklearn.neighbors.ball_tree import BallTree

# tree = BallTree(df, leaf_size=2)

# dist, ind = tree.query(df[:1], k=3)                # doctest: +SKIP
# print(ind)  # indices of 3 closest neighbors
# #[0 3 1]
# print(dist)  # distances to 3 closest neighbors
# #[ 0.          0.19662693  0.29473397]

# v1 = df.iloc[0,:]
# v2 = df.iloc[ind[:,1],:]
# v3 = df.iloc[ind[:,2],:]

# V1 = np.array(v1)
# V2 = np.array(v2)
# V3 = np.array(v3)

# # print(V1)

# # print(WandV['The'])


# #print(v1)
# #print(type(v1))

# for k,v in WandV.items():
#     comparison = v == V1
#     if comparison.all() == True:
#         print(k)
#     comparison2 = v == V2
#     if comparison2.all() == True:
#         print(k)
#     comparison3 = v == V3
#     if comparison3.all() == True:
#         print(k)        
    