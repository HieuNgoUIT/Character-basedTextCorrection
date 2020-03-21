from gensim.models import Word2Vec
import numpy as np 
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
data = ""
for entry in entries:
    path = "./corpus.viwiki/mysmalltest/" + entry
    with open(path, encoding="utf8") as f:
        currentdata = f.read()
    data = data + currentdata
print("Loading data from file .........")

#print(data)
# with open("./corpus.viwiki/viwiki/chính trị.txt") as f:
#     data = f.read()

#print(data)

from pyvi import ViTokenizer

print('Preprocessing word .....')
wordsstr = ViTokenizer.tokenize(data).lower()
words = wordsstr.split(" ")

print('Loading word2vec model......')
model = Word2Vec.load("word2vec.model") #character to vec


print('Creating word vector in vocabulary....')
sumVector = np.zeros((1,100))
WandV = {}
for w in words: 
        #print(w)
        chars = prepare_sequence(w)
        for c in chars:
            #print(chars)
            try:
                vectorInVocab = model.wv[c]
                sumVector = np.add(sumVector, vectorInVocab)
            except:
                sumVector = None    
        try:
            WandV[w] = sumVector.flatten()
            #print(WandV[w])
        except:
            WandV[w] = np.zeros(100)  
        #print(sumVector.flatten())
        #print(sumVector.flatten().shape)
        #print(type(sumVector.flatten()))
        sumVector = np.zeros((1,100))

# for i in WandV.keys():
#     print(i)
#     print(WandV[i])
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.DataFrame()

for i in WandV.values():
    #print(pd.DataFrame(i))
    df = df.append(pd.Series(i), ignore_index=True)
#print(df.shape)
df = df[(df.T != 0).all()]
#print(df.shape)

print('Training model KNN .........')
from sklearn.neighbors.ball_tree import BallTree

tree = BallTree(df, leaf_size=2)

#print(df.shape) # 7839 100


index = np.expand_dims(df.iloc[69,:], axis =0)

dist, ind = tree.query(index, k=3)                # doctest: +SKIP
print(ind)  # indices of 3 closest neighbors
#[0 3 1]
print(dist)  # distances to 3 closest neighbors
# #[ 0.          0.19662693  0.29473397]

v1 = df.iloc[ind[:,0],:]
v2 = df.iloc[ind[:,1],:]
v3 = df.iloc[ind[:,2],:]

V1 = np.array(v1)
V2 = np.array(v2)
V3 = np.array(v3)


for k,v in WandV.items():
    comparison = v == V1
    if comparison.all() == True:
        pass
        print(k)
    comparison2 = v == V2
    if comparison2.all() == True:
        pass
        print(k)
    comparison3 = v == V3
    if comparison3.all() == True:
        pass
        print(k)        
# print("________________________________")
# print(df[:1])
# print(df.iloc[0,:])
# print(df[:1].shape)