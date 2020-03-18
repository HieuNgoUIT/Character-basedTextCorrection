from gensim.models import Word2Vec

def prepare_sequence(word):
    if word == None: return None
    vowels = ['a', 'i', 'e', 'o', 'u','A','I','E','O','U']
    #if word[0] == ".": return None
    try:
        result = word[0]
    except:
        return None
    for character in range(1,len(word)):
        if word[character] in vowels and word[character-1] in vowels \
           or word[character] not in vowels and word[character-1] not in vowels:
            result = result + word[character]
        else:
            result = result + " " + word[character]
    my_list = result.split(" ")
    sub_list = []
    for i in my_list:
        sub_list.append([i])
    return sub_list
        
# f = open("test.txt", "r")
# paragraph = f.readlines()
# f.close() 
# print(paragraph)


# sentences = " ".join(str(x) for x in paragraph)

# import re

# sentences = re.sub(r'[^\w\s]','',sentences)
#print(sentences)

# words = sentences.split(" ")
#words = words.remove('')
# print(words)

"""
write charater to file
"""

from nltk.corpus import brown

words = " ".join(brown.words()[:20000])
words = words.split(" ")
#print(sentences)
# print(type(sentences))

#wtest = ['Today', 'is', 'a', 'good', 'day']

result = []
for i in words:
    print(i)
    result = result + (prepare_sequence(i)) 


# f = open("character.txt","w")
# for i in range(len(result)):
#     f.write(str(result[i]) + " ")
# f.close()

"""
____________________________________
"""
#print(result[])

#f = open("character.txt","r")

#characters = f.read()

#f.close()
#print(type(characters))
#mylist = characters.split(" ")
#print(mylist)


import numpy as np 
kq = np.array(result)
#kq = kq.reshape(len(kq),1)
# #print(kq[:,:])
print(kq.shape)



w2v = Word2Vec(sentences=kq, size=100, window=4, iter=10)
# vocab = w2v.wv.vocab
# print(type(vocab))
# print(vocab)
#vector = w2v.wv["ao"] #get vector by character
#print(vector)

#for index, k in enumerate(vocab.keys()):
"""
find distance between 2 words
# """
# charw1 = prepare_sequence("hello")
# charw2 = prepare_sequence("hello")

# print(charw1)
# print(type(charw1))
# print(charw2)

# wordVector = np.zeros((1,100))
# for i in charw1:
#     print("sum",wordVector.shape)
#     print("cha",w2v.wv[i].shape)
#     wordVector = np.add(wordVector, w2v.wv[i])
#     print(i)

# wordVector2 = np.zeros((1,100))
# for i in charw2:
#     print("sum",wordVector2.shape)
#     print("cha",w2v.wv[i].shape)
#     wordVector2 = np.add(wordVector2, w2v.wv[i])
#     print(i)

# from scipy import spatial
# result = 1 - spatial.distance.cosine(wordVector, wordVector2)
# print(result)  
#print(w2v.wv['h'])
#print(w2v.wv[['h']])

# print(wordVector[:2,:])
# print(wordVector.shape)
# print(w2v.wv['h'][:2])
# print(w2v.wv['e'][:2])
# print(w2v.wv['ll'][:2])
# print(w2v.wv['o'][:2])
"""
________________________________________________________________
"""
sumVector = np.zeros((1,100))
WandV = {}
for w in words: 
    print(w)
    chars = prepare_sequence(w)
    for c in chars:
        print(chars)
        try:
            vectorInVocab = w2v.wv[c]
            sumVector = np.add(sumVector, vectorInVocab)
        except:
            sumVector = None      
    WandV[w] = sumVector
    sumVector = np.zeros((1,100))
print(len(WandV))
