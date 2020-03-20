import pandas as pd
import numpy as np  
from pyvi import ViUtils
#print(prepare_sequence("Chính_trị"))
#print(ViUtils.add_accents('toi la trung hieu dep trai'))


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



#print(prepare_sequence("Xin_chào"))


df = pd.DataFrame(np.array([[1,2,3], [4,5,6], [7,8,9]]))
print(df)
print(df.shape)
print("____________________")
print(df[:1])
print("shape of df[:1]", df[:1].shape)
print("_____________________")
print(df.iloc[0,:])
print("shape of dfiloc[0,:]", df.iloc[0,:].shape)
print("________________")
a = df.iloc[0,:]
a = np.expand_dims(a, axis=0)
print(a)
print(a.shape)
# X i n 