#importing libraries
import numpy as np
import pandas as pd
import re
import string
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#from data_preprocessing import preproc
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
def preproc(line):   
   # '''
   #Description: This script is used to pre-processed ticket summary and remove unwanted charaters from the data.
   #line: single text document at a time
   #'''
    ## Note: Note removing alpha neumeric charecters as ISIM data has lot of machine data. For other data you may choose to 
      #remove alpha neumeric chars.
        
    line = str(line) # convert into string
    # convert french chars to latin equivalents
    #line = normalize('NFD', line).encode('ascii', 'ignore')
    #line = line.decode('UTF-8')
    
    line = re.sub(r'[^<a-zA-Z0-9>][\d]+',' ', line)
    
    line = re.sub('\w*[0-9]\w*', ' ', line)
    
    line = re.sub(r"INC+\d+","", line)
    line = re.sub(r"REQ+\d+","", line)
    line = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',"#URL#", line)
    line = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+","#email#", line)
    line = re.sub(r"PO+\d+","", line)
    #Remove date/timestamp
    line = re.sub(r"^\\d{4}[-]?\\d{1,2}[-]?\\d{1,2} \\d{1,2}:\\d{1,2}:\\d{1,2}[,]?\\d{1,3}$", "", line)
    
    ### Start Added based on training data. ###
    line = line.lower()
    line = line.replace('nom_person','')
    line = line.replace('alphanumeric_id','')
    line = line.replace('num_telephone','')
    line = line.replace('adresse_mail','')
    line = line.replace('adresse_ip','')
    line = line.replace('adresse_web','')
    line = line.replace('&','')
    line = line.replace('cretaes','creates')
    line = line.replace('plt','plot')
    line = line.replace('-','')
    line = line.replace('py','')
    line = line.replace('nan','')

    #Replace punctuations(except [  and ] ) with space
    line = re.sub(r"[-()\"#/@;:<>+=_~|{}.?,]", " ", line)
    
    #Replace punctuations( [  and ] ) with space
    stoplist = list(string.punctuation)
    token=[word for word in word_tokenize(line) if word not in stoplist]
    strline =' '.join(token)
    strline = re.sub(r" +"," ", strline)    

    # Split combined words into separate words from dictionary words.
    #line = replaceByDict(strline,area_dict)

    return line

def preprocess(text):

    # Remove punctuation
    REPLACE_BY_SPACE_RE = re.compile(r'''[=\+"\/()<>{}\*\[\]\|@,;\\\:_\-\.\'!%$1]''')
    text = REPLACE_BY_SPACE_RE.sub(' ',str(text))
    #text = REPLACE_BY_SPACE_RE
    # Convert the titles to lowercase
    stopwordlist = stopwords.words('english')
    #print(stopwordlist)
    #stopwordlist.append('bot')
    sline = [word.lower() for word in text.split() if word.lower() not in stopwordlist and not word.isdigit()]
    #stemmer = SnowballStemmer('english')
    #sline = [stemmer.stem(word) for word in sline if len(word)>1]
    #print ('After stemming', sline)  
    
    strline = ' '.join(sline) 
    strline = re.sub(r" +"," ", strline)#convrt multiple space into one space
    strline = str.strip(strline)#removes all the leading and trailing spaces from a string
    #print (strline)

    return strline

def similar_botdesc(embedding_dict,bot_data,query):
    
    line = query
    text = preproc(line)
    ntext = preprocess(text)
    q1 = ntext.split()
    #print(q1, '\n')
    a = np.zeros((1,150))
    for i in q1:
        a = (embedding_dict[i].reshape(1,150)) + a
    #print(a)

    #sim_val = dict()
    similarity_lst= list()
    for line in bot_data:
        conv = line.split()
        #print(conv)
        a_v = np.zeros((1,150))
        notaval = list()
        for word in conv:
            #print(word)
            if word not in embedding_dict:
                notaval.append(word)
            else:
                a_v = (embedding_dict[word].reshape(1,150)) + a_v
        #print(a_v, a_v.shape)
        #print(notaval)
        #coef=np.asarray(values[1:], dtype = 'float32')
        #sim_val[line]=a_v
    

        A = a
        B = a_v
        #print(A,'\n', B)
        similarity = (cosine_similarity(A, B))
        #conv = similarity.tolist()
        similarity_lst.append(similarity[0][0])
        #sim_val[line]= conv
        #print(similarity, type(similarity))
        #print(conv)
        #sim_val.append(conv)
        
    return similarity_lst

def create_sent_vect(q):
    q1 = q.split()
    #print(q1)
    a = np.zeros((1,150))
    for i in q1:
        a = (embedding_dict[i].reshape(1,150)) + a
    #print(a)
    return a
def create_sent_vect(q):
    q1 = q.split()
    #print(q1)
    a = np.zeros((1,150))
    for i in q1:
        a = (embedding_dict[i].reshape(1,150)) + a
    #print(a)
    return a
def queryprocessing(query):
    df = pd.read_excel("data/Deployment.xlsx",sheet_name='bots')
    df['cdata'] = list(map(preproc,df["Bot Desc"]))
    df['clean_Bot Desc'] = df['cdata'].apply(lambda x: preprocess(str(x)))
    df_original=df.copy()
    # Find the maxlength of the list
    MAX_SEQUENCE_LENGTH=10
    for eachSentence in df['clean_Bot Desc']:
        wordCount = len(re.findall(r'\w+', eachSentence))
    #    totalwordcnt = totalwordcnt + wordCount
        if wordCount > MAX_SEQUENCE_LENGTH:
            MAX_SEQUENCE_LENGTH = wordCount
    MAX_NUM_WORDS=5000
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,filters='!')
    tokenizer.fit_on_texts(df['clean_Bot Desc'])
    sequences = tokenizer.texts_to_sequences(df['clean_Bot Desc'])
    word_index = tokenizer.word_index
    MAX_NUM_WORDS = len(word_index)    
    new_df = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    embedding_dict={}
    with open ("data/bot_dict.txt",encoding='utf-8') as f:
        for line in f:
            values=line.split()
            word=values[0]
            coef=np.asarray(values[1:], dtype = 'float32')
            embedding_dict[word]=coef
    lst = []
    for word, i in word_index.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is None:
            lst.append(word)
    bot_data = df["clean_Bot Desc"]
    df["Similarity_index"] = similar_botdesc(embedding_dict,bot_data,query)#similarity_lst
    df = df.sort_values("Similarity_index", ascending = False)
    df.reset_index(inplace= True, drop = True)
    '''
    sentence = "this bot will consolidate both mbs "
    detail = df[df["clean_Bot Desc"].str.contains(sentence, na = False)]
    rfile = df['clean_Bot Desc']
    q='this bot will consolidate both mbs'
    q1 = q.split()
    
    a = np.zeros((1,150))
    for i in q1:
        a = (embedding_dict[i].reshape(1,150)) + a
    #print(a)
    
    sim_val = dict()
    for line in rfile:
        conv = line.split()
        #print(conv)
        a_v = np.zeros((1,150))
        notaval = list()
        for word in conv:
            #print(word)
            if word not in embedding_dict:
                notaval.append(word)
            else:
                a_v = (embedding_dict[word].reshape(1,150)) + a_v
        
        A = a
        B = a_v
        #sim_val = list()
        similarity = (cosine_similarity(A, B))
        conv = similarity.tolist()
        sim_val[line]= conv
        
    ls = (sorted(sim_val.items(), key=lambda x: x[1], reverse = True)[:10])
    filenames = ['data/Eng_Email_GTD_Glove_100D_08Oct20.txt', 'data/vectors.txt']
    with open('data/bot_dict.txt', 'w', encoding="utf8") as outfile:
        for fname in filenames:
            with open(fname, encoding="utf8") as infile:
                outfile.write(infile.read())
    for line in rfile:
    conv = line.split()
    print(conv)
    a_v = np.zeros((1,150))
    notaval = list()
    for word in conv:
        print(word)
        if word not in embedding_dict:
            notaval.append(word)
        else:
            a_v = (embedding_dict[word].reshape(1,150)) + a_v
    q1 = 'get user details'#input("Enter the query1: \n")
    A = create_sent_vect(q1)
    
    q2 = 'get tablespace details'#input("\nEnter the query2: \n")
    B = create_sent_vect(q2)
    q='get user details'
    q1 = q.split()
    
    a = np.zeros((1,150))
    for i in q1:
        a = (embedding_dict[i].reshape(1,150)) + a
    #print(a)
    
    sim_val = list()
    for line in rfile:
        conv = line.split()
        #print(conv)
        a_v = np.zeros((1,150))
        notaval = list()
        for word in conv:
            #print(word)
            if word not in embedding_dict:
                notaval.append(word)
            else:
                a_v = (embedding_dict[word].reshape(1,150)) + a_v
        #print(a_v, a_v.shape)
        #print(notaval)
    
        A = a
        B = a_v
        #sim_val = list()
        similarity = (cosine_similarity(A, B))
        conv = similarity.tolist()
        #print(type(similarity))
        sim_val.append(conv)
    ls = sorted(sim_val, reverse = True)
    '''
    return df_original,df

