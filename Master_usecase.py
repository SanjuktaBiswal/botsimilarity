import numpy as np
import pandas as pd
import re
import string
import pickle
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
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


#df = pd.read_excel("C:/Users/Admin/Documents/Capgemini_projects/Microbot/Deployment.xlsx")
#print(df.head())
#print(df.shape)
#print(df['Use case'].isnull().sum())
#print(df["MasterUseCaseMapping"].isnull().sum())

def read_master_data(path):
    master_data= pd.read_excel(path)
    return master_data

#df = data("C:/Users/Admin/Documents/Capgemini_projects/Microbot/Deployment.xlsx")"""

def preproc(line):
    # '''
    # Description: This script is used to pre-processed ticket summary and remove unwanted charaters from the data.
    # line: single text document at a time
    # '''
    ## Note: Note removing alpha neumeric charecters as ISIM data has lot of machine data. For other data you may choose to
    # remove alpha neumeric chars.

    line = str(line)  # convert into string
    # convert french chars to latin equivalents
    # line = normalize('NFD', line).encode('ascii', 'ignore')
    # line = line.decode('UTF-8')

    line = re.sub(r'[^<a-zA-Z0-9>][\d]+', ' ', line)

    line = re.sub('\w*[0-9]\w*', ' ', line)

    line = re.sub(r"INC+\d+", "", line)
    line = re.sub(r"REQ+\d+", "", line)
    line = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', "#URL#", line)
    line = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "#email#", line)
    line = re.sub(r"PO+\d+", "", line)
    # Remove date/timestamp
    line = re.sub(r"^\\d{4}[-]?\\d{1,2}[-]?\\d{1,2} \\d{1,2}:\\d{1,2}:\\d{1,2}[,]?\\d{1,3}$", "", line)

    ### Start Added based on training data. ###
    line = line.lower()
    line = line.replace('nom_person', '')
    line = line.replace('alphanumeric_id', '')
    line = line.replace('num_telephone', '')
    line = line.replace('adresse_mail', '')
    line = line.replace('adresse_ip', '')
    line = line.replace('adresse_web', '')
    line = line.replace('decommision', 'decommission')
    line = line.replace('&', '')
    line = line.replace('cretaes', 'creates')
    line = line.replace('plt', 'plot')
    line = line.replace('-', '')
    # line = line.replace('py','')
    line = line.replace('nan', '')

    # Replace punctuations(except [  and ] ) with space
    line = re.sub(r"[-()\"#/@;:<>+=_~|{}.?,]", " ", line)

    # Replace punctuations( [  and ] ) with space
    stoplist = list(string.punctuation)
    token = [word for word in word_tokenize(line) if word not in stoplist]
    strline = ' '.join(token)
    strline = re.sub(r" +", " ", strline)

    # Split combined words into separate words from dictionary words.
    # line = replaceByDict(strline,area_dict)

    return line


def preprocess(text):
    # Remove punctuation
    REPLACE_BY_SPACE_RE = re.compile(r'''[=\+"\/()<>{}\*\[\]\|@,;\\\:_\-\.\'!%$1]''')
    text = REPLACE_BY_SPACE_RE.sub(' ', str(text))
    # text = REPLACE_BY_SPACE_RE
    # Convert the titles to lowercase
    stopwordlist = stopwords.words('english')
    # stopwordlist.append('bot')
    sline = [word.lower() for word in text.split() if word.lower() not in stopwordlist and not word.isdigit()]
    # stemmer = SnowballStemmer('english')
    # sline = [stemmer.stem(word) for word in sline if len(word)>1]
    # print ('After stemming', sline)

    strline = ' '.join(sline)
    strline = re.sub(r" +", " ", strline)  # convrt multiple space into one space
    strline = str.strip(strline)  # removes all the leading and trailing spaces from a string
    # print (strline)

    return strline


def master_embedding(df_series, glove_file_path):
    # Find the maxlength of the list
    MAX_SEQUENCE_LENGTH=10
    totalwordcnt = 0
    for eachSentence in df_series:
        wordCount = len(re.findall(r'\w+', eachSentence))
        totalwordcnt = totalwordcnt + wordCount
        if wordCount > MAX_SEQUENCE_LENGTH:
            MAX_SEQUENCE_LENGTH = wordCount
    MAX_NUM_WORDS=totalwordcnt
    #print ('MAX_SEQUENCE_LENGTH', MAX_SEQUENCE_LENGTH)
    #print("MAX_NUM_WORDS", MAX_NUM_WORDS)

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,filters='!')
    tokenizer.fit_on_texts(df_series)
    sequences = tokenizer.texts_to_sequences(df_series)
    word_index = tokenizer.word_index
    #print('Found {} unique tokens but initial config is {} words'.format(len(word_index),MAX_NUM_WORDS))
    MAX_NUM_WORDS = len(word_index)
    #print ("MAX_NUM_WORDS becomes:",MAX_NUM_WORDS)
    new_df = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Build index mapping words in the embeddings set to their embedding vector
    #print('Indexing word vectors.')


    embedding_dict={}

    with open (glove_file_path,encoding='utf-8') as f:
        for line in f:
            values=line.split()
            word=values[0]
            coef=np.asarray(values[1:], dtype = 'float32')
            embedding_dict[word]=coef
        #f.close()
    #print('Found %s word vectors.' % len(embedding_dict))

    return embedding_dict

#embedding_dict = embedding(df)

def related_usecase(search_bar,data, glove_file_path):
    line = search_bar
    text = preproc(line)
    ntext = preprocess(text)
    q1 = ntext.split()
    # print(q1, '\n')
    zero_vec = np.zeros((1, 150))
    embedding_dict = master_embedding(data, glove_file_path)
    for i in q1:
        zero_vec = (embedding_dict[i].reshape(1, 150)) + zero_vec

    similarity_lst = list()
    for line in data:
        conv = line.split()
        zero_vec2 = np.zeros((1, 150))
        notaval = list()
        for word in conv:
            if word not in embedding_dict:
                notaval.append(word)
            else:
                zero_vec2 = (embedding_dict[word].reshape(1, 150)) + zero_vec2

        cos_parameter_A = zero_vec
        cos_parameter_B = zero_vec2
        similarity = (cosine_similarity(cos_parameter_A, cos_parameter_B))
        similarity_lst.append(similarity[0][0])

    return similarity_lst

def top_master_similarity_output(search_bar,input_file_path, glove_file_path):
    input_df = read_master_data(input_file_path)
    input_df['cdata'] = list(map(preproc, input_df['MasterUseCaseMapping']))
    input_df['clean_MasterUseCase'] = list(map(preprocess, input_df["cdata"]))
    data = input_df["clean_MasterUseCase"]
    input_df["Similarity_index"] = related_usecase(search_bar, data, glove_file_path) # calling function
    output_df = input_df.sort_values("Similarity_index", ascending=False)
    output_df.reset_index(inplace=True, drop=True)
    master_table_cl = (output_df[["DeploymentID", "clean_MasterUseCase", "Similarity_index"]])
    master_table_columns = master_table_cl[(master_table_cl.Similarity_index >= 0.5)]

    return master_table_columns



#search_bar=input("enter")
#call_funtion = top_master_similarity_output(search_bar,"C:/Users/Admin/Documents/Capgemini_projects/Microbot/Deployment.xlsx", "C:/Users/Admin/Documents/Capgemini_projects/Microbot/haseeb/bot_dict.txt")
#print(call_funtion)
#call = string_search_master(search_bar.lower(), "C:/Users/Admin/Documents/Capgemini_projects/Microbot/Deployment.xlsx")
#print(call)