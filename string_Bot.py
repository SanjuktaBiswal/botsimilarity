import re
import string
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords

def read_master_data(path):
    master_data= pd.read_excel(path,sheet_name="bots")
    return master_data

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

def string_search_bot(search_bar,input_file_path):
    input_df = read_master_data(input_file_path)
    input_df['cdata'] = list(map(preproc, input_df['Bot Desc']))
    input_df['clean_Bot Desc'] = list(map(preprocess, input_df["cdata"]))
    new_input_df = (input_df[["DemandID", "BoTID", "clean_Bot Desc"]].dropna())

    text = preproc(search_bar)
    ntext = preprocess(text)
    data = new_input_df[new_input_df["clean_Bot Desc"].str.contains(ntext)]
    data.reset_index(inplace=True, drop=True)

    return new_input_df,data

#search_bar=input("enter")
#call = string_search_bot(search_bar.lower(), "C:/Users/Admin/Documents/Capgemini_projects/Microbot/bots.xlsx")
#print(call)