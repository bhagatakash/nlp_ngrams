import pandas as pd

import re
# Gensim
import gensim  #gensim==4.1.2
import logging
import warnings


# NLTK Stop words
from nltk.corpus import stopwords #nltk==3.6.5
from sklearn.feature_extraction.text import CountVectorizer
import pickle
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

global topics_

#ldy=pd.read_sql('select DataDate from [dbo].[Nlp_DailyTopics] order by DataDate desc', con=engine)['DataDate'][0]
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


def sent_to_words(sentences):
    for sent in sentences:
        sent=str(sent)
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  



pkl_file = open('df.pkl', 'rb')
df = pickle.load(pkl_file)
pkl_file.close()



             
print(df.shape)  #> (2361, 3)
df=df.dropna(how='any')    #to drop if any value in the row has a nan
df = df.drop_duplicates(subset = ['FullContent'], keep = 'last').reset_index(drop = True)


# Convert to list
data = df.FullContent.values.tolist()
data = set(data)


data_words = list(sent_to_words(data))


c_vec = CountVectorizer(stop_words=stop_words, ngram_range=(2,2))


ngrams = c_vec.fit_transform(df['FullContent'])
vocab = c_vec.vocabulary_
count_values=ngrams[:5000].toarray().sum(axis=0)
for i in range(5000,ngrams.shape[0],5000):
    ct=ngrams[i:i+5000].toarray().sum(axis=0)
    
    count_values=ct+count_values
    


df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)).rename(columns={0: 'frequency', 1:'bigram'})
df_ngram = df_ngram[df_ngram['frequency']>5]
df_dict = dict(zip(df_ngram.bigram, df_ngram.frequency))

col=['Bigram','Count','DataDate', 'DateFetched','TotalNews', 'YesterDayMatch', 'PreviousMonthMatch','PreviousYearMatch']

dct={'Bigram':df_dict,'Count':len(df_dict), 'TotalNews':len(data), 'YesterDayMatch':'', 'PreviousMonthMatch':"",'PreviousYearMatch':''} 


   
  

   
   
   
   
   

             
