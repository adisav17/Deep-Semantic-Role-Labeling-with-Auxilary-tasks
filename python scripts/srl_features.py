## set up features with respect to each word for the nominal semantic role labeling tasks

from collections import OrderedDict
from nltk.stem import *
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from google.colab import files
from joblib import dump, load
import pickle
from gensim.test.utils import datapath

from gensim import models

from gensim.test.utils import datapath

from gensim.test.utils import datapath
import gensim
import gensim.downloader

class Word():
    def __init__(self,word_stem,pos_tag,bio_tag):
        self.bio_tag = bio_tag
        self.word = word_stem
        self.pos_tag = pos_tag

        self.feature_dict = OrderedDict()

        self.feature_list = []

        self.attributes = ["word","pos_tag","bio_tag","prev_word","prev_pos_tag","prev_bio_tag",
        "prev_2_word","prev_2_postag","prev_2_bio_tag","next_word","next_pos_tag","next_bio_tag","next_2_word",
        "next_2_pos_tag","next_2_bio_tag","isbehind","isahead","sent_position","sent_pos_num","ispred", "isarg"]


        self.used_attributes = ["word","pos_tag","bio_tag","prev_word","prev_pos_tag","prev_bio_tag",
        "prev_2_word","prev_2_postag","prev_2_bio_tag","next_word","next_pos_tag","next_bio_tag","next_2_word",
        "next_2_pos_tag","next_2_bio_tag","isbehind","isahead","sent_position"]





        for key in self.attributes:
            self.feature_dict[key] = 'null'



    def get_word_features(self):


        for key in self.used_attributes:
            self.feature_list.append(self.feature_dict[key])

        return self.feature_list

def fill_word_tokens(filename):

    beginning_setence = True
    ending_sentence = False
    Words = []
    stemmer = PorterStemmer()
    fp = open(filename)


    for fline in fp:

        line = fline.split()

        if(not line):
            ending_sentence = True

        if(beginning_setence):

             word = 'SOS'
             pos_tag = 'SOS'
             bio_tag = 'SOS'

             token1 = Word(word,pos_tag, bio_tag)
             token2 = Word(word,pos_tag, bio_tag)

             Words.append(token1)

             Words.append(token2)

             beginning_setence = False


        if(ending_sentence):

            word = 'EOS'
            pos_tag = 'EOS'
            bio_tag = 'EOS'


            token1 = Word(word,pos_tag, bio_tag)
            token2 = Word(word,pos_tag, bio_tag)


            Words.append(token1)

            Words.append(token2)

            ending_sentence = False
            beginning_setence = True

            continue


        word = line[0]
        pos_tag = line[1]
        bio_tag = line[2]

        token = Word(word,pos_tag, bio_tag)

        try:

          if(line[5]=='PRED'):
             token.feature_dict["ispred"]  = 1

        except:
          pass


        try:

          if(line[5]=='ARG1'):
             token.feature_dict["isarg"]  = 1

        except:
          pass


        Words.append(token)


    return Words

def normalize(Word_tokens, sentence_index, sentence_length):

  for i in range(sentence_index,sentence_index + sentence_length):

   # print("sentence index" + " " + str(sentence_index))
   # print("sentence count" + " " + str(sentence_count))
    #print("loop i" +" " + str(i) )
    #print("sentence position" + " " +str(Word_tokens[i].feature_dict["sentence_position"]))
    #print("Word" + " " + Word_tokens[i].word)
    #print(Word_tokens[i].feature_dict["sentence_position"])

    Word_tokens[i].feature_dict["sent_position"] = Word_tokens[i].feature_dict["sent_position"]/sentence_length

def relative_to_sent(Word_tokens, sentence_index, sentence_length):

  for i in range(sentence_index,sentence_index + sentence_length):

   # print("sentence index" + " " + str(sentence_index))
   # print("sentence count" + " " + str(sentence_count))
    #print("loop i" +" " + str(i) )
    #print("sentence position" + " " +str(Word_tokens[i].feature_dict["sentence_position"]))
    #print("Word" + " " + Word_tokens[i].word)
    #print(Word_tokens[i].feature_dict["sentence_position"])

    if(Word_tokens[i].feature_dict["ispred"] == 1):

      for j in range(sentence_index,i):

        Word_tokens[j].feature_dict["isbehind"] = 1

        Word_tokens[j].feature_dict["isahead"] = 0

      for j in range(i+1,sentence_index + sentence_length):

        Word_tokens[j].feature_dict["isbehind"] = 0

        Word_tokens[j].feature_dict["isahead"] = 1

def fill_feature_list_for_words(Words):

  Word_list = []
  Sentences = []

  i=1



  while(i<len(Words)):


    if(Words[i].word =='SOS'):


      beginning_sentence = True
      sentence_count = 0
      i+=1
      eos_flag = 0
      continue



    if(Words[i].word =='EOS'):

      if(eos_flag==0):

         normalize(Words, sentence_index, sentence_count)
         relative_to_sent(Words, sentence_index, sentence_count)
         Sentences.append(Words[sentence_index:sentence_index+ sentence_count])


         eos_flag= 1


      i+=1

      continue



    if(beginning_sentence):

      sentence_index = i
      beginning_sentence = False
      sentence_count = 0



    sentence_count+=1

    Words[i].feature_dict["word"]  = Words[i].word
    Words[i].feature_dict["pos_tag"]  = Words[i].pos_tag
    Words[i].feature_dict["bio_tag"]  = Words[i].bio_tag
    Words[i].feature_dict["sent_position"]  = sentence_count

    Words[i].feature_dict["sent_pos_num"]  = sentence_count


    Words[i].feature_dict["prev_word"]  = Words[i-1].word
    Words[i].feature_dict["prev_pos_tag"]  = Words[i-1].pos_tag
    Words[i].feature_dict["prev_bio_tag"]  = Words[i-1].bio_tag


    Words[i].feature_dict["next_word"]  = Words[i+1].word
    Words[i].feature_dict["next_pos_tag"]  = Words[i+1].pos_tag
    Words[i].feature_dict["next_bio_tag"]  = Words[i+1].bio_tag


    Words[i].feature_dict["prev_2_word"]  = Words[i-2].word
    Words[i].feature_dict["prev_2_pos_tag"]  = Words[i-2].pos_tag
    Words[i].feature_dict["prev_2_bio_tag"]  = Words[i-2].bio_tag


    Words[i].feature_dict["next_2_word"]  = Words[i+2].word
    Words[i].feature_dict["next_2_pos_tag"]  = Words[i+2].pos_tag
    Words[i].feature_dict["next_2_bio_tag"]  = Words[i+2].bio_tag

    Word_list.append(Words[i])

    i+=1


  return Word_list, Sentences

def get_Y_values_bulk(Word_list):

  Y = []
  for i in range(len(Word_list)):
    if(Word_list[i].feature_dict["isarg"]==1):

      Y.append(1)

    else:
      Y.append(0)


  return Y

def get_X_values_bulk(Word_list):

  X = []
  for i in range(len(Word_list)):

    X.append(Word_list[i].get_word_features())


  return X

def fill_feature_list_for_words(Words):

  Word_list = []
  Sentences = []

  i=1



  while(i<len(Words)):


    if(Words[i].word =='SOS'):


      beginning_sentence = True
      sentence_count = 0
      i+=1
      eos_flag = 0
      continue



    if(Words[i].word =='EOS'):

      if(eos_flag==0):

         normalize(Words, sentence_index, sentence_count)
         relative_to_sent(Words, sentence_index, sentence_count)
         Sentences.append(Words[sentence_index:sentence_index+ sentence_count])


         eos_flag= 1


      i+=1

      continue



    if(beginning_sentence):

      sentence_index = i
      beginning_sentence = False
      sentence_count = 0



    sentence_count+=1

    Words[i].feature_dict["word"]  = Words[i].word
    Words[i].feature_dict["pos_tag"]  = Words[i].pos_tag
    Words[i].feature_dict["bio_tag"]  = Words[i].bio_tag
    Words[i].feature_dict["sent_position"]  = sentence_count

    Words[i].feature_dict["sent_pos_num"]  = sentence_count


    Words[i].feature_dict["prev_word"]  = Words[i-1].word
    Words[i].feature_dict["prev_pos_tag"]  = Words[i-1].pos_tag
    Words[i].feature_dict["prev_bio_tag"]  = Words[i-1].bio_tag


    Words[i].feature_dict["next_word"]  = Words[i+1].word
    Words[i].feature_dict["next_pos_tag"]  = Words[i+1].pos_tag
    Words[i].feature_dict["next_bio_tag"]  = Words[i+1].bio_tag


    Words[i].feature_dict["prev_2_word"]  = Words[i-2].word
    Words[i].feature_dict["prev_2_pos_tag"]  = Words[i-2].pos_tag
    Words[i].feature_dict["prev_2_bio_tag"]  = Words[i-2].bio_tag


    Words[i].feature_dict["next_2_word"]  = Words[i+2].word
    Words[i].feature_dict["next_2_pos_tag"]  = Words[i+2].pos_tag
    Words[i].feature_dict["next_2_bio_tag"]  = Words[i+2].bio_tag

    Word_list.append(Words[i])

    i+=1


  return Word_list, Sentences



words_train = fill_word_tokens('partitive_group_nombank.clean.train')
words_test =  fill_word_tokens('partitive_group_nombank.clean.test')
train_list, train_sentences =  fill_feature_list_for_words(words_train)
test_list, test_sentences =  fill_feature_list_for_words(words_test)

Y_train =  get_Y_values_bulk(train_list)
Y_test = get_Y_values_bulk(test_list)
X_train = get_X_values_bulk(train_list)
X_test = get_X_values_bulk(test_list)

X_agg= X_train + X_test

col = ["word","pos_tag","bio_tag","prev_word","prev_pos_tag","prev_bio_tag",
        "prev_2_word","prev_2_pos_tag","prev_2_bio_tag","next_word","next_pos_tag","next_bio_tag","next_2_word",
        "next_2_pos_tag","next_2_bio_tag","isbehind","isahead","sent_position"]

data_agg = pd.DataFrame(X_agg, columns = col)



cols_one_hot = ['pos_tag', 'bio_tag','prev_pos_tag',
        'prev_bio_tag', 'prev_2_pos_tag', 'prev_2_bio_tag','next_pos_tag', 'next_bio_tag', 'next_2_pos_tag', 'next_2_bio_tag', 'isbehind', 'isahead']

cols_agg_words = ['word','prev_word','prev_2_word','next_word','next_2_word']

col_word = ['word']




temp_file = datapath("/content/drive/MyDrive/word_vec_model")
loaded_word_vectors = gensim.models.keyedvectors.Word2VecKeyedVectors.load(temp_file)



def word_embeddings(word):

  try:
    return word_vec[word]

  except:
    return np.zeros(len(word_vec['the']))

def agg_word_embeddings(prev_2_word,prev_word,word,next_word,next_2_word):

   emb = word_embeddings[prev_2_word] + word_embeddings[prev_word] + word_embeddings[word] + word_embeddings[next_2_word] + word_embeddings[next_2_word]
   return emb/5



word_embs = data_agg['word'].apply(word_embeddings)



word_embs_df = pd.DataFrame(word_embs)



agg_word_embs_df = data_agg[cols_agg_words].apply(agg_word_embeddings, axis =1)
