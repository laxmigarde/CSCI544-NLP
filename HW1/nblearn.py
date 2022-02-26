#!/usr/bin/env python
# coding: utf-8
# new approach without using numpy

import os
import string
import sys
from collections import Counter, OrderedDict, defaultdict

import regex

# ## Constants
MODEL_FILE = "nbmodel.txt"

CLASSES = ["positive", "negative", "truthful", "deceptive"]

# #### Train
# Stores the number of data points available for each class
# class_count = {cls: 0 for cls in CLASSES}
class_count = {'positive': 0, 'negative': 0, 'deceptive': 0, 'truthful': 0}

# Stores the prior probability of each class
# priors = OrderedDict({cls: 0 for cls in CLASSES})
priors = {}

# Stores word frequencies for each class
cls_vocab = {cls: set() for cls in CLASSES}

# Stores the probability of words for each class
cls_word_probabilities = defaultdict(lambda: defaultdict(float))

# ## Stopwords
STOP_WORDS = [
    "a", "an", "am", "at", "are", "after", "above", "about", "against", "all", "any", "and", "as", "again",
    "but", "both", "between", "before", "below", "be", "been", "being", "because", "by", "couldnt",
    "could", "cant", "can", "do", "does", "doesnt", "did", "doing", "didnt", "during", "down", "don", "dont",
    "each", "for", "from", "further", "few", "go", "get", "he", "hell", "him", "his", "himself",
    "i", "ive", "ill", "im", "id", "is", "it", "itll", "in", "into", "it", "its", "itself", "if", "isnt",
    "just", "there", "theres", "therell", "the", "they", "theyd", "theyve", "they", "them", "their",
    "theirs", "themselves", "this", "that", "these", "those", "then", "to", "through", "than", "too",
    "whose", "who", "wholl", "whod", "was", "wasnt", "were", "werent", "well", "were", "whatll", "would",
    "we", "will", "when", "where", "what", "which", "who", "whom", "wasnt", "while", "with", "why", "whyll", "wont",
    "how", "have", "has", "had", "having", "heres", "here", "howll", "hadnt", "her", "hers", "herself",
    "lets", "let", "my", "me", "myself","mustve", "more", "most", "mostly", "l", "s", "t",
    "no", "nor", "not", "now", "onto", "our", "ours", "ourselves", "of", "out", "on", "off", "over", "once", "or", "other", "own", "only",
    "you", "youll", "youed", "yours", "youre", "wouldve","wouldnt", "yourself", "yourselves",
    "shell", "shed", "hed", "she", "should", "shouldnt", "same", "similar", "some", "such", "so", "until", "up", "under", "very", "next", "btw",
    "went", "ago", "able", "ha", "na", "nah", "ma", "thats", "one", "two", "try", "i-", "arrive", "arriving", "arrived", "stay", "stayed"
]

def dataloader(path):
  for root, _, files in os.walk(path, topdown=False):
    for file_name in files:
      # Only parse text files, used to ignore .DS_store file on macOS
      if '.txt' in file_name and file_name != "README.txt":
        file_path = os.path.join(root, file_name)
        # print(file_path)
        get_valid_tokens = do_preprocessing(file_path)
        # print(get_valid_tokens)
    return get_valid_tokens


# ## Preprocess the data
def to_lowercase(data):
  return data.lower().split()

def remove_punctuations(data):
  data_without_punctations = []
  for val in data:
      val = regex.sub('[^a-zA-Z0-9\n\-]' , '' , val)
      data_without_punctations.append(val)
    # if val not in string.punctuation:
    #     data_without_punctations.append(val)
  return data_without_punctations

def filter_out_stopwords(data):
    data_without_stopwords = []
    for words in data:
        if not words in STOP_WORDS:
            data_without_stopwords.append(words)
    return data_without_stopwords

def remove_digits(data):
    data_without_digits = []
    for numbers in data:
        numbers = regex.sub('[0-9]', '', numbers)
        data_without_digits.append(numbers)
    return data_without_digits


# ## Implement Naive Bayes
# #### Smoothing
def additive_smoothing(vocab, alpha = 1):
  for key in vocab.keys():
    vocab[key] += alpha
  return vocab

# ## call preprocessing steps
def do_preprocessing(file_path):
    with open(file_path, mode="r") as file:
        data = file.read()
        data_tokens = to_lowercase(data)
        data_tokens_without_punctations = remove_punctuations(data_tokens)
        data_without_digits = remove_digits(data_tokens_without_punctations)
        data_without_stopwords = filter_out_stopwords(data_without_digits)
    return data_without_stopwords

# ## calc. prior and conditional probabilities
def train(p_dict, n_dict, t_dict, d_dict, no_of_documents, vocabulary):
    #print("no_of_docs:", no_of_documents)
    # print(vocabulary)
    p_count = {}
    n_count = {}
    t_count = {}
    d_count = {}

    p_condProb = {}
    n_condProb = {}
    t_condProb = {}
    d_condProb = {}

    # p_dict = {}
    # n_dict = {}
    # t_dict = {}
    # d_dict = {} 

    """Takes in preprocessed train data
  Args:
      train_data: Preprocessed train data
  """
    for class_type in CLASSES:
        priors[class_type] = class_count[class_type] / no_of_documents
        # print("priors:: " , priors[class_type])
        
        if class_type == 'positive':
          for word in vocabulary:
            if word in p_dict:
              p_count[word] = p_dict[word]
            else:
              p_count[word] = 0
          # word_probability = probabilityCalc(vocabulary, p_count, p_dict, p_condProb)
          wordcount_class = getSize(p_count)
          # print("wordcount_class: " , wordcount_class)
          vocab_len = len(vocabulary)
          for word in vocabulary:
            # calculate conditional probability of each word for given class
            p_condProb[word] = (p_count[word] + 1) / (vocab_len + wordcount_class)

        elif class_type == 'negative':
          for word in vocabulary:
            if word in n_dict:
              n_count[word] = n_dict[word]
            else:
              n_count[word] = 0
          # word_probability = probabilityCalc(vocabulary, p_count, p_dict, p_condProb)
          wordcount_class = getSize(n_count)
          # print("wordcount_class: " , wordcount_class)
          vocab_len = len(vocabulary)
          for word in vocabulary:
            n_condProb[word] = (n_count[word] + 1) / (vocab_len + wordcount_class)

        elif class_type == 'truthful':
          for word in vocabulary:
            if word in t_dict:
              t_count[word] = t_dict[word]
            else:
              t_count[word] = 0
          # word_probability = probabilityCalc(vocabulary, p_count, p_dict, p_condProb)
          wordcount_class = getSize(t_count)
          # print("wordcount_class: " , wordcount_class)
          vocab_len = len(vocabulary)
          for word in vocabulary:
            t_condProb[word] = (t_count[word] + 1) / (vocab_len + wordcount_class)

        elif class_type == 'deceptive':
          for word in vocabulary:
            if word in d_dict:
              d_count[word] = d_dict[word]
            else:
              d_count[word] = 0
          wordcount_class = getSize(d_count)
          # print("wordcount_class: " , wordcount_class)
          vocab_len = len(vocabulary)
          for word in vocabulary:
            d_condProb[word] = (d_count[word] + 1) / (vocab_len + wordcount_class)
            # word_probability = probabilityCalc(vocabulary, d_count, d_dict, d_condProb)
    return (priors, p_condProb, n_condProb, t_condProb, d_condProb)

def probabilityCalc(vocabulary, count, class_dict, condProb):
    for word in vocabulary:
        if word in class_dict:
            count[word] = class_dict[word]
        else:
            count[word] = 0
      # get the size of the total count of that class
    wordcount_class = getSize(count)
    # print("word count :: ", wordcount_class)
    vocab_len = len(vocabulary)

      # loop again for all words
    for word in vocabulary:
        # calculate conditional probability of each word for given class
        condProb[word] = (count[word] + 1) / (vocab_len + wordcount_class)
        # print("condprob: " , condProb[word])


def write_model(out_file, vocabulary, priors, p_prob, n_prob, t_prob, d_prob):
  with open(out_file, mode="w") as output:
    for i in priors.keys():
      # Write prior for this class
      output.write(f"#### {i} {priors[i]}\n")
      # Write the word probabilities for this class
      for probability in p_prob[i]:
        output.write(f"{' '.join(probability)}\n")

      for probability in n_prob[i]:
        output.write(f"{' '.join(probability)}\n")

      for probability in t_prob[i]:
        output.write(f"{' '.join(probability)}\n")

      for probability in d_prob[i]:
        output.write(f"{' '.join(probability)}\n")

def getSize(vocab):
    value = 0
    for word in vocab:
        value += vocab[word]
    return value

def storeInDictionary(dataList, dict):
    value = dict
    for token in dataList:
        if token != '':
            if token not in value:
                value[token] = 1
            else:
                value[token] += 1
    return value


pd_path = "/positive_polarity/deceptive_from_MTurk/"
pt_path = "/positive_polarity/truthful_from_TripAdvisor/"
nd_path = "/negative_polarity/deceptive_from_MTurk/"
nt_path = "/negative_polarity/truthful_from_Web/"

# ## Run Naive Bayes
# #### Main Function
def main():
  p_dict = {}
  n_dict = {}
  t_dict = {}
  d_dict = {}  
  count = {} 
  # class_dict = {}
  vocabulary = {}
  condProb = {}
  no_of_documents = 0 
  
  file_path = sys.argv[1]
  DIRECTORY = { "positive": [file_path + pd_path, file_path + pt_path],
            "negative": [file_path + nd_path, file_path + nt_path],
            "deceptive": [file_path + pd_path, file_path + nd_path],
            "truthful": [file_path + pt_path , file_path + nt_path] }

  for key,value in DIRECTORY.items():
    for route in value:
      for root, _, files in os.walk(route, topdown=False):
        for file_name in files:
          # Only parse text files, used to ignore .DS_store file on macOS
          if '.txt' in file_name and file_name != "README.txt":
            file_path = os.path.join(root, file_name)
            get_valid_tokens = do_preprocessing(file_path)
            if key == 'positive':
                p_dict = storeInDictionary(get_valid_tokens, p_dict)
                vocabulary = storeInDictionary(get_valid_tokens, vocabulary)
                class_count["positive"] += 1
            
            elif key == 'negative':
                n_dict = storeInDictionary(get_valid_tokens, n_dict)
                vocabulary = storeInDictionary(get_valid_tokens, vocabulary)
                class_count["negative"] += 1
            
            elif key == 'truthful':
                t_dict = storeInDictionary(get_valid_tokens, t_dict)
                vocabulary = storeInDictionary(get_valid_tokens, vocabulary)
                class_count["truthful"] += 1

            elif key == 'deceptive':
                d_dict = storeInDictionary(get_valid_tokens, d_dict)
                vocabulary = storeInDictionary(get_valid_tokens, vocabulary)
                class_count["deceptive"] += 1
            no_of_documents+=1
  no_of_documents /= 2
   
  # Train - Learn Model
  priors, p_prob, n_prob, t_prob, d_prob = train(p_dict, n_dict, t_dict, d_dict, no_of_documents, vocabulary)
  # print("priors: ", priors)
  # print("p:", p_prob)
  # print("n:", n_prob)
  # print("t:", t_prob)
  # print("d:", d_prob)
  
  # # Write Model
  file_obj = open(MODEL_FILE, 'w')
  file_obj.write(str(priors))
  file_obj.write("\n")
  file_obj.write(str(p_prob))
  file_obj.write("\n")
  file_obj.write(str(n_prob))
  file_obj.write("\n")
  file_obj.write(str(t_prob))
  file_obj.write("\n")
  file_obj.write(str(d_prob))
  file_obj.close()
  # write_model(MODEL_FILE, vocabulary, priors, p_prob, n_prob, t_prob, d_prob)


# #### Run
if __name__ == "__main__":
  main()