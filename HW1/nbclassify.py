#!/usr/bin/env python
# coding: utf-8

import glob
import math
import os
import sys

import regex

MODEL_FILE = "nbmodel.txt"
MODEL_OUT_FILE = "nboutput.txt"
CLASSES = ["positive", "negative", "truthful", "deceptive"]
score = {}

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

# ## call preprocessing steps
def do_preprocessing(file_name):
    file_object = open(file_name, "r")
    data = file_object.read()
    data_tokens = to_lowercase(data)
    data_tokens_without_punctations = remove_punctuations(data_tokens)
    data_without_digits = remove_digits(data_tokens_without_punctations)
    data_without_stopwords = filter_out_stopwords(data_without_digits)
    return data_without_stopwords

def storeInDictionary(dataList, dict):
    value = dict
    for token in dataList:
        if token != '':
            if token in value:
                value[token] += 1
            else:
                value[token] = 1
    return value

def noTokenInDict(condProb_dict, token):
    if token in condProb_dict:
        # print("val :", condProb_dict[token])
        return condProb_dict[token]
    else:
        return 0

def compare_scores(class_score):
    if class_score["positive"] > class_score["negative"]:
        label_a = "positive"
    else:
        label_a = "negative"

    if class_score["deceptive"] > class_score["truthful"]:
        label_b = "deceptive"
    else:
        label_b = "truthful"
    return label_b, label_a

def classifier(path_n):
    
    # preprocess the files data and add tokens to a dict
    tokens = do_preprocessing(path_n)
    tokens_dict = {}
    tokens_dict = storeInDictionary(tokens_dict, tokens)

    for class_type in CLASSES:
        score[class_type] = math.log(priors[class_type], 10)

        for token in tokens_dict:
            if class_type == "positive":
                token_present = noTokenInDict(p_condProb, token)
                if token_present != 0:
                    score[class_type] = score[class_type] + math.log((p_condProb[token]), 10)
                else:
                    continue
                    
            elif class_type == "negative":
                token_present = noTokenInDict(n_condProb, token)
                if token_present != 0:
                    score[class_type] = score[class_type] + math.log((n_condProb[token]), 10)
                else:
                    continue

            elif class_type == "truthful":
                token_present = noTokenInDict(t_condProb, token)
                if token_present != 0:
                    score[class_type] = score[class_type] + math.log((t_condProb[token]), 10)
                else:
                    continue

            elif class_type == "deceptive":
                token_present = noTokenInDict(d_condProb, token)
                if token_present != 0:
                    score[class_type] = score[class_type] + math.log((d_condProb[token]), 10)
                else:
                    continue            
    return score

# reading the nbmodel.txt file
read_model_file = open(MODEL_FILE, "r")
priors = eval(read_model_file.readline())
p_condProb = eval(read_model_file.readline())
n_condProb = eval(read_model_file.readline())
t_condProb = eval(read_model_file.readline())
d_condProb = eval(read_model_file.readline())
read_model_file.close()

pd_path = "/positive_polarity/deceptive_from_MTurk/"
pt_path = "/positive_polarity/truthful_from_TripAdvisor/"
nd_path = "/negative_polarity/deceptive_from_MTurk/"
nt_path = "/negative_polarity/truthful_from_Web/"

def main():

    file_path = sys.argv[1]
    DIRECTORY = { "positive": [file_path + pd_path, file_path + pt_path],
            "negative": [file_path + nd_path, file_path + nt_path],
            "deceptive": [file_path + pd_path, file_path + nd_path],
            "truthful": [file_path + pt_path , file_path + nt_path] }

    # output file nboutput.txt
    out_file = open(MODEL_OUT_FILE, "w")
    file_path = glob.glob(os.path.join(sys.argv[-1], '*/*/*/*.txt'))
    for path_n in file_path:
        # get the score
        class_score = classifier(path_n)
        # print("class_score :: " , class_score)
        # initialize output
        result = []
        # compare scores
        label_b, label_a = compare_scores(class_score)
        result = label_b + " " + label_a + " " + path_n + "\n"
        out_file.write(result)
    out_file.close()



# #### Run
if __name__ == "__main__":
  main()