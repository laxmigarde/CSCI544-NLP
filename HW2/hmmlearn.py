#!/usr/bin/env python
# coding: utf-8

import os
import sys

# ## Constants
MODEL_FILE = "hmmmodel.txt"
MODEL_OUT_FILE = "hmmoutput.txt"

file_tokens = []
count_of_tags = {}
count_of_transition_tags = {}
count_of_emission_tags = {}
transition_p = {}
emission_p = {}


# calculate the transition - hidden state probability
def transition_probability(file):
    for tags in count_of_transition_tags.keys():
        first_tag = tags.split('->')[0]
        second_tag = tags.split('->')[1]
    
        if count_of_tags[first_tag] > 0:
            # transition conditional probability 
            transition_p[tags] = count_of_transition_tags[tags] / (count_of_tags[first_tag] + len(count_of_tags))

            if first_tag == 'st@rt':
                output = 'st@rt->' + second_tag + '::' + str(transition_p[tags]) + '\n'
            else:
                output = tags + '::' + str(transition_p[tags]) + '\n'

            file.write(output)

# calculate the emission - observational state probability
def emission_probability(file):
    for pair in count_of_emission_tags.keys():
        index = 0
        for index_tag in range(len(pair) - 1, 0, -1):
              if pair[index_tag] == '/':
                  index = index_tag
                  break

        tag = pair[index + 1:]
        word = pair[0:index]
        # print ('Tag ::' + tag + " , Word :: " + word)

        if count_of_tags[tag] > 0:
            # emission probability
            emission_p[pair] = count_of_emission_tags[pair] / count_of_tags[tag]
            # print('emission_p :: '  , emission_p[pair])

            if emission_p[pair] > 1:
                emission_p[pair] = 1
            
            output = word + '|' + tag + '::' + str(emission_p[pair]) + '\n'
            file.write(output)

def find_tag_counts(tags):
    start_tag = 'st@rt'
    current_tag = ''
    # find tag in every (word | tag) present in sentence
    for tag in tags:
    #   print("Tag :: " , tag)
        index = 0
    #   length = len(tag) - 1
        for index_tag in range(len(tag) - 1, 0 , -1):
            if tag[index_tag] == '/':
                index = index_tag
                break
        # find the tag 
        current_tag = tag[index_tag + 1:]
    #   print("Index :: " , index)
    #   print("Current tag :: " , current_tag)

        if current_tag not in count_of_tags:
            count_of_tags[current_tag] = 1
        #   print("Count of tags :: " , count_of_tags[current_tag])
        else:
            count_of_tags[current_tag] += 1
        #   print("Count of tags :: " , count_of_tags[current_tag])

        if start_tag == '' or current_tag == '':
            start_tag = current_tag
            continue


        transition_tag =  start_tag + '->' + current_tag
        # transition tag count - hidden 
        if transition_tag not in count_of_transition_tags:
            count_of_transition_tags[transition_tag] = 1
            #print("Count of t_p :: ", count_of_transition_tags[transition_tag])
        else:
            count_of_transition_tags[transition_tag] += 1
        #print("Count of t_p :: ", count_of_transition_tags[transition_tag])
        

        # emission count - observations           
        if tag not in count_of_emission_tags.keys():
            count_of_emission_tags[tag] = tags.count(tag)
            #   print("Count of e_p :: ", count_of_emission_tags[tag])
        else:
            count_of_emission_tags[tag] += tags.count(tag)
        #   print("Count of e_p :: ", count_of_emission_tags[tag])

        # total count of tags 
        if start_tag in count_of_tags.keys():
            count_of_tags[start_tag] += 1
        else:
            count_of_tags[start_tag] = 1
        
        start_tag = current_tag

# ### write the required details to hmmmodel.txt
def write_model_file():
    file_obj = open(MODEL_FILE, 'w')
    #   file_obj.write('Tags with count' + '\n')
    #   file_obj.write(str(count_of_tags) + '\n')
    file_obj.write('Total number of tags:' + '\n')
    file_obj.write(str(len(count_of_tags)) + '\n')

    #   easy to fetch in decode.py instead of above approach
    file_obj.write('Tags with count:' + '\n')
    for pair in count_of_tags.keys():
        file_obj.write(pair + '=' + str(count_of_tags[pair]) + '\n')

    #   file_obj.write('============================ \n')
    file_obj.write('Transition Probabilities:' + '\n')
    transition_probability(file_obj)
    #   file_obj.write('============================ \n')
    file_obj.write('Emission Probabilities:' + '\n')
    emission_probability(file_obj)
    file_obj.close()


# #### Main Function
def main():
  
  file_path = sys.argv[1]

#   read the tagged train file
  file_text = open(file_path, 'r')

#   store all lines in array as tokens per sentence
  for input in file_text:
      file_tokens.append(input.split())  

  file_text.close()

#   parse the (word/tag) to separate out 'word' and 'tags' from the tokens in array
#   for per sentence tokens generated, find tags
  for tags in file_tokens:
    # print("Tags ::::: ", tags)
    find_tag_counts(tags)
    
#   write to the hmmmodel file
  write_model_file()


# #### Run
if __name__ == "__main__":
  main()
