#!/usr/bin/env python
# coding: utf-8

# viterbi decoding algorithm to find most likely sequence
# learning Prob. matrices - learn from corpus , EM - FB algorithm 

import os
import sys

# ### Constants
MODEL_FILE = "hmmmodel.txt"
MODEL_OUT_FILE = "hmmoutput.txt"

index = 0
total_tags = 0
final_add_tags = ''
words_from_model = {}
tag_names = []
tag_names_from_model = {}
transition_p = {}
emission_p = {}
count_of_tags = {}

# ### reading the hmmmodel.txt file 
read_model_file = open(MODEL_FILE, "r")
for line in read_model_file:
    if line == 'Total number of tags:\n':
        index = 1
        continue
    if line == 'Tags with count:\n':
        index = 2
        continue
    if line == 'Transition Probabilities:\n':
        index = 3
        continue
    if line == 'Emission Probabilities:\n':
        index = 4
        continue

    # read from model file and store
    if index == 1:
        # total number of tags  
        total_tags = int(line.strip())
    if index == 2:
        # count per tag
        count_of_tags[line.split('=')[0]] = int(line.split('=')[1].strip('\n'))
        tag_names.append(line.split('=')[0])
        # print('count of tags :: ' , line.split('=')[0] , 'is :' , count_of_tags[line.split('=')[0]])
        # print('tag names: ' , tag_names)
    if index == 3:
        transition_data =  line.split('::')
        states = transition_data[0]
        transition_p[states] = float(transition_data[1].strip('\n'))
        # print('transition_p : ' , transition_p[states])
    if index == 4:
        emission_data = line.split('::')
        word_tag_pair = emission_data[0]
        word = word_tag_pair.split('|')[0]
        tag = word_tag_pair.split('|')[1]
        emission_p[word_tag_pair] = float(emission_data[1].strip('\n'))
        words_from_model[word] = 1
        # print('emission_p : ' , emission_p[word_tag_pair])
read_model_file.close()


# ### viterbi decoding - to decode or find the sequence of most likely states 
# i.e. find state with highest probability and backtrack from that state
def viterbi_decoding(sentence):
    sequence = sentence.split(' ')
    sequenceLen = len(sequence)
    # print(words_from_model)
    # print(sequence)
    # 'probability' and 'backpointer' matrices as per Viterbi algorithm
    probability_matrix = [[0 for x in range(sequenceLen)] for y in range(total_tags + 1)]
    backpointer_matrix = [[0 for x in range(sequenceLen)] for y in range(total_tags + 1)]
    # print('total tags ::' , total_tags)
    # print('tag names ::' , tag_names_from_model)

    for index in tag_names_from_model.keys():
        
        transition = 'st@rt' + tag_names_from_model[index]
        if transition not in transition_p.keys():
            prob_transition = float(1 / (count_of_tags['st@rt'] + total_tags))
        else:
            prob_transition = float(transition_p[transition])

        # print('tagnames for model keys', tag_names_from_model.keys())
        emission = sequence[0] + '|' + tag_names_from_model[index]
        # print('emission:' , emission)
        # if emission not in emission_p.keys():
        #     prob_emission = 0.0
        if sequence[0] not in words_from_model.keys():
            prob_emission = 1.0
        elif emission not in emission_p.keys():
            prob_emission = 0.0
        else:
            prob_emission = float(emission_p[emission])

        # probability(q,1) = a(q0,q) * b(q,o1), backpointer(q,1) = q0
        probability_matrix[index][0] = prob_emission * prob_transition
        backpointer_matrix[index][0] = 0
        # print('probability_matrix:' , probability_matrix[index][0])
        # print('backpointer_matrix:' , backpointer_matrix[index][0])

    iterate_all_sequence(sequence, probability_matrix, prob_emission, prob_transition, backpointer_matrix)
    final_add_tags = most_likely_prob_backtrack(probability_matrix, sequence, sequenceLen, backpointer_matrix)
    return final_add_tags

# ### recursion step t: 2 to T part
def iterate_all_sequence(sequence, probability_matrix, prob_emission, prob_transition, backpointer_matrix):
    probability = 0
    sequenceLen = len(sequence)

    for index1 in range(1, sequenceLen):
        # print('index1:', index1)
        for last_tag in tag_names_from_model.keys():
            for start_tag in tag_names_from_model.keys():
                emission = sequence[index1] + '|' + tag_names_from_model[last_tag]
                # print('emission: ' , emission)

                if sequence[index1] not in words_from_model.keys():
                    prob_emission = 1.0
                elif emission not in emission_p.keys():
                    prob_emission = 0.0
                    continue
                else:
                    prob_emission = float(emission_p[emission])
                    if 0 == prob_emission:
                        continue

                transition = tag_names_from_model[start_tag] + '->' + tag_names_from_model[last_tag]
                # print('transition:' , transition)
                if transition not in transition_p.keys():
                    count = count_of_tags[tag_names_from_model[start_tag]]
                    prob_transition = float(1 / int((count) + total_tags))
                else:
                    prob_transition = float(transition_p[transition])
                    if 0 == prob_transition:
                        continue

                # probability(q,t) = max probability(q0,t −1) * a(q0,q) * b(q,ot)
                # backpointer(q,t) = argmax probability(q0,t −1) ∗ a(q0,q)
                probability = float(probability_matrix[start_tag][index1 - 1] * prob_emission * prob_transition)
                if 0 == probability:
                    continue
                elif probability > float(probability_matrix[last_tag][index1]):
                    probability_matrix[last_tag][index1] = probability
                    backpointer_matrix[last_tag][index1] = start_tag
                else:
                    continue

# find the most likely sequence of states i.e. decode the seq. of states here as per Viterbi algo.  
def most_likely_prob_backtrack(probability_matrix, sequence, sequenceLen, backpointer_matrix):
    global final_add_tags
    # sequenceLen = len(sequence)
    most_likely_index = 0
    # loop through each tag
    for index2 in tag_names_from_model.keys():
        if probability_matrix[index2][sequenceLen - 1] > probability_matrix[most_likely_index][sequenceLen - 1]:
            most_likely_index = index2

    add_tags = sequence[sequenceLen - 1] + '/' + tag_names_from_model[most_likely_index] + ' '
    # print(add_tags)

    for index3 in range(sequenceLen - 1, 0 , -1):
        most_likely_index = backpointer_matrix[most_likely_index][index3]
        add_tags = sequence[index3 - 1] + '/' + tag_names_from_model[most_likely_index] + ' ' + add_tags

    final_add_tags += add_tags + '\n'
    return final_add_tags

# store the list to dictionary 
def get_tag_dict():
    i = 0
    for tag in tag_names:
        tag_names_from_model[i] = tag
        i += 1        

# ### Main Function
def main():
    # global final_add_tags
    # read the input dev file with raw corpus
    input_file = sys.argv[1]
    raw_file_obj = open(input_file, 'r')

    get_tag_dict()
    # print('tag_names model : ' , tag_names_from_model)
    for sentence in raw_file_obj:
        # sequence = sentence.split(' ')
        final_add_tags = viterbi_decoding(sentence.strip())
    
    output_file_obj = open(MODEL_OUT_FILE, 'w')
    output_file_obj.write(final_add_tags)
    raw_file_obj.close()
    output_file_obj.close()


# ### Run
if __name__ == "__main__":
  main()
