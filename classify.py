#!/usr/bin/env python3

import spacy
import nltk

#nlp = spacy.load("en_core_web_sm")

from nltk.corpus import stopwords

stops = stopwords.words('english')



def get_words(filename, genre):
    '''Takes an input file which is a .txt file where each line is a new song.
       output a list of each song in a tuple (song lyrics, genre) : (list, str)
    '''
    lines = []
    with open(filename) as fp:
        for line in fp:
            #real_lyrics = []
            line = line.rstrip('\n')
            lyrics = line.split(' ')
            #for word in lyrics:
            #    if word not in stops:
            #        real_lyrics.append(word)
            lines.append((lyrics, genre))
    return lines

country_words = get_words('country_lyrics.txt', 'country')
country_test = country_words[:1000]
country_train = country_words[1000:]

rock_words = get_words('rock_lyrics.txt', 'rock')
rock_test = rock_words[:1000]
rock_train = rock_words[1000:]

pop_words = get_words('pop_lyrics.txt', 'pop')
pop_test = pop_words[:1000]
pop_train = pop_words[1000:]

dance_words = get_words('dance_lyrics.txt', 'dance')
dance_test = dance_words[:1000]
dance_train = dance_words[1000:]

rap_words = get_words('rap_lyrics.txt', 'rap')
rap_test = rap_words[:1000]
rap_train = rap_words[1000:]

print("length of rap training set: ", len(rap_train), " songs") #16114 songs
print("length of rock training set: ", len(rock_train), " songs") #93992 songs
print("length of dance training set: ", len(dance_train), " songs") #10895 songs
print("length of country training set: ", len(country_train), " songs") #9630
print("length of pop training set: ", len(pop_train), " songs") #52858 songs

print()
print("==============================================================")
print()



all_train = country_train + rock_train + pop_train + dance_train + rap_train
all_test = country_test + rock_test + pop_test + dance_test + rap_test

'''
At this point, we have a list of all the words contained in any of the songs
from that genre
This is all the training data... will eventually do this for the test data
'''

def feats(song):
    features = {}
    for word in song:
        if word not in stops:
            features["contains-" + word.lower()] = 1
    return features
# If I include stop words, the prediction does worse

train_feats = [(feats(song), genre) for song, genre in all_train]
#for i in train_feats[::1000]:
#    print(i[1])
# All genres are included in train_feats (training data)

rock_feats =  [(feats(song), genre) for song, genre in rock_test]
country_feats =  [(feats(song), genre) for song, genre in country_test]
dance_feats =  [(feats(song), genre) for song, genre in dance_test]
pop_feats =  [(feats(song), genre) for song, genre in pop_test]
rap_feats =  [(feats(song), genre) for song, genre in rap_test]

#test_feats = [(feats(song), genre) for song, genre in all_test]
test_feats = rock_feats + country_feats + dance_feats + pop_feats + rap_feats


whichgenre = nltk.NaiveBayesClassifier.train(train_feats)
#print("Successfully Trained!")

rock_accuracy = nltk.classify.accuracy(whichgenre, rock_feats)
print("Accuracy score for Rock Songs:", rock_accuracy)

country_accuracy = nltk.classify.accuracy(whichgenre, country_feats)
print("Accuracy score for Country Songs:", country_accuracy)

dance_accuracy = nltk.classify.accuracy(whichgenre, dance_feats)
print("Accuracy score for Dance Songs:", dance_accuracy)

rap_accuracy = nltk.classify.accuracy(whichgenre, rap_feats)
print("Accuracy score for Rap Songs:", rap_accuracy)                     

pop_accuracy = nltk.classify.accuracy(whichgenre, pop_feats)
print("Accuracy score for Pop Songs:", pop_accuracy)


accuracy = nltk.classify.accuracy(whichgenre, test_feats)
print("Accuracy score for All Lyrics:", accuracy) #23.7%


'''

Accuracy score for Rock Songs: 0.001
Accuracy score for Country Songs: 0.005
Accuracy score for Dance Songs: 0.0
Accuracy score for Rap Songs: 1.0
Accuracy score for Pop Songs: 0.0
Accuracy score for All Lyrics: 0.2012

'''



matrix = [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]
    

for song, genre in all_test:
    guess = whichgenre.classify(feats(song))
    if genre == "country" and guess == "country":
        matrix[0][0] += 1
    elif genre == "country" and guess == "dance":
        matrix[0][1] += 1
    elif genre == "country" and guess == "pop":
        matrix[0][2] += 1
    elif genre == "country" and guess == "rap":
        matrix[0][3] += 1
    elif genre == "country" and guess == "rock":
        matrix[0][4] += 1
    elif genre == "dance" and guess == "country":
        matrix[1][0] += 1
    elif genre == "dance" and guess == "dance":
        matrix[1][1] += 1
    elif genre == "dance" and guess == "pop":
        matrix[1][2] += 1
    elif genre == "dance" and guess == "rap":
        matrix[1][3] += 1
    elif genre == "dance" and guess == "rock":
        matrix[1][4] += 1
    elif genre == "pop" and guess == "country":
        matrix[2][0] += 1
    elif genre == "pop" and guess == "dance":
        matrix[2][1] += 1
    elif genre == "pop" and guess == "pop":
        matrix[2][2] += 1
    elif genre == "pop" and guess == "rap":
        matrix[2][3] += 1
    elif genre == "pop" and guess == "rock":
        matrix[2][4] += 1
    elif genre == "rap" and guess == "country":
        matrix[3][0] += 1
    elif genre == "rap" and guess == "dance":
        matrix[3][1] += 1
    elif genre == "rap" and guess == "pop":
        matrix[3][2] += 1
    elif genre == "rap" and guess == "rap":
        matrix[3][3] += 1
    elif genre == "rap" and guess == "rock":
        matrix[3][4] += 1
    elif genre == "rock" and guess == "country":
        matrix[4][0] += 1
    elif genre == "rock" and guess == "dance":
        matrix[4][1] += 1
    elif genre == "rock" and guess == "pop":
        matrix[4][2] += 1
    elif genre == "rock" and guess == "rap":
        matrix[4][3] += 1
    elif genre == "rock" and guess == "rock":
        matrix[4][4] += 1

print()
print()
print("CONFUSION MATRIX:")
print("Horizontal = predicted, Vertical = Actual")
print()
print("\t cntry\t dance\t pop\t rap\t rock")
print(
    "country\t",
    matrix[0][0],
    "\t",
    matrix[0][1],
    "\t",
    matrix[0][2],
    "\t",
    matrix[0][3],
    "\t",
    matrix[0][4],
)

print(
    "dance\t",
    matrix[1][0],
    "\t",
    matrix[1][1],
    "\t",
    matrix[1][2],
    "\t",
    matrix[1][3],
    "\t",
    matrix[1][4],
)

print(
    "pop\t",
    matrix[2][0],
    "\t",
    matrix[2][1],
    "\t",
    matrix[2][2],
    "\t",
    matrix[2][3],
    "\t",
    matrix[2][4],
)

print(
    "rap\t",
    matrix[3][0],
    "\t",
    matrix[3][1],
    "\t",
    matrix[3][2],
    "\t",
    matrix[3][3],
    "\t",
    matrix[3][4],
)

print(
    "rock\t",
    matrix[4][0],
    "\t",
    matrix[4][1],
    "\t",
    matrix[4][2],
    "\t",
    matrix[4][3],
    "\t",
    matrix[4][4],
)

#whichgenre.show_most_informative_features(100)


"""

CURRENT CONFUSION MATRIX:
Horizontal = predicted, Vertical = Actual

	 cntry	 dance	 pop	 rap	 rock
country	 5 	 0 	 0 	 994 	 1
dance	 0 	 0 	 0 	 1000 	 0
pop	 3 	 0 	 0 	 997 	 0
rap	 0 	 0 	 0 	 1000 	 0
rock	 12 	 0 	 2 	 985 	 1

"""
