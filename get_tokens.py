#!/usr/bin/env python3

# Train a classifier (?)

import csv
import spacy
from collections import Counter

data = []
with open("my_data.csv", 'r') as dat:
    csvreader = csv.reader(dat)
    header = next(csvreader)
    for row in csvreader:
        data.append(row)

cnt = Counter()
for song in data:
    cnt[song[0]] += 1

#print(cnt.most_common())
# Rock:    94992 songs
# Pop:     53858 songs
# Rap:     17114 songs
# Dance:   11895 songs
# Country: 10630 songs

def get_only_genre(data, genre):
    songs = []
    for song in data:
        if song[0].lower() == genre.lower():
            songs.append(song)
    return songs



rap_data = get_only_genre(data, 'rap')
rock_data = get_only_genre(data, 'rock')
pop_data = get_only_genre(data, 'pop')
country_data = get_only_genre(data, 'country')
dance_data = get_only_genre(data, 'dance')



# Tokenize
# Split and train on Words
# Or sentences

"""
    This next part of code will closely follow the code from the class lecture
    on classification from 03-31-2022.
"""


nlp = spacy.load("en_core_web_sm")
# this line should make it not split on apostrophes
nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}

# A function to get all the songs for a specific genre
# the 'sentences' from the in class example will be the songs
# where the 'authors' will be the genre


# seperate songs by genre before this lol
def get_songs(data, genre: str) -> list:
    songs = []
    for song in data:
        lyrics = song[3]
        lyrics = lyrics.replace("\n", ". ").replace("--", " -- ")
        #lyrics = lyrics.replace("'", "")
        #lyrics is a string of the lyrics
        
        tokens = [token.text for token in nlp(lyrics)
                                       if not(token.text.isspace()
                                              or
                                              token.is_punct)]
        song_string = " ".join(tokens)
        songs.append(song_string)
    filename = genre+"_lyrics.txt"
    with open(filename, "w") as outfile:
        outfile.write("\n".join(songs))
    # tokens are written to a .txt output file where each line is a new token
    # not sure if this is the best way to do it as not seperated by song

'''this will eventually be used to create the tokens for the test data'''


'''
songs.extend(tokens)
write file with name 'genre' that contains songs
'''

'''
    going forward this function will instead just create a list of every token
    in all songs and then write that list to a .txt file which will then be
    used to train the model rather than running and keeping the results
    contained in this file.
'''
print('Starting Rap songs!')
rap_songs = get_songs(rap_data, 'rap')

print('\nDone with Rap songs...Moving onto Rock!\n')
rock_songs = get_songs(rock_data, 'rock')

print('\nDone with Rock songs...Moving onto Country!\n')
country_songs = get_songs(country_data, 'country')

print('\nDone with Country songs...Moving onto Dance!\n')
dance_songs = get_songs(dance_data, 'dance')

print('\nDone with Dance songs...Moving onto Pop!\n')
pop_songs = get_songs(pop_data, 'pop')

#print(len(rock_songs))

#print("\nRap Songs:")
#print(rap_songs)

#print("\nRock Songs:")
#print(rock_songs)

#print("\nCountry Songs:")
#print(country_songs)

#print("\nDance Songs:")
#print(dance_songs)

#print("\nPop Songs:")
#print(pop_songs)

#print(pop_data[4][3])
