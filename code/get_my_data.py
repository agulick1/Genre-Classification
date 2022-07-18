#!/usr/bin/env python3


# This program reads the data from Kaggle and uses it to write a new csv file
# With only the data that I want to use for my project
# Songs in english, by artists listed as one of the 5 genres


import csv
import spacy

artist_data = []
with open("artists-data.csv", 'r') as art:
    csvreader = csv.reader(art)
    art_header = next(csvreader)
    for row in csvreader:
        artist_data.append(row)
#print(header)
#print(rows)

lyrics = []
with open("lyrics-data.csv", 'r') as lyr:
    csvreader = csv.reader(lyr)
    lyr_head = next(csvreader)
    for row in csvreader:
        if row[4] == 'en':
            lyrics.append(row)
#print(len(lyrics))
#191814 total english songs


# sort the artists by genre
# for this project, I will only be using Rock, Rap, Pop, Country, Dance
# TO BE UPDATED AS NECESSARY
genres = ['Rock', 'Rap', 'Pop', 'Country', 'Dance']
rock_artists = []
rap_artists = []
pop_artists = []
country_artists = []
dance_artists = []
all_artists = []
for line in artist_data:
    if 'Rock' in line[1]:
        rock_artists.append(line[4])
        all_artists.append(line[4])
    if 'Rap' in line[1]:
        rap_artists.append(line[4])
        all_artists.append(line[4])
    if 'Pop' in line[1]:
        pop_artists.append(line[4])
        all_artists.append(line[4])
    if 'Country' in line[1]:
        country_artists.append(line[4])
        all_artists.append(line[4])
    if 'Dance' in line[1]:
        dance_artists.append(line[4])
        all_artists.append(line[4])

# Remove Duplicates
all_artists = list(set(all_artists))

#print("Rock artists: ", rock_artists[:10])
#print("Rap artists: ", rap_artists)
#print("Pop artists: ", pop_artists)
#print("Country artists: ", country_artists)
#print("Dance artists: ", dance_artists)

lyr = []
for song in lyrics:
    if  song[0] in all_artists:
        lyr.append(song)

# At this point, lyr is all english songs from one of the artists
# in the 5 genres. total of 141186 songs across 5 genres from 2160 artists
print("Total Number of genres:\t", len(genres))
print("Total Number of artists:", len(all_artists)) 
print("Total Number of songs:\t", len(lyr))

#print(all_artists[1:3])
#print()
#print(lyr[1:3])

"""
At this point, we have the following lists
genres:      ['Rock', 'Rap', 'Pop', 'Country', 'Dance']
all_artists: ['/artist-name/', '/artist-name/', ...]
lyr:         [
             ['/artist-name/', 'song-title', 'html-link', 'lyrics', 'en'] 
             ['/artist-name/', 'song-title', 'html-link', 'lyrics', 'en']
             ]

the data that i want in my csv is:
'genre    lyrics]
'genre'  'the list of all the lyrics'

"""

header = ['Genre', 'Artist',  'Song Title', 'Lyrics']
data = []
for song in lyr:
    if song[0] in rock_artists:
        data.append(['Rock', song[0], song[1], song[3]])
    if song[0] in rap_artists:
        data.append(['Rap', song[0], song[1], song[3]])
    if song[0] in pop_artists:
        data.append(['Pop', song[0], song[1], song[3]])
    if song[0] in country_artists:
        data.append(['Country', song[0], song[1], song[3]])
    if song[0] in dance_artists:
        data.append(['Dance', song[0], song[1], song[3]])

filename = 'my_data.csv'
with open(filename, 'w', newline="") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)
    csvwriter.writerows(data)


"""
At this point:
my_data.csv is:
Genre    Artist    Song Title    Lyrics

for only the english songs of the 5 genres I selected

"""
