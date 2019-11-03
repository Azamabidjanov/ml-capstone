# Machine-Learning Capstone Project.

## Authors: Natalie Kalin, Armando Ocampo, Jay Harrison, Azam Abidjanov

## Dataset
All song data originates from Spotify, however a precompiled dataset of ~80,000 songs was retrieved from https://www.kaggle.com/grasslover/spotify-music-genre-list. Credit to Adri Molina 

## Development Environment - `pipenv`
We are using pipenv to keep our development/dependencies consistent.
Pipenv uses Pipfile/Pipfile.lock to ensure we are using the same version
of packages for our projects. This avoids "I swear it just worked on my machine!" type situations.
Whenever you are introducing a new package, please add it to the pipfile (instructions below).

## Pipenv installation
+ Clone the repo: `git clone`
+ In the repo directory, set your environment and dependencies: `pipenv install --dev`
+ Activate your environment: `pipenv shell`
+ Run python scripts from the activated shell: `python pull-data.py`

## Pipenv: Adding new packages
+ To add a new package (scipy for example) into the pipfile: `pipenv install scipy`

## Good Attributes for Songs

Good: Acousticness, Danceability, Energy, Instrumentalness, Loudness, valence, tempo

Bad: Liveness (detecting live music), Speechiness (used for detecting podcasts), Key (key song is played in), Mode (binary value...not useful)

https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/

## Spectral Clustering

-Looked at three different characteristics with 4 different genres using Spectral Clustering

-First, evaluated Danceability, Acousticness, and Loudness (Distinction not clear)

-Second, evaluated Danceability, Acousticness, and Valence (much cleaner distinction)

-Third, evaluated Danceability, Energy, and Valence (much cleaner distinction)

-fourth, evaluated Tempo, Energy, and Valence (not great separation)

---------------------------------------------------------------------------------------------------------------------------------------

# Recycling Project

## Dataset
https://github.com/garythung/trashnet

## Data Parsing
https://towardsdatascience.com/how-to-build-an-image-classifier-for-waste-sorting-6d11d3c9c478

https://nbviewer.jupyter.org/github/collindching/Waste-Sorter/blob/master/Waste%20sorter.ipynb
