import pandas as pd
import numpy as np

def pullData():
    # Read tsv data file and store in dataframe
    data = pd.read_csv('../data/songDb.tsv', sep='\t', encoding='latin-1')
    return data

def cleanData(data):
    # Index song by name
    data = data.set_index(['Name'])
    # Only use features that are numerical
    clean_data = data[['Danceability', 'Energy', 'Key', 'Loudness']]
                       #'Mode', 'Speechness', 'Acousticness',
                       #'Instrumentalness', 'Liveness', 'Valence']]
    return clean_data

def main():
	data = cleanData(pullData())
	return data

if __name__ == "__main__":
	main()
