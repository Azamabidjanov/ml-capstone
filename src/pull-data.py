import pandas as pd
import numpy as np

def PullData():
	data = pd.read_csv('../data/songDb.tsv', sep='\t', encoding='latin-1')
	return data

def main():
	data = PullData()
	print( data.head() )

if __name__ == "__main__":
	main()
