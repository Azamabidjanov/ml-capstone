import pandas as pd
import numpy as np

def PullData():
	data = pd.DataFrame.from_csv('../data/songDb.tsv', sep='\t')
	return data

def main():
	data = PullData()
	print(data.head())

if __name__ == "__main__":
	main()
