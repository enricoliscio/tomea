import glob
import os
import sys
import argparse
import pandas as pd

def group_and_dump(input_folder, output_folder):
	"""read the lemmatized csv files in subdirectories and join the importance
	based on lemmata
	"""
	# Create output folder, if it does not exist.
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	# Get all directories in the input folder, each directory corresponding to a domain.
	domains = os.listdir(input_folder)

	for domain in domains:
		# Create domain folder in output folder.
		output_domain_folder = os.path.join(output_folder, domain)
		if not os.path.exists(output_domain_folder):
			os.makedirs(output_domain_folder)

		# Get all csv files in the source domain folder.
		csv_files = glob.glob(os.path.join(input_folder, domain, '*.csv'))

		for filename in csv_files:
			df = pd.read_csv(filename,sep=',',header=[0])

			#lemma is empty for things that were not lemmatize, create new column
			#that contains lemma if present,  token otherwise
			df['grouping'] = df['lemma'].fillna(df['features'])

			#aggregate multiple words, importance add together importance
			aggregation_functions = {'idx': 'count', 'features': ' '.join , 'importance': 'sum'}
			df_new = df.groupby(df['grouping']).aggregate(aggregation_functions).sort_values('importance',ascending=False)

			outfilename = os.path.join(output_domain_folder, os.path.basename(filename))

			df_new.to_csv(outfilename)


def parse_args(args):
	""" Parse the command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Group lemmatized SHAP lexicons.')
	parser.add_argument('--input-folder',  help='Folder containing the domain folders with lemmatized lexicons.', type=str)
	parser.add_argument('--output-folder', help='Destination folder.',                                     type=str)

	return parser.parse_args(args)


def main(args=None):
	# Parse command line arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	group_and_dump(args.input_folder, args.output_folder)


if __name__ == '__main__':
	main()