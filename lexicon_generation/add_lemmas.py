from nltk.corpus import wordnet31 as wn31
import glob
import os
import sys
import argparse


def lemmatize_and_dump(input_folder, output_folder):
	"""read the csv files in subdirectories and create new ones with
	lemmata estimates for the  words - multiple lemmas are not joined (yet)
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
			with open(filename, "r") as csv_input_file:
				outfilename = os.path.join(output_domain_folder, os.path.basename(filename))
				with open(outfilename, "w") as csv_output_file:
					header = csv_input_file.readline().split(",")
					header[0] = 'idx'
					header.insert(2,"lemma")
					csv_output_file.write(",".join(header))
					for line in csv_input_file:
						fields = line.split(",")
						inflected = fields[1]
						lemma = ""
						#check if form exists in wordnet, then add it as lemma
						if wn31.morphy(inflected):
							lemma = wn31.morphy(inflected)
						fields.insert(2, lemma)
						csv_output_file.write(",".join(fields))


def parse_args(args):
	""" Parse the command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Lemmatize a SHAP lexicon.')
	parser.add_argument('--input-folder',  help='Folder containing the domain folders with raw lexicons.', type=str)
	parser.add_argument('--output-folder', help='Destination folder.',                                     type=str)

	return parser.parse_args(args)


def main(args=None):
	# Parse command line arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	lemmatize_and_dump(args.input_folder, args.output_folder)


if __name__ == '__main__':
	main()
