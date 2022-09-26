import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from itertools import product
import os
import sys
import argparse


lexicon_paths = {
	'ALM'      : 'ALM/shap_scores_ALM_{}.csv',
	'Baltimore': 'Baltimore/shap_scores_Baltimore_{}.csv',
	'BLM'      : 'BLM/shap_scores_BLM_{}.csv',
	'Davidson' : 'Davidson/shap_scores_Davidson_{}.csv',
	'Election' : 'Election/shap_scores_Election_{}.csv',
	'MeToo'    : 'MeToo/shap_scores_MeToo_{}.csv',
	'Sandy'    : 'Sandy/shap_scores_Sandy_{}.csv',
}

domains = lexicon_paths.keys()

morals = ['care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal',
               'authority', 'subversion', 'purity', 'degradation']


def read_lexicon(path):
	df = pd.read_csv(path, index_col=0)
	return df


def load_lexicons(input_folder):
	lexicons = defaultdict(dict)
	for lexicon_name, lexicon_path in lexicon_paths.items():
		for moral_name in morals:
			file_path = os.path.join(input_folder, lexicon_path.format(moral_name))
			lexicons[lexicon_name][moral_name] = read_lexicon(file_path)

	return lexicons


def normalize_lexicons(lexicons):
	for domain in domains:
		for moral in morals:
			lexicons[domain][moral]['importance'] = zscore(lexicons[domain][moral]['importance'])

	return lexicons


def distance_matrix(domain_i, domain_j, lexs_i, lexs_j, normalization='length'):
	matrix = []
	for i, moral_name_i in enumerate(morals):
		lex_i = lexs_i[domain_i][moral_name_i].copy()

		matrix_i = []
		for j, moral_name_j in enumerate(morals):
			lex_j = lexs_j[domain_j][moral_name_j].copy()

			common_vocab = pd.merge(
				left=lexs_i[domain_i][moral_name_i],
				right=lexs_j[domain_j][moral_name_j],
				on='features', suffixes=['_i', '_j'])
			common_vocab.sort_values('features', inplace=True)

			dist_i = common_vocab.importance_i.values
			dist_j = common_vocab.importance_j.values

			distance = round(euclidean(dist_i, dist_j), 2)

			if normalization == 'length':
				distance = distance / len(common_vocab)
			elif normalization == 'sqrt':
				distance = distance / np.sqrt(len(common_vocab))
			else:
				raise ValueError("Please indicate a valid normalization.")

			matrix_i.append(distance)
		matrix.append(matrix_i)

	return pd.DataFrame(matrix, columns=morals, index=morals)


def moral_lexicons_comparison(lexicons):
	matrixes = dict()

	for combination in product(lexicons.keys(), repeat=2):
		d = distance_matrix(combination[0], combination[1], lexicons, lexicons)
		matrixes[combination] = d

	return matrixes


def dump_all_m_scores(output_folder, matrixes):
	output_folder = os.path.join(output_folder, 'all_m_scores')
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	for combination, matrix in matrixes.items():
		output_file = os.path.join(output_folder, f'{combination[0]}_{combination[1]}.csv')
		matrix.to_csv(output_file)


def dump_m_scores(output_folder, matrixes):
	output_folder = os.path.join(output_folder, 'm_scores')
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	m_scores = dict()
	for moral in morals:
		m_scores[moral] = pd.DataFrame(index=domains, columns=domains)

	for combination, matrix in matrixes.items():
		for moral in morals:
			m_scores[moral].loc[combination[0], combination[1]] = matrix.loc[moral, moral]

	for moral, matrix in m_scores.items():
		output_file = os.path.join(output_folder, f'{moral}.csv')
		matrix.to_csv(output_file)


def context_lexicons_comparison(lexicons, matrixes):
	context_scores = pd.DataFrame(columns=lexicons.keys(), index=lexicons.keys())
	for domain_i in lexicons.keys():
		for domain_j in lexicons.keys():
			diag = np.diag(matrixes[(domain_i, domain_j)])
			norm_diag = np.linalg.norm(diag)
			context_scores.loc[domain_i, domain_j] = norm_diag

	return context_scores


def parse_args(args):
	""" Parse the command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Group lemmatized SHAP lexicons.')
	parser.add_argument('--input-folder',  help='Folder containing the domain folders with lemmatized lexicons.', type=str)
	parser.add_argument('--output-folder', help='Destination folder.', type=str)
	parser.add_argument('--normalization', help='Type of normalization on the words distance.', default='length', type=str)

	return parser.parse_args(args)


def main(args=None):
	# Parse command line arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# Create output folder.
	if not os.path.exists(args.output_folder):
		os.makedirs(args.output_folder)

	# Load lexicons.
	lexicons = load_lexicons(args.input_folder)

	# z-score normalization.
	lexicons = normalize_lexicons(lexicons)

	# Get moral lexicons comparisons.
	matrixes = moral_lexicons_comparison(lexicons)
	dump_all_m_scores(args.output_folder, matrixes)
	dump_m_scores(args.output_folder, matrixes)

	# Get context lexicons comparisons.
	context_matrixes = context_lexicons_comparison(lexicons, matrixes)
	context_matrixes.to_csv(os.path.join(args.output_folder, 'c_scores.csv'))


if __name__ == '__main__':
	main()
