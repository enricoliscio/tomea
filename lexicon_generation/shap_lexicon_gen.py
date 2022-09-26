import numpy as np
import pandas as pd
import os
import shap
import argparse
import sys
import time
import datetime
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig, TextClassificationPipeline, pipeline

MFT_values = ['care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal',
              'authority', 'subversion', 'purity', 'degradation', 'non-moral']


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate value lexicon from trained BERT model.')
    parser.add_argument('--corpus', help='One of the seven MFTC datasets.',  type=str)
    parser.add_argument('--path',   help='Detination of the lexicon files.', type=str)

    return parser.parse_args(args)


def make_config(args):
    config = {}
    config['corpus'] = args.corpus
    config['input_file'] = f"data/MFTC/{config['corpus']}.csv"
    config['model_path'] = f"trained_models/bert_target_{config['corpus']}"
    config['output_folder'] = os.path.join(args.path, f"{config['corpus']}_{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}")
    config['class_names'] = MFT_values

    return config


def get_prediction_pipeline(model_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print(f'We will use the GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model_config = BertConfig.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, config=model_config)

    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    #return TextClassificationPipeline(model=model, tokenizer=tokenizer, num_workers=2, return_all_scores=True)


def get_feature_importance(shap_values, label):
    cohorts = {"": shap_values[:, :, label].mean(0)}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)

    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])

    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)

    return feature_importance


def dump_feature_importance(shap_values, config):
    for value in config['class_names']:
        feature_importance = get_feature_importance(shap_values, value)
        feature_importance.to_csv(
                os.path.join(config['output_folder'], f"shap_scores_{config['corpus']}_{value}.csv"))


def main(args=None):
    start_time = time.perf_counter()

    # Parse command line arguments.
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Parse command line and configuration file settings.
    config = make_config(args)

    # If the output folder doesn't exist, create it.
    if not os.path.isdir(config['output_folder']):
        os.makedirs(config['output_folder'])

    pred = get_prediction_pipeline(config['model_path'])

    with open(config['input_file'], 'r', encoding='utf-8') as f_in:
        df = pd.read_csv(f_in)

    # data = df['text'][:20]
    data = df['text']  # uncomment this line for using the whole dataset

    shap.initjs()
    explainer = shap.Explainer(pred, output_names=config['class_names'])
    shap_values = explainer(data)

    dump_feature_importance(shap_values, config)

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f'Total running time: {datetime.timedelta(seconds=duration)}')


if __name__ == "__main__":
    main()
