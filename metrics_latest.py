"""
    2Wiki-Multihop QA evaluation script
    Adapted from HotpotQA evaluation at https://github.com/hotpotqa/hotpot
"""
import sys
import ujson as json
import re
import string
from collections import Counter
import pickle
import numpy as np
from argparse import ArgumentParser

def extract_answer_deeprag(cot):
    import re
    cot = cot.split("<end>")[0]
    if "<answer short>" in cot:
        return cot.split("<answer short>")[-1].split("</answer short>")[0]
    elif "<answer long>" in cot:
        return cot.split("<answer long>")[-1].split("</answer long>")[0]
    else:
        cot = cot.split("</answer long>")[0].split("</answer short>")[0]
    
    if cot.endswith("<|im_end|>"):
        cot = cot[:-len("<|im_end|>")]
    if cot.endswith("<|eot_id|>"):
        cot = cot[:-len("<|eot_id|>")]
    pattern = r'<answer>([^<]+)</answer>(?!.*<answer>)'
    match = re.findall(pattern, cot)
    # print(cot)
    if len(match)>0:
        last_answer = match[-1]
        # print(last_answer)
        return last_answer
    else:
        return cot


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(
        prediction,
        ground_truth
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set([str(i) for i in ground_truth])
        correct = np.max([int(normalize_answer(gt) == normalize_answer(prediction)) for gt in ground_truths])
        cover_em = np.max([int(normalize_answer(gt) in normalize_answer(prediction)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct, 'cover_em': cover_em }


def f1_score(
        prediction,
        ground_truth
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set([str(i) for i in ground_truth])
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

def compute_metrics(entries):
    metrics = ["EM","Accuracy", "F1", "Precision", "Recall"]
    value = [[] for _ in range(len(metrics))]

    for entry in entries:
        gold_answers = [entry['gold']]

        predicted = entry['predicted']

        if any(empty_marker in normalize_answer(predicted) for empty_marker in ['no answer','none','null','not found','not possible','unknown','not enough']):
            predicted = 'noanswer'

        em_ret = exact_match_score(
                predicted, 
                gold_answers, 
            )
        f1_ret = f1_score(
            predicted, 
            gold_answers, 
        )
        value[0].append(em_ret["correct"])
        value[1].append(em_ret["cover_em"])
        for i, k in enumerate(f1_ret.keys()):
            value[i+1].append(f1_ret[k])

    ret = []
    for i, metric in enumerate(metrics[:3]):
        val = np.array(value[i])
        ret.append([metric, val.mean()])

    return ret[0], ret[1], ret[2]





if __name__ == '__main__':

    # parser = ArgumentParser()
    # parser.add_argument("--predictions_path",type=str,required=True)
    # args = parser.parse_args()
    # with open(args.predictions_path, 'r') as file:
    #     entries = json.load(file)

    parser = ArgumentParser()
    parser.add_argument("--predictions_path",type=str,required=True)
    parser.add_argument("--deeprag",type=bool,default=False)
    args = parser.parse_args()

    entries = []
    if 'jsonl' in args.predictions_path:
        with open(args.predictions_path, 'r') as file:
            for line in file: entries.append(json.loads(line))
    else:
        with open(args.predictions_path, 'r') as file:
            entries = json.load(file)


    predictions = []
    if args.deeprag:
        for entry in entries:
            pred = entry["prediction"]
            if isinstance(pred, list):
                if pred[-1] == False or pred[-1] is None:
                    pred = pred[-2]
                else:
                    pred = pred[-1]
            if isinstance(pred, dict):            
                pred = pred["answer"]

            if pred.endswith("<|im_end|>"):
                pred = pred[:-len("<|im_end|>")]
            if pred.endswith("<|eot_id|>"):
                pred = pred[:-len("<|eot_id|>")]
            predictions.append({'predicted':extract_answer_deeprag(pred),'gold':entry['answer']})

    else:
        predictions = entries
    
    print(compute_metrics(predictions))

   
