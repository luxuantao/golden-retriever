import os
import time
import torch
import logging
import json
from DrQA.drqa.reader import Predictor
from search.search import bulk_text_query
from utils.general import chunks, make_context

import random
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path
import argparse
from joblib import Parallel, delayed
from BiDAFpp.util import NUM_OF_PARAGRAPHS, MAX_PARAGRAPH_LEN
from stanfordnlp.server import CoreNLPClient
import bisect
import re

from torch import optim, nn
from BiDAFpp.model import Model #, NoCharModel, NoSelfModel
from BiDAFpp.sp_model import SPModel
from BiDAFpp.util import convert_tokens, evaluate
from BiDAFpp.util import get_buckets, HotpotDataset, DataIterator, IGNORE_INDEX
import shutil

import logging

predictor1, predictor2 = None, None  # 预测模型
word_mat, char_mat = None, None  # qa answer用
logger = None

def init():
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        os.system('rm corenlp_server-*')
    except Exception as e:
        pass
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    # print('加载预测模型...')
    global predictor1, predictor2
    predictor1 = Predictor(
        model="models/hop1.mdl",
        # tokenizer=args.tokenizer,
        # embedding_file=args.embedding_file,
        num_workers=1
    )
    predictor2 = Predictor(
        model="models/hop2.mdl",
        # tokenizer=args.tokenizer,
        # embedding_file=args.embedding_file,
        num_workers=1
    )
    # print('预测模型加载完毕')

    # print('加载词向量和字向量...')
    global word_mat, char_mat
    with open('word_emb_hops.json', "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open('char_emb_hops.json', "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    # print('词向量和字向量加载完毕')

    logger.info('初始化完毕')


def squadify_question(data):
    rows = []
    for entry in data:
        assert 'question' in entry, 'every entry must have a question'
        assert '_id' in entry, 'every entry must have an _id'
        row = {
            'title': '',
            'paragraphs': [{
                'context': entry['question'],
                'qas': [{
                    'question': entry['question'],
                    'id': entry['_id'],
                    'answers': [{'answer_start': 0, 'text': ''}]
                }]
            }]
        }
        rows.append(row)
    # print('原始问题格式化完毕')
    return {'data': rows}


def make_prediction(question, model, top_n=1, official=True):
    if model == "models/hop1.mdl":
        predictor = predictor1
    elif model == "models/hop2.mdl":
        predictor = predictor2

    examples = []
    qids = []
    data = question['data']
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                qids.append(qa['id'])
                examples.append((context, qa['question']))

    results = {}
    for i in range(0, len(examples), 128):
        predictions = predictor.predict_batch(
            examples[i:i + 128], top_n=top_n
        )
        for j in range(len(predictions)):
            # Official eval expects just a qid --> span
            if official:
                results[qids[i + j]] = predictions[j][0][0]
            # Otherwise we store top N and scores for debugging.
            else:
                results[qids[i + j]] = [(p[0], float(p[1])) for p in predictions[j]]
    # print('预测完毕')
    return results


def merge_with_es(query_data, question_data, top_n=5):
    out_data = []

    for chunk in list(chunks(question_data, 100)):
        queries = []
        for datum in chunk:
            _id = datum['_id']
            queries.append(query_data[_id] if isinstance(query_data[_id], str) else query_data[_id][0][0])

        es_results = bulk_text_query(queries, topn=top_n, lazy=False)
        for es_result, datum in zip(es_results, chunk):
            _id = datum['_id']
            question_t = datum['question']
            query = query_data[_id] if isinstance(query_data[_id], str) else query_data[_id][0][0]
            context = make_context(question_t, es_result)
            json_context = [
                [p['title'], p['data_object']['text']]
                for p in es_result
            ]

            out_data.append({
                '_id': _id,
                'question': question_t,
                'context': context,
                'query': query,
                'json_context': json_context
            })
    # print("查询es完毕")
    return out_data


def parse_data(data):
    rows = []
    SHUFFLE = False
    for d in data:
        row = {}
        row['title'] = ''
        paragraph = {}
        paragraph['context'] = d['context']
        qas = {}
        qas['question'] = d['question']

        # For test set evaluation, we don't have labels
        # Instead we just use (0, "")
        if 'label_offsets' in d:
            start = d['label_offsets'][0]
            span = d['context'][d['label_offsets'][0]:d['label_offsets'][1]]
        else:
            start = 0
            span = ''

        qas['answers'] = [{'answer_start': start, 'text': span}]
        qas['id'] = d['_id']
        paragraph['qas'] = [qas]
        row['paragraphs'] = [paragraph]
        rows.append(row)

    # print("第二次预测输入数据格式化完毕")
    return {'data': rows}


def merge_hops_results(hop1_data, hop2_data, include_queries=True, num_each=5):
    out_data = []

    for hop1, hop2 in zip(hop1_data, hop2_data):
        # We're assuming that the hop1 and hop2 files are sorted in the same
        # order. If this doesn't hold, then we would just make a map
        # {id -> entry} for one file.
        assert hop1['_id'] == hop2['_id']

        entry = {}
        entry['_id'] = hop1['_id']
        entry['question'] = hop1['question']
        if include_queries:
            entry['hop1_query'] = hop1['query']
            entry['hop2_query'] = hop2['query']

        entry['context'] = []
        all_titles = set()
        for doc in hop1['json_context'][:num_each] + hop2['json_context'][:num_each]:
            if doc[0] not in all_titles:
                entry['context'].append(doc)
                all_titles.add(doc[0])

        out_data.append(entry)

    # print("两次查询数据整合完毕")
    return out_data


###################### QA prepro ########################

tokenizer_client = CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=30000, memory='16G', properties={'tokenize.ptb3Escaping': False, 'tokenize.options': "splitHyphenated=true,invertible=true", 'ssplit.eolonly': True}, threads=1)


def word_tokenize(text):
    if isinstance(text, str):
        ann = tokenizer_client.annotate(text.replace('%', '%25').replace('<dfn id=">', '<dfn id=\'\'>'))
        res = [token.originalText.replace('<dfn id=\'\'>', '<dfn id=">') for token in ann.sentence[0].token]
        return res
    else:
        ann = tokenizer_client.annotate('\n'.join([x for x in text if len(x.strip())]).replace('%', '%25').replace('<dfn id=">', '<dfn id=\'\'>'))

        res = [[token.originalText.replace('<dfn id=\'\'>', '<dfn id=">') for token in sentence.token] for sentence in ann.sentence]

        res1 = []
        resi = 0
        for i, x in enumerate(text):
            if len(x.strip()) == 0:
                res1.append([])
            else:
                if not all(token in x for token in res[resi]):
                    print(x)
                    print(res[resi])
                assert all(token in x for token in res[resi])
                res1.append(res[resi])
                resi += 1
        return res1


def find_nearest(a, target, test_func=lambda x: True):
    idx = bisect.bisect_left(a, target)
    if (0 <= idx < len(a)) and a[idx] == target:
        return target, 0
    elif idx == 0:
        return a[0], abs(a[0] - target)
    elif idx == len(a):
        return a[-1], abs(a[-1] - target)
    else:
        d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
        d2 = abs(a[idx-1] - target) if test_func(a[idx-1]) else 1e200
        if d1 > d2:
            return a[idx-1], d2
        else:
            return a[idx], d1

def fix_span(para, offsets, span):
    span = span.strip()
    parastr = "".join(para)
    assert span in parastr, '{}\t{}'.format(span, parastr)
    begins, ends = map(list, zip(*[y for x in offsets for y in x]))

    best_dist = 1e200
    best_indices = None

    if span == parastr:
        return parastr, (0, len(parastr)), 0

    for m in re.finditer(re.escape(span), parastr):
        begin_offset, end_offset = m.span()

        fixed_begin, d1 = find_nearest(begins, begin_offset, lambda x: x < end_offset)
        fixed_end, d2 = find_nearest(ends, end_offset, lambda x: x > begin_offset)

        if d1 + d2 < best_dist:
            best_dist = d1 + d2
            best_indices = (fixed_begin, fixed_end)
            if best_dist == 0:
                break

    assert best_indices is not None
    return parastr[best_indices[0]:best_indices[1]], best_indices, best_dist


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        pre = current
        current = text.find(token, current)
        if current < 0:
            print(f'Token |{token}| not found in |{text}|')
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def prepro_sent(sent):
    return sent
    # return sent.replace("''", '" ').replace("``", '" ')


def _process_article(article):
    paragraphs = article['context']
    # some articles in the fullwiki dev/test sets have zero paragraphs
    if len(paragraphs) == 0:
        paragraphs = [['some random title', 'some random stuff']]

    text_context, context_tokens, context_chars = '', [], []
    offsets = []
    flat_offsets = []
    start_end_facts = [] # (start_token_id, end_token_id, is_sup_fact=True/False)
    sent2title_ids = []

    def _process(sent, sent_tokens, is_sup_fact, is_title=False):
        nonlocal text_context, context_tokens, context_chars, offsets, start_end_facts, flat_offsets
        N_chars = len(text_context)

        sent = sent
        #sent_tokens = word_tokenize(sent)
        if is_title:
            sent = ' <t> {} </t> '.format(sent)
            sent_tokens = ['<t>'] + sent_tokens + ['</t>']
            # Change 1: If we see a new paragraph we add an empty list (to which later we will add the paragraphs)
            context_tokens.append([])
            context_chars.append([])
            start_end_facts.append([])
        if len(context_tokens[-1]) >= MAX_PARAGRAPH_LEN:
            return
        # truncate sentence if paragraph is too long
        if len(context_tokens[-1]) + len(sent_tokens) >= MAX_PARAGRAPH_LEN:
            sent_tokens = sent_tokens[:MAX_PARAGRAPH_LEN - len(context_tokens[-1])]
        sent_chars = [list(token) for token in sent_tokens]
        sent_spans = convert_idx(sent, sent_tokens)
        if len(context_tokens[-1]) + len(sent_tokens) >= MAX_PARAGRAPH_LEN:
            sent = sent[:sent_spans[-1][1]]

        sent_spans = [[N_chars+e[0], N_chars+e[1]] for e in sent_spans]
        N_tokens, my_N_tokens = len(context_tokens[-1]), len(sent_tokens)

        text_context += sent
        # Change 2: Add items to the empty list
        ## First occurence of when a flattened list is made
        context_tokens[-1].extend(sent_tokens)
        context_chars[-1].extend(sent_chars)
        start_end_facts[-1].append((N_tokens, N_tokens+my_N_tokens, is_sup_fact))
        ## the above context tokens is then used to populate the context_idxs
        # end change
        offsets.append(sent_spans)
        flat_offsets.extend(sent_spans)

    if 'supporting_facts' in article:
        sp_set = set(list(map(tuple, article['supporting_facts'])))
    else:
        sp_set = set()

    to_tokenize = [prepro_sent(article['question'])]
    for para in paragraphs:
        to_tokenize.extend([para[0]])
        to_tokenize.extend(para[1])
    tokens = word_tokenize(to_tokenize)
    ques_tokens = tokens[0]
    tokens_id = 1

    for para in paragraphs:
        cur_title, cur_para = para[0], para[1]
        sent2title_ids.append((cur_title, -1))
        _process(prepro_sent(cur_title), tokens[tokens_id], False, is_title=True)
        tokens_id += 1
        for sent_id, sent in enumerate(cur_para):
            is_sup_fact = (cur_title, sent_id) in sp_set
            _process(prepro_sent(sent), tokens[tokens_id], is_sup_fact)
            tokens_id += 1
            sent2title_ids.append((cur_title, sent_id))

    if 'answer' in article:
        answer = article['answer'].strip()
        if answer.lower() == 'yes':
            best_indices = [-1, -1]
        elif answer.lower() == 'no':
            best_indices = [-2, -2]
        else:
            if article['answer'].strip() not in ''.join(text_context):
                # in the fullwiki setting, the answer might not have been retrieved
                # use (0, 1) so that we can proceed
                best_indices = (0, 1)
            else:
                _, best_indices, _ = fix_span(text_context, offsets, article['answer'])
                answer_span = []
                for idx, span in enumerate(flat_offsets):
                    if not (best_indices[1] <= span[0] or best_indices[0] >= span[1]):
                        answer_span.append(idx)
                best_indices = (answer_span[0], answer_span[-1])
    else:
        # some random stuff
        answer = 'random'
        best_indices = (0, 1)

    #ques_tokens = word_tokenize(prepro_sent(article['question']))
    ques_chars = [list(token) for token in ques_tokens]

    example = {'context_tokens': context_tokens,'context_chars': context_chars, 'ques_tokens': ques_tokens, 'ques_chars': ques_chars, 'y1s': [best_indices[0]], 'y2s': [best_indices[1]], 'id': article['_id'], 'start_end_facts': start_end_facts}
    eval_example = {'context': text_context, 'spans': flat_offsets, 'answer': [answer], 'id': article['_id'],
            'sent2title_ids': sent2title_ids}
    return example, eval_example


def process_file(data):
    eval_examples = {}
    outputs = [_process_article(article) for article in data]
    examples = [e[0] for e in outputs]
    for _, e in outputs:
        if e is not None:
            eval_examples[e['id']] = e
    # random.shuffle(examples)
    # print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def build_features(examples, out_file, word2idx_dict, char2idx_dict):
    para_limit, ques_limit = 0, 0
    for example in examples:
        para_limit = max(para_limit, len(example['context_tokens']))
        ques_limit = max(ques_limit, len(example['ques_tokens']))

    # char_limit = config.char_limit
    # char_limit = 16

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    # print("Processing {} examples...".format(data_type))
    datapoints = []
    total = 0
    total_ = 0
    for example in examples:
        total_ += 1

        if filter_func(example):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        # Convert text to indices (leave tensorization to the data iterator)
        context_idxs = [[_get_word(w) for w in para] for para in example['context_tokens']]
        ques_idxs = [_get_word(w) for w in example['ques_tokens']]

        context_char_idxs = [[[_get_char(c) for c in token] for token in para] for para in example['context_chars']]
        ques_char_idxs = [[_get_char(c) for c in token] for token in example['ques_chars']]

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1, y2 = start, end

        datapoints.append({'context_idxs': context_idxs,
            'context_char_idxs': context_char_idxs,
            'ques_idxs': ques_idxs,
            'ques_char_idxs': ques_char_idxs,
            'y1': y1,
            'y2': y2,
            'id': example['id'],
            'start_end_facts': example['start_end_facts']})
    # print("Build {} / {} instances of features in total".format(total, total_))
    torch.save(datapoints, out_file)
    # return datapoints


def prepro(data):
    random.seed(13)
    examples, eval_examples = process_file(data)

    with open('word2idx_hops.json', "r") as fh:
        word2idx_dict = json.load(fh)

    with open('char2idx_hops.json', "r") as fh:
        char2idx_dict = json.load(fh)

    record_file = 'fullwiki.test_record_hops.pkl'
    build_features(examples, record_file, word2idx_dict, char2idx_dict)

    with open('fullwiki.test_eval_hops.json', "w") as fh:
        json.dump(eval_examples, fh)
    # return build_features(examples, word2idx_dict, char2idx_dict), eval_examples


###################### QA answer ########################

def predict(data_source, model, eval_file):
    answer_dict = {}
    sp_dict = {}
    # sp_th = config.sp_threshold
    sp_th = 0.33
    for step, data in enumerate(data_source):
        with torch.no_grad():
            # if config.cuda:
            #     data = {k:(data[k].cuda() if k != 'ids' else data[k]) for k in data}
            context_idxs = data['context_idxs']
            ques_idxs = data['ques_idxs']
            context_char_idxs = data['context_char_idxs']
            ques_char_idxs = data['ques_char_idxs']
            context_lens = data['context_lens']
            start_mapping = data['start_mapping']
            end_mapping = data['end_mapping']
            all_mapping = data['all_mapping']

            logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, context_lens.sum(1).max().item(), return_yp=True)
            answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(predict_support[:, :, 1] - predict_support[:, :, 0]).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = data['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                if predict_support_np[i, j] > sp_th:
                    cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
            sp_dict.update({cur_id: cur_sp_pred})

    prediction = {'answer': answer_dict, 'sp': sp_dict}

    # print('最终回答:', list(prediction['answer'].values())[0])
    return list(prediction['answer'].values())[0]


def test():
    # with open('word_emb_hops.json', "r") as fh:
    #     word_mat = np.array(json.load(fh), dtype=np.float32)
    # with open('char_emb_hops.json', "r") as fh:
    #     char_mat = np.array(json.load(fh), dtype=np.float32)
    with open('fullwiki.test_eval_hops.json', 'r') as fh:
        dev_eval_file = json.load(fh)
    # with open('idx2word_hops.json', 'r') as fh:
    #     idx2word_dict = json.load(fh)

    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    # if config.cuda:
    #     torch.cuda.manual_seed_all(config.seed)

    para_limit = None
    ques_limit = None
    dev_buckets = get_buckets('fullwiki.test_record_hops.pkl')

    def build_dev_iterator():
        dev_dataset = HotpotDataset(dev_buckets)
        para_limit = 1000
        ques_limit = 80
        char_limit = 16
        sent_limit = 100
        batch_size = 16
        return DataIterator(dev_dataset, para_limit, ques_limit, char_limit, sent_limit, batch_size=batch_size, num_workers=2)

    sp_lambda = 10.0
    if sp_lambda > 0:
        model = SPModel(word_mat, char_mat)
    # else:
    #     model = Model(config, word_mat, char_mat)

    # ori_model = model.cuda() if config.cuda else model
    ori_model = model
    ori_model.load_state_dict(torch.load(os.path.join('QAModel', 'model.pt'), map_location=lambda storage, loc: storage))
#    model = nn.DataParallel(ori_model)
    model = ori_model

    model.eval()
    return predict(build_dev_iterator(), model, dev_eval_file)


def main(input_question):
    # input_question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    # input_question = "Were Donald Trump and Barack Obama of the same nationality?"
    start = time.time()

    question = [{"_id": "1", "question": input_question}]
    logger.info('输入问题为: %s', input_question)

    # 格式化原问题
    squadified_question = squadify_question(question)
    # print('用时：', time.time() - start)
    # print(squadified_question)

    # 第一次预测
    pred1_result = make_prediction(question=squadified_question, model="models/hop1.mdl")
    # print('用时：', time.time() - start)
    # print('第一次查询关键词:', list(pred1_result.values())[0])

    # 用第一次预测结果查询ES
    merged_result1 = merge_with_es(query_data=pred1_result, question_data=question)
    # print('用时：', time.time() - start)
    # print(merged_result1)

    # 格式化前面的得到的结果
    second_input = parse_data(merged_result1)
    # print('用时：', time.time() - start)
    # print(second_input)

    # 第二次预测
    pred2_result = make_prediction(question=second_input, model="models/hop2.mdl")
    # print('用时：', time.time() - start)
    # print('第二次查询关键词:', list(pred2_result.values())[0])

    # 用第二次预测结果查询ES
    merged_result2 = merge_with_es(query_data=pred2_result, question_data=question)
    # print('用时：', time.time() - start)
    # print(merged_result2)

    two_hops_results = merge_hops_results(merged_result1, merged_result2)
    # print('用时：', time.time() - start)
    # print('问答系统输入:', two_hops_results)

    # python main.py --mode prepro --data_file ../mine/qa_input.json --para_limit 2250 --data_split test --fullwiki
    prepro(data=two_hops_results)
    # print('用时：', time.time() - start)

    # python main.py --mode test --data_split test --save QAModel --prediction_file ../mine/golden.json --sp_threshold .33 --sp_lambda
    result = test()
    logger.info('最终回答为: %s', result)
    # print('用时：', time.time() - start)

    return {'question': input_question, 'answer': result if result != '<t> some random title' else 'Sorry, I have no idea...', 'hop1': two_hops_results[0]['hop1_query'], 'hop2': two_hops_results[0]['hop2_query'], 'hop1_document': two_hops_results[0]['context'][0][1][0] if len(two_hops_results[0]['context']) > 0 else '', 'hop2_document': two_hops_results[0]['context'][5][1][0] if len(two_hops_results[0]['context']) > 5 else ''}
