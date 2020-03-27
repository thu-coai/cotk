import numpy as np
import json
import copy
from metric_base import *
from nltk.translate.bleu_score import corpus_bleu

def metric_output(metric_class, data, dataloader=None):
    name = metric_class._name
    version = metric_class._version
    if isinstance(dataloader, FakeMultiDataloader):
        tmp_file_id = "../dataloader/dummy_ubuntucorpus#Ubuntu"
    else:
        tmp_file_id = '../dataloader/dummy_languageprocessing'
    if dataloader:
        tmp_set_names = ["train", "dev", "test"]
        tmp_vocab = GeneralVocab.from_predefined(data['init']['dataloader']['all_vocab_list'], \
														 	data['init']['dataloader']['valid_vocab_len'])
        tmp_toker = SimpleTokenizer('space', ['<pad>', '<unk>', '<go>', '<eos>'])
        if isinstance(dataloader, FakeMultiDataloader):
            tmp_sent = SessionDefault(tmp_toker, tmp_vocab, convert_to_lower_letter=True)
            tmp_fields = {set_name: {'session': tmp_sent} for set_name in tmp_set_names}
        else:
            tmp_sent = SentenceDefault(tmp_toker, tmp_vocab, convert_to_lower_letter=True)
            tmp_fields = {set_name: {'sent': tmp_sent} for set_name in tmp_set_names}

        tmp_dataloader = dataloader.simple_create(tmp_file_id, tmp_fields)
        if isinstance(dataloader, FakeMultiDataloader):
            tmp_dataloader.set_default_field("train", "session")
        else:
            tmp_dataloader.set_default_field("train", "sent")
        data['init']['dataloader'] = tmp_dataloader

    metric = metric_class(**data['init'])
    for batch in data['forward']:
        metric.forward(**batch)
    return metric.close()

def get_vocab(size):
    elm = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>"]
    for _ in range(size):
        while True:
            word = []
            wlen = np.random.randint(1, 10)
            for i in range(wlen):
                word.append(np.random.choice(elm)[0])
            word_str = "".join(word)
            if word_str not in all_vocab_list:
                break
        all_vocab_list.append("".join(word))
    return all_vocab_list

def get_prec_rec_data(all_vocab_list, valid_vocab_len, num_gen_per_inst):
    go_id = 2
    eos_id = 3
    unk_id = 1
    all_vocab_ids = list(range(2)) + list(range(4, len(all_vocab_list)))
    valid_vocab_ids = list(range(2)) + list(range(4, valid_vocab_len))
    for ref_rng, gen_rng, num_insts in zip([[0, 5], [20, 25], [0, 5], [20, 25]], [[0, 5], [20, 25], [20, 25], [0, 5]], [20] * 4):
        reference = []
        gen = []
        for cid in range(num_insts):
            tmp_ref = []
            ngrams = set()
            for _ in range(np.random.randint(1, 2 * num_gen_per_inst)):
                rlen = np.random.randint(*ref_rng)
                r = [int(word) for word in np.random.choice(all_vocab_ids, rlen)]
                tmp_ref.append([go_id] + r + [eos_id])
                for gramlen in range(1, min(rlen + 1, 8)):
                    for i in range(rlen - gramlen + 1):
                        ngrams.add(tuple(r[i: i + gramlen]))
            reference.append(tmp_ref)
            ngrams = list(ngrams)
            tmp_gen = []
            for _ in range(num_gen_per_inst):
                g = []
                for __ in range(2):
                    g += [int(word) for word in np.random.choice(valid_vocab_ids, np.random.randint(0, 3, 1))]
                    if ngrams and np.random.rand() < 0.5:
                        g += list(ngrams[np.random.choice(list(range(len(ngrams))), 1)[0]])
                g += [int(word) for word in np.random.choice(valid_vocab_ids, np.random.randint(0, 3, 1))]
                tmp_gen.append(g + [eos_id])
            gen.append(tmp_gen)
        idx = list(range(len(gen)))
        np.random.shuffle(idx)
        reference = [reference[i] for i in idx]
        gen = [gen[i] for i in idx]
        yield reference, gen

def get_ref_gen(all_vocab_list, valid_vocab_len, empty_ref, empty_gen):
    go_id = 2
    eos_id = 3
    unk_id = 1
    all_vocab_ids = list(range(2)) + list(range(4, len(all_vocab_list)))
    valid_vocab_ids = list(range(2)) + list(range(4, valid_vocab_len))
    if empty_ref and empty_gen:
        return [[go_id, eos_id]], [[eos_id]]
    if empty_ref and not empty_gen:
        return [[go_id, eos_id]], [[int(word) for word in np.random.choice(valid_vocab_ids, 10)]]
    if not empty_ref and empty_gen:
        return [[go_id] + [int(word) for word in np.random.choice(valid_vocab_ids, 10)] + [eos_id]], [[eos_id]]
    reference = []
    gen = []
    for rng, num_insts in zip([[0, 5], [20, 25]], [20, 80]):
        for _ in range(num_insts):
            rlen = np.random.randint(*rng)
            r = [int(word) for word in np.random.choice(all_vocab_ids, rlen)]
            ngrams = set()
            for gramlen in range(1, min(rlen + 1, 8)):
                for i in range(rlen - gramlen + 1):
                    ngrams.add(tuple(r[i: i + gramlen]))
            ngrams = list(ngrams)
            g = []
            for __ in range(2):
                g += [int(word) for word in np.random.choice(valid_vocab_ids, np.random.randint(0, 3, 1))]
                if ngrams and np.random.rand() < 0.5:
                    g += list(ngrams[np.random.choice(list(range(len(ngrams))), 1)[0]])
            g += [int(word) for word in np.random.choice(valid_vocab_ids, np.random.randint(0, 3, 1))]
            reference.append([go_id] + r + [eos_id])
            gen.append(g + [eos_id])
    idx = list(range(len(gen)))
    np.random.shuffle(idx)
    reference = [reference[i] for i in idx]
    gen = [gen[i] for i in idx]
    assert len(gen) == len(reference)
    return reference, gen

def get_ref_prob_len(all_vocab_list, valid_vocab_len, use_all_vocab):
    go_id = 2
    eos_id = 3
    unk_id = 1
    all_vocab_ids = list(range(2)) + list(range(4, len(all_vocab_list)))
    valid_vocab_ids = list(range(2)) + list(range(4, valid_vocab_len)) if not use_all_vocab else all_vocab_ids
    vocab_size = valid_vocab_len if not use_all_vocab else len(all_vocab_list)
    reference = []
    gen = []
    gen_log_prob = []
    for rng, num_insts in zip([[0, 5], [20, 25]], [20, 80]):
        for cid in range(num_insts):
            rlen = np.random.randint(*rng)
            r = [int(word) for word in np.random.choice(all_vocab_ids, rlen)]
            ngrams = set()
            for gramlen in range(1, min(8, rlen + 1)):
                for i in range(rlen - gramlen + 1):
                    ngrams.add(tuple(r[i: i + gramlen]))
            ngrams = list(ngrams)
            g = []
            for i, word in enumerate(r):
                if np.random.rand() < 0.4:
                    g.append(word)
                else:
                    g.append(int(np.random.choice(all_vocab_ids, 1)[0]))
            reference.append([go_id] + r + [eos_id])
            g += [eos_id]
            gen.append(g)
    idx = list(range(len(gen)))
    np.random.shuffle(idx)
    reference = [reference[i] for i in idx]
    gen = [gen[i] for i in idx]

    max_gen_len = max([len(g) for g in gen])
    for g in gen:
        sent_prob = []
        for word in g:
            if not use_all_vocab and word >= valid_vocab_len:
                word = unk_id
            dis = np.random.randint(1, 1000, vocab_size)
            dis[word] += np.sum(dis)
            sent_prob.append(np.log(dis / np.sum(dis)).tolist())
        for _ in range(max_gen_len - len(g)):
            dis = np.random.randint(1, 1000, vocab_size)
            sent_prob.append(np.log(dis / np.sum(dis)).tolist())
        gen_log_prob.append(sent_prob)
    return reference, gen_log_prob, [len(r) for r in reference]


from cotk.metric import AccuracyMetric
with open('version_test_data/AccuracyMetric_v2.jsonl', 'w') as file:
    obj = {'init': {'dataloader': None, 'label_key': '_label', 'prediction_key': '_prediction'},
          'forward': []}
    l = []
    p = []
    for i, offset in zip(range(10), range(9, -1, -1)):
        cnt = offset + 1
        l.extend([i] * cnt * 2)
        p.extend([i + offset] * cnt)
        p.extend([i - offset] * cnt)
        print(i, offset, cnt)
    idx = list(range(len(l)))
    np.random.shuffle(idx)
    l = [l[i] for i in idx]
    p = [p[i] for i in idx]
    mid = len(l) // 2
    obj['forward'].append({'data': {'_label': l[:mid], '_prediction': p[:mid]}})
    obj['forward'].append({'data': {'_label': l[mid:], '_prediction': p[mid:]}})
    accuracy_metric = AccuracyMetric(**obj['init'])
    obj['output'] = metric_output(AccuracyMetric, obj, dataloader=None)
    file.write(json.dumps(obj))

from cotk.metric import BleuCorpusMetric
with open("version_test_data/BleuCorpusMetric_v2.jsonl", "w") as file:
    for empty_ref in [True, False]:
        for empty_hyp in [True, False]:
            all_vocab_list = get_vocab(20)
            valid_vocab_len = 20
            obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                            'ignore_smoothing_error': False, 'reference_allvocabs_key':'_ref_allvocabs', 'gen_key': '_gen'},
                  'forward': []}
            reference, gen = get_ref_gen(all_vocab_list, valid_vocab_len, empty_ref, empty_hyp)
            if len(gen) == 1:
                obj['forward'].append({'data': {'_ref_allvocabs': reference, '_gen': gen}})
            else:
                mid = len(gen) // 2
                obj['forward'].append({'data': {'_ref_allvocabs': reference[:mid], '_gen': gen[:mid]}})
                obj['forward'].append({'data': {'_ref_allvocabs': reference[mid:], '_gen': gen[mid:]}})
            obj['output'] = metric_output(BleuCorpusMetric, copy.deepcopy(obj), FakeDataLoader())
            file.write(json.dumps(obj) + "\n")


from cotk.metric import SelfBleuCorpusMetric
with open("version_test_data/SelfBleuCorpusMetric_v2.jsonl", "w") as file:
    for sample in [10, 1000]:
        all_vocab_list = get_vocab(20)
        valid_vocab_len = 20
        obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                        'gen_key': '_gen', 'sample': sample, 'seed': 1229, 'cpu_count': None},
              'forward': []}
        reference, gen = get_ref_gen(all_vocab_list, valid_vocab_len, False, False)
        mid = len(gen) // 2
        obj['forward'].append({'data': {'_gen': gen[:mid]}})
        obj['forward'].append({'data': {'_gen': gen[mid:]}})
        obj['output'] = metric_output(SelfBleuCorpusMetric, copy.deepcopy(obj), FakeDataLoader())
        file.write(json.dumps(obj) + "\n")


from cotk.metric import FwBwBleuCorpusMetric
with open("version_test_data/FwBwBleuCorpusMetric_v2.jsonl", "w") as file:
    for test_sz, gen_sz in zip([50, 100], [100, 50]):
        for sample in [25, 75, 125]:
            all_vocab_list = get_vocab(20)
            valid_vocab_len = 20
            reference, gen = get_ref_gen(all_vocab_list, valid_vocab_len, False, False)
            reference = reference[:test_sz]
            gen = gen[:gen_sz]
            obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                            'reference_test_list': reference,
                            'gen_key': '_gen', 'sample': sample, 'seed': 1229, 'cpu_count': None},
                  'forward': []}
            mid = len(gen) // 2
            obj['forward'].append({'data': {'_gen': gen[:mid]}})
            obj['forward'].append({'data': {'_gen': gen[mid:]}})
            obj['output'] = metric_output(FwBwBleuCorpusMetric, copy.deepcopy(obj), FakeDataLoader())
            file.write(json.dumps(obj) + "\n")


from cotk.metric import MultiTurnBleuCorpusMetric
with open("version_test_data/MultiTurnBleuCorpusMetric_v2.jsonl", "w") as file:
    all_vocab_list = get_vocab(20)
    valid_vocab_len = 20
    reference, _gen = get_ref_gen(all_vocab_list, valid_vocab_len, False, False)
    turn_length = []
    turn = []
    gen = []
    s = 0
    while s < len(reference):
        l = np.random.randint(1, min(10, len(reference) - s + 1))
        turn_length.append(l)
        turn.append(reference[s: s + l])
        gen.append(_gen[s: s + l])
        s += l
    obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                   'ignore_smoothing_error': True,
                   'multi_turn_reference_allvocabs_key': '_reference_allvocabs',
                   'multi_turn_gen_key': '_multi_turn_gen',
                   'turn_len_key': '_turn_length'},
          'forward': []}
    mid = len(turn) // 2
    obj['forward'].append({'data': {'_reference_allvocabs': turn[:mid], '_multi_turn_gen': gen[:mid], '_turn_length': turn_length[:mid]}})
    obj['forward'].append({'data': {'_reference_allvocabs': turn[mid:], '_multi_turn_gen': gen[mid:], '_turn_length': turn_length[mid:]}})
    obj['output'] = metric_output(MultiTurnBleuCorpusMetric, copy.deepcopy(obj), FakeMultiDataloader())
    file.write(json.dumps(obj) + "\n")



from cotk.metric import PerplexityMetric
with open("version_test_data/PerplexityMetric_v2.jsonl", "w") as file:
    for use_all_vocab in [True, False]:
        all_vocab_list = get_vocab(20)
        valid_vocab_len = 20
        reference, gen_log_prob, gen_len = get_ref_prob_len(all_vocab_list, valid_vocab_len, use_all_vocab)
        obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                       'reference_allvocabs_key': '_ref_allvocabs',
                        'reference_len_key': '_ref_length',
                       'gen_log_prob_key': '_gen_log_prob',
                       #'invalid_vocab': use_all_vocab,
                       'generate_rare_vocab': use_all_vocab,
                       'full_check': False},
              'forward': []}
        mid = len(reference) // 2
        obj['forward'].append({'data': {'_ref_allvocabs': reference[:mid], '_ref_length': gen_len[:mid], '_gen_log_prob': gen_log_prob[:mid]}})
        obj['forward'].append({'data': {'_ref_allvocabs': reference[mid:], '_ref_length': gen_len[mid:], '_gen_log_prob': gen_log_prob[mid:]}})
        obj['output'] = metric_output(PerplexityMetric, copy.deepcopy(obj), FakeDataLoader())
        file.write(json.dumps(obj) + "\n")



from cotk.metric import MultiTurnPerplexityMetric
with open("version_test_data/MultiTurnPerplexityMetric_v2.jsonl", "w") as file:
    for use_all_vocab in [True, False]:
        all_vocab_list = get_vocab(20)
        valid_vocab_len = 20
        _reference, _gen_log_prob, _gen_len = get_ref_prob_len(all_vocab_list, valid_vocab_len, use_all_vocab)
        reference = []
        gen_log_prob = []
        gen_len = []
        s = 0
        while s < len(_reference):
            l = np.random.randint(1, min(10, len(_reference) - s + 1))
            gen_len.append(_gen_len[s: s + l])
            reference.append(_reference[s: s + l])
            gen_log_prob.append(_gen_log_prob[s: s + l])
            if l > 3:
                gen_len[-1][1] = 0
            s += l
        obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                       'multi_turn_reference_allvocabs_key': '_multi_turn_ref_allvocabs',
                        'multi_turn_reference_len_key': '_multi_turn_ref_length',
                       'multi_turn_gen_log_prob_key': '_multi_turn_gen_log_prob',
                       'generate_rare_vocab': use_all_vocab,
                       'full_check': False},
              'forward': []}
        mid = len(reference) // 2
        obj['forward'].append({'data': {'_multi_turn_ref_allvocabs': reference[:mid], '_multi_turn_ref_length': gen_len[:mid], '_multi_turn_gen_log_prob': gen_log_prob[:mid]}})
        obj['forward'].append({'data': {'_multi_turn_ref_allvocabs': reference[mid:], '_multi_turn_ref_length': gen_len[mid:], '_multi_turn_gen_log_prob': gen_log_prob[mid:]}})
        obj['output'] = metric_output(MultiTurnPerplexityMetric, copy.deepcopy(obj), FakeMultiDataloader())
        file.write(json.dumps(obj) + "\n")


from cotk.metric import NgramFwBwPerplexityMetric
with open("version_test_data/NgramFwBwPerplexityMetric_v2.jsonl", "w") as file:
    for ngram in range(1, 5):
        all_vocab_list = get_vocab(20)
        valid_vocab_len = 20
        reference, gen = get_ref_gen(all_vocab_list, valid_vocab_len, False, False)
        obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                        'ngram': ngram,
                        'reference_test_list': reference, 'gen_key': '_gen', 'cpu_count': None},
              'forward': []}
        mid = len(gen) // 2
        obj['forward'].append({'data': {'_gen': gen[:mid]}})
        obj['forward'].append({'data': {'_gen': gen[mid:]}})
        obj['output'] = metric_output(NgramFwBwPerplexityMetric, copy.deepcopy(obj), FakeDataLoader())
        file.write(json.dumps(obj) + "\n")


from cotk.metric import BleuPrecisionRecallMetric
with open("version_test_data/BleuPrecisionRecallMetric_v2.jsonl", "w") as file:
    for ngram in range(1, 4):
        all_vocab_list = get_vocab(20)
        valid_vocab_len = 20
        for reference, gen in get_prec_rec_data(all_vocab_list, valid_vocab_len, 3):
            obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                           'ngram': ngram, 'generated_num_per_context': 3, 'candidates_allvocabs_key': '_candidate_allvocabs',
                           'multiple_gen_key': '_multiple_gen'},
                  'forward': []}
            mid = len(gen) // 2
            obj['forward'].append({'data': {'_candidate_allvocabs': reference[:mid], '_multiple_gen': gen[:mid]}})
            obj['forward'].append({'data': {'_candidate_allvocabs': reference[mid:], '_multiple_gen': gen[mid:]}})
            obj['output'] = metric_output(BleuPrecisionRecallMetric, copy.deepcopy(obj), FakeMultiDataloader())
            file.write(json.dumps(obj) + "\n")


from cotk.metric import EmbSimilarityPrecisionRecallMetric
with open("version_test_data/EmbSimilarityPrecisionRecallMetric_v2.jsonl", "w") as file:
    for mode in ['avg', 'extrema']:
        all_vocab_list = get_vocab(20)
        valid_vocab_len = 20
        word2vec = {}
        for word in all_vocab_list[2: valid_vocab_len] + [all_vocab_list[0]]:
            emb = (np.random.choice(list(range(3)), 1)[0] + np.random.rand(10)).tolist()
            word2vec[word] = emb
        for reference, gen in get_prec_rec_data(all_vocab_list, valid_vocab_len, 3):
            obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                           'word2vec': word2vec, 'mode': mode, 'generated_num_per_context': 3, 'candidates_allvocabs_key': '_candidate_allvocabs',
                           'multiple_gen_key': '_multiple_gen'},
                  'forward': []}
            mid = len(gen) // 2
            obj['forward'].append({'data': {'_candidate_allvocabs': reference[:mid], '_multiple_gen': gen[:mid]}})
            obj['forward'].append({'data': {'_candidate_allvocabs': reference[mid:], '_multiple_gen': gen[mid:]}})
            obj['output'] = metric_output(EmbSimilarityPrecisionRecallMetric, copy.deepcopy(obj), FakeMultiDataloader())
            file.write(json.dumps(obj) + "\n")
    # empty gen
    obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                           'word2vec': word2vec, 'mode': mode, 'generated_num_per_context': 3, 'candidates_allvocabs_key': '_candidate_allvocabs',
                           'multiple_gen_key': '_multiple_gen'},
                  'forward': [{'data': {'_candidate_allvocabs': reference[:1], '_multiple_gen': [[[3]] * 3]}}]}
    obj['output'] = metric_output(EmbSimilarityPrecisionRecallMetric, copy.deepcopy(obj), FakeMultiDataloader())
    file.write(json.dumps(obj) + "\n")
    # empty ref
    obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                           'word2vec': word2vec, 'mode': mode, 'generated_num_per_context': 3, 'candidates_allvocabs_key': '_candidate_allvocabs',
                           'multiple_gen_key': '_multiple_gen'},
                  'forward': [{'data': {'_candidate_allvocabs': [[[2, 3]]], '_multiple_gen': gen[:1]}}]}
    obj['output'] = metric_output(EmbSimilarityPrecisionRecallMetric, copy.deepcopy(obj), FakeMultiDataloader())
    file.write(json.dumps(obj) + "\n")
    # empty ref & gen
    obj = {'init': {'dataloader': {'all_vocab_list': all_vocab_list, 'valid_vocab_len': valid_vocab_len},
                           'word2vec': word2vec, 'mode': mode, 'generated_num_per_context': 3, 'candidates_allvocabs_key': '_candidate_allvocabs',
                           'multiple_gen_key': '_multiple_gen'},
                  'forward': [{'data': {'_candidate_allvocabs': [[[2, 3]]], '_multiple_gen': [[[3]] * 3]}}]}
    obj['output'] = metric_output(EmbSimilarityPrecisionRecallMetric, copy.deepcopy(obj), FakeMultiDataloader())
    file.write(json.dumps(obj) + "\n")
