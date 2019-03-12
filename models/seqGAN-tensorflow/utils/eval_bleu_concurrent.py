import nltk
import random
import cPickle
import multiprocessing
from multiprocessing import Pool

SAMPLES = 1000
REFSIZE = 5000

def run_f(ele):
    reference, fn, weight = ele
    BLEUscore_f = nltk.translate.bleu_score.sentence_bleu(reference, fn, weight)  
    return BLEUscore_f


def belu_eval(generate_file, log_file):
    log = open(log_file, 'w')
    data_Name = "cotra"
    vocab_file = "vocab_" + data_Name + ".pkl"

    word, vocab = cPickle.load(open('save/'+vocab_file))

    pad = vocab[' ']
    print pad

    reference_file = 'save/realtest_coco.txt'
    reverse_ref_file = 'save/reverse_ref.txt'
    hypothesis_file = 'save/' + generate_file

    #################################################
    reference = []
    with open(reference_file)as fin:
        for line in fin:
            line = line.split()
            while line[-1] == str(pad):
                line.remove(str(pad))
            reference.append(line)
    #################################################
    hypothesis_list = []
    with open(hypothesis_file) as fin:
        for line in fin:
            line = line.split()
            while len(line) > 0 and line[-1] == str(pad):
                line.remove(str(pad))
            # 4839 is vocab size
            if all(i < 4839 for i in map(int, line)) is False:
                continue
            hypothesis_list.append(line)
    #################################################
    hypo2 = []
    with open(reverse_ref_file) as fin:
        for line in fin:
            line = line.split()
            while len(line) > 0 and line[-1] == str(pad):
                line.remove(str(pad))
            hypo2.append(line)
    #################################################
    random.shuffle(hypothesis_list)
    ref2 = hypothesis_list[:REFSIZE]
    #################################################

    for ngram in range(2, 6):
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(multiprocessing.cpu_count())
        bleu_irl = pool.map(run_f, [(reference, hypothesis_list[i], weight) for i in range(SAMPLES)])
        bleu_irl2 = pool.map(run_f, [(ref2, hypo2[i], weight) for i in range(SAMPLES)])
        pool.close()
        pool.join()

        print ('irl_text')
        print(len(weight), '-gram BLEU(b) score : ', 1.0 * sum(bleu_irl2) / len(bleu_irl2))
        print(len(weight), '-gram BLEU(f) score : ', 1.0 * sum(bleu_irl) / len(bleu_irl))

        log.write(str(len(weight)) + '-gram BLEU(b) score : ' + str(1.0 * sum(bleu_irl2) / len(bleu_irl2)) + '\n')
        log.write(str(len(weight)) + '-gram BLEU(f) score : ' + str(1.0 * sum(bleu_irl) / len(bleu_irl)) + '\n')
    log.close()


if __name__ == '__main__':
    belu_eval('evaler_file0.3550', 'save/evaler_file0.3550n.log')
