import nltk
import random
import cPickle
import multiprocessing
from multiprocessing import Pool

SAMPLES = 1000
chencherry = nltk.translate.bleu_score.SmoothingFunction()

def run_f(ele):
    reference, fn, weight = ele
    BLEUscore_f = nltk.translate.bleu_score.sentence_bleu(reference, fn, weight, smoothing_function=chencherry.method1)
    return BLEUscore_f


def belu_eval(generate_file, log_file):
    log = open(log_file, 'w')
    data_Name = "cotra"
    vocab_file = "vocab_" + data_Name + ".pkl"

    word, vocab = cPickle.load(open('save/'+vocab_file))

    pad = vocab[' ']
    print pad

    reference_file = 'save/realtest_coco.txt'
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
            if all(i < 4839 for i in map(int, line)) is False:
                continue
            hypothesis_list.append(line)
    #################################################
    random.shuffle(hypothesis_list)
    ref2 = hypothesis_list[:SAMPLES]
    #################################################

    for ngram in range(2, 6):
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(multiprocessing.cpu_count())
        bleu_irl = pool.map(run_f, [(reference, hypothesis_list[i], weight) for i in range(SAMPLES)])
        bleu_irl2 = pool.map(run_f, [(ref2[:i]+ref2[i+1:SAMPLES], ref2[i], weight) for i in range(SAMPLES)])
        pool.close()
        pool.join()

        print ('irl_text')
        print(len(weight), '-gram self BLEU score : ', 1.0 * sum(bleu_irl2) / len(bleu_irl2))
        print(len(weight), '-gram BLEU score : ', 1.0 * sum(bleu_irl) / len(bleu_irl))

        log.write(str(len(weight)) + '-gram self BLEU score : ' + str(1.0 * sum(bleu_irl2) / len(bleu_irl2)) + '\n')
        log.write(str(len(weight)) + '-gram BLEU score : ' + str(1.0 * sum(bleu_irl) / len(bleu_irl)) + '\n')
    log.close()

if __name__ == '__main__':
    belu_eval('evaler_file0.3550', 'save/evaler_file0.3550n.log')
   # belu_eval('evaler_file0.4550', 'save/evaler_file0.4550n.log')
   # belu_eval('evaler_file0.5550', 'save/evaler_file0.5550n.log')
   # belu_eval('evaler_file0.6550', 'save/evaler_file0.6550n.log')
   # belu_eval('evaler_file0.7550', 'save/evaler_file0.7550n.log')
