r"""
``contk.metrics`` provides functions evaluating results of models. It provide 
a fair metric for every model.
"""
from nltk.translate.bleu_score import sentence_bleu

def bleu1(references, hypothesis):
	"""Calculate BLEU score (Bilingual Evaluation Understudy) from
	Papineni, Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.
	"BLEU: a method for automatic evaluation of machine translation."
	In Proceedings of ACL. http://www.aclweb.org/anthology/P02-1040.pdf

	Example:
		>>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
		...               'ensures', 'that', 'the', 'military', 'always',
		...               'obeys', 'the', 'commands', 'of', 'the', 'party']

		>>> hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
		...               'forever', 'hearing', 'the', 'activity', 'guidebook',
		...               'that', 'party', 'direct']

		>>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
		...               'ensures', 'that', 'the', 'military', 'will', 'forever',
		...               'heed', 'Party', 'commands']

		>>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
		...               'guarantees', 'the', 'military', 'forces', 'always',
		...               'being', 'under', 'the', 'command', 'of', 'the',
		...               'Party']

		>>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
		...               'army', 'always', 'to', 'heed', 'the', 'directions',
		...               'of', 'the', 'party']

		>>> bleu1([reference1, reference2, reference3], hypothesis1)
	"""
	return sentence_bleu(references, hypothesis, (1, 0, 0, 0))
