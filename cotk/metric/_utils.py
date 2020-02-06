
def _replace_unk(_input, _unk_id, _target=-1):
	r'''Auxiliary function for replacing the unknown words:

	Arguments:
		_input (list): the references or hypothesis.
		_unk_id (int): id for unknown words.
		_target: the target word index used to replace the unknown words.

	Returns:

		* list: processed result.
	'''
	output = []
	for _list in _input:
		_output = []
		for ele in _list:
			_output.append(_target if ele == _unk_id else ele)
		output.append(_output)
	return output