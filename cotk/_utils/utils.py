r"""
``cotk._utils`` is a function lib for internal use.
"""

import os
from typing import List, Any, Tuple
from itertools import chain

def trim_before_target(lists, target):
	'''Trim the list before the target. If there is no target,
	return the origin list.

	Arguments:
		lists (list)
		target

	'''
	try:
		lists = lists[:lists.index(target)]
	except ValueError:
		pass
	return lists

def chain_sessions(sessions: List[List[Any]]) -> Tuple[List[Any], List[int]]:
	chained_sessions = list(chain(*sessions))
	session_lengths = [len(session) for session in sessions]
	return chained_sessions, session_lengths

def restore_sessions(chained_sessions: List[Any], session_lengths: List[int]) -> List[List[Any]]:
	sessions: List[List[Any]] = []
	last = 0
	for session_length in session_lengths:
		sessions.append(chained_sessions[last: last + session_length])
		last += session_length
	return sessions

def replace_unk(sentences: List[Any], unk_token, target_token: Any = -1):
	r'''Auxiliary function for replacing the unknown words to another words

	Arguments:
		input (list[List[Any]]): the sentences
		unk_tokens (Any): id for unknown words.
		target: the target word index used to replace the unknown words.

	Returns:
		* list: processed result.
	'''
	return [[target_token if token == unk_token else token for token in sentence] for sentence in sentences]

def is_build_private_docs():
	return os.environ.get('COTK_DOCS_TYPE', None) == 'private'
