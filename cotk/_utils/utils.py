r"""
``cotk._utils`` is a function lib for internal use.
"""

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
