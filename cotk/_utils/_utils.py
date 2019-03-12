r"""
``cotk._utils`` is a function lib for internal use.
"""

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
