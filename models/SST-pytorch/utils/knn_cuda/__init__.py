import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


def load_cpp_ext(ext_name):
	root_dir = os.path.join(os.path.split(__file__)[0])
	ext_csrc = os.path.join(root_dir, "csrc")
	ext_path = os.path.join(ext_csrc, "_ext", ext_name)
	os.makedirs(ext_path, exist_ok=True)
	assert torch.cuda.is_available(), "torch.cuda.is_available() is False."
	ext_sources = [
		os.path.join(ext_csrc, "cuda", "{}.cpp".format(ext_name)),
		os.path.join(ext_csrc, "cuda", "{}.cu".format(ext_name))
	]
	extra_cuda_cflags = [
		"-DCUDA_HAS_FP16=1",
		"-D__CUDA_NO_HALF_OPERATORS__",
		"-D__CUDA_NO_HALF_CONVERSIONS__",
		"-D__CUDA_NO_HALF2_OPERATORS__",
	]
	ext = load(
		name=ext_name,
		sources=ext_sources,
		extra_cflags=["-O2"],
		build_directory=ext_path,
		extra_cuda_cflags=extra_cuda_cflags,
		verbose=False,
		with_cuda=True
	)
	return ext


_knn = load_cpp_ext("knn")


def knn(ref, query, k):
	d, i = _knn.knn(ref, query, k)
	i -= 1
	return d, i


def _T(t, mode=False):
	if mode:
		return t.transpose(0, 1).contiguous()
	else:
		return t


class KNN(nn.Module):

	def __init__(self, k, transpose_mode=False):
		super(KNN, self).__init__()
		self.k = k
		self._t = transpose_mode

	def forward(self, ref, query, grad=True):
		ref, query = _T(ref, self._t), _T(query, self._t)
		with torch.no_grad():
			d, i = knn(ref.float(), query.float(), self.k)
			d, i = _T(d, self._t), _T(i, self._t)
		#if grad:
		#	assert self.k == 1
		#	pos = torch.index_select(ref, dim=1, index=i[:, 0])
		#	d = _T(torch.norm(query - pos, dim=0, keepdim=True), self._t)
		return d, i
