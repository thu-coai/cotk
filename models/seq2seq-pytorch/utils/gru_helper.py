#coding: utf-8
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .cuda_helper import zeros, Tensor, LongTensor, cuda
from .gumbel import gumbel_max, gumbel_max_with_mask
from .storage import Storage

F_GRUCell = torch._C._VariableFunctions.gru_cell

def sortSequence(data, length):
	shape = data.shape
	len, fsize = shape[0], shape[-1]
	data = data.reshape(len, -1, fsize)
	batch_size = data.shape[1]
	length = length.reshape(-1)

	zero_num = np.sum(length == 0)
	memo = list(reversed(np.argsort(length).tolist()))[:batch_size-zero_num]
	res = zeros(data.shape[0], batch_size - zero_num, data.shape[-1])
	for i, idx in enumerate(memo):
		res[:, i, :] = data[:, idx, :]
	return res, sorted(length, reverse=True)[: batch_size - zero_num], (shape, memo, zero_num)

def sortSequenceByMemo(data, memo):
	data = data.reshape(-1, data.shape[-1])
	batch_size = data.shape[0]
	shape, memo, zero_num = memo
	res = zeros(batch_size - zero_num, data.shape[-1])
	for i, idx in enumerate(memo):
		res[i, :] = data[idx, :]
	return res

def revertSequence(data, memo, isseq=False):
	shape, memo, zero_num = memo
	if isseq:
		res = zeros(data.shape[0], data.shape[1]+zero_num, data.shape[2])
		for i, idx in enumerate(memo):
			res[:, idx, :] = data[:, i, :]
		return res.reshape(*((res.shape[0], )+shape[1:-1]+(res.shape[-1], )))
	else:
		res = zeros(data.shape[0]+zero_num, data.shape[1])
		for i, idx in enumerate(memo):
			res[idx, :] = data[i, :]
		return res.reshape(*(shape[1:-1]+(res.shape[-1], )))

def flattenSequence(data, length):
	arr = []
	for i in range(length.size):
		arr.append(data[0:length[i], i])
	return torch.cat(arr, dim=0)

def copySequence(data, length): # for BOW loss
	arr = []
	for i in range(length.size):
		arr.append(data[i].repeat(length[i], 1))
	return torch.cat(arr, dim=0)

def generateMask(seqlen, length, type=int):
	return Tensor(
		(np.expand_dims(np.arange(seqlen), 1) < np.expand_dims(length, 0)).astype(type))

def maskedSoftmax(data, length):
	mask = generateMask(data.shape[0], length)
	return data.masked_fill(mask == 0, -1e9).softmax(dim=0)

def maskedLogSoftmax(data, length):
	mask = generateMask(data.shape[0], length)
	return torch.log_softmax(data.masked_fill(mask == 0, -1e9), dim=0)

class MyGRU(nn.Module):
	def __init__(self, input_size, hidden_size, layers=1, bidirectional=False, initpara=True):
		super(MyGRU, self).__init__()

		self.input_size, self.hidden_size, self.layers, self.bidirectional = \
				input_size, hidden_size, layers, bidirectional
		self.GRU = GRU(input_size, hidden_size, layers, bidirectional=bidirectional)
		self.initpara = initpara
		if initpara:
			if bidirectional:
				self.h_init = Parameter(torch.Tensor(2 * layers, 1, hidden_size))
			else:
				self.h_init = Parameter(torch.Tensor(layers, 1, hidden_size))
		self.reset_parameters()

	def reset_parameters(self):
		if self.initpara:
			stdv = 1.0 / math.sqrt(self.hidden_size)
			self.h_init.data.uniform_(-stdv, stdv)

	def getInitialParameter(self, batch_size):
		return self.h_init.repeat(1, batch_size, 1)

	def forward(self, incoming, length, h_init=None, need_h=False):
		sen_sorted, length_sorted, memo = sortSequence(incoming, length)
		left_batch_size = sen_sorted.shape[-2]
		sen_packed = pack_padded_sequence(sen_sorted, length_sorted)
		if h_init is None:
			h_init = self.getInitialParameter(left_batch_size)
		else:
			h_init = torch.unsqueeze(sortSequenceByMemo(h_init, memo), 0)
		h, h_n = self.GRU(sen_packed, h_init)
		h_n = h_n.transpose(0, 1).reshape(left_batch_size, -1)
		h_n = revertSequence(h_n, memo)
		if need_h:
			h = pad_packed_sequence(h)[0]
			h = revertSequence(h, memo, True)
			return h_n, h
		else:
			return h_n, None

class DecoderRNN(nn.Module):
	def __init__(self):
		super().__init__()

	def _freerun(self, inp, nextStep, wLinearLayerCallback, mode='max', input_callback=None, no_unk=True, top_k=10):
		# inp contains: batch_size, dm, embLayer, max_sent_length, [init_h]
		# input_callback(i, embedding):   if you want to change word embedding at pos i, override this function
		# nextStep(embedding, flag):  pass embedding to RNN and get gru_h, flag indicates i th sentence is end when flag[i]==1
		# wLinearLayerCallback(gru_h): input gru_h and give a probability distribution on vocablist

		# output: w_o emb length

		start_id = inp.dm.go_id if no_unk else 0

		batch_size = inp.batch_size
		dm = inp.dm

		first_emb = inp.embLayer(LongTensor([dm.go_id])).repeat(batch_size, 1)

		gen = Storage()
		gen.w_pro = []
		gen.w_o = []
		gen.emb = []
		flag = zeros(batch_size).byte()
		EOSmet = []

		next_emb = first_emb
		#nextStep = self.init_forward(batch_size, inp.get("init_h", None))

		for i in range(inp.max_sent_length):
			now = next_emb
			if input_callback:
				now = input_callback(i, now)

			gru_h = nextStep(now, flag)
			if isinstance(gru_h, tuple):
				gru_h = gru_h[0]

			w = wLinearLayerCallback(gru_h)
			gen.w_pro.append(w.softmax(dim=-1))
			#TODO: didn't consider copynet
			if mode == "max":
				w = torch.argmax(w[:, start_id:], dim=1) + start_id
				next_emb = inp.embLayer(w)
			elif mode == "gumbel" or mode == "sample":
				w_onehot = gumbel_max(w[:, start_id:])
				w = torch.argmax(w_onehot, dim=1) + start_id
				next_emb = torch.sum(torch.unsqueeze(w_onehot, -1) * inp.embLayer.weight[start_id:], 1)
			elif mode == "samplek":
				_, index = w[:, start_id:].topk(top_k, dim=-1, largest=True, sorted=True) # batch_size, top_k
				mask = torch.zeros_like(w[:, start_id:]).scatter_(-1, index, 1.0)
				w_onehot = gumbel_max_with_mask(w[:, start_id:], mask)
				w = torch.argmax(w_onehot, dim=1) + start_id
				next_emb = torch.sum(torch.unsqueeze(w_onehot, -1) * inp.embLayer.weight[start_id:], 1)

			gen.w_o.append(w)
			gen.emb.append(next_emb)

			EOSmet.append(flag)
			flag = flag | (w == dm.eos_id)
			if torch.sum(flag).detach().cpu().numpy() == batch_size:
				break

		EOSmet = 1-torch.stack(EOSmet)
		gen.w_o = torch.stack(gen.w_o) * EOSmet.long()
		gen.emb = torch.stack(gen.emb) * EOSmet.float().unsqueeze(-1)
		gen.length = torch.sum(EOSmet, 0).detach().cpu().numpy()

		return gen

	def _beamsearch(self, inp, top_k, nextStep, wLinearLayerCallback, input_callback=None, no_unk=True, length_penalty=0.7):
		# inp contains: batch_size, dm, embLayer, max_sent_length, [init_h]
		# input_callback(i, embedding):   if you want to change word embedding at pos i, override this function
		# nextStep(embedding, flag):  pass embedding to RNN and get gru_h, flag indicates i th sentence is end when flag[i]==1
		# wLinearLayerCallback(gru_h): input gru_h and give logits on vocablist

		# output: w_o emb length

		#start_id = inp.dm.go_id if no_unk else 0

		batch_size = inp.batch_size
		dm = inp.dm

		first_emb = inp.embLayer(LongTensor([dm.go_id])).repeat(batch_size, top_k, 1)

		w_pro = []
		w_o = []
		emb = []
		flag = zeros(batch_size, top_k).byte()
		EOSmet = []
		score = zeros(batch_size, top_k)
		score[:, 1:] = -1e9
		now_length = zeros(batch_size, top_k)
		back_index = []
		regroup = LongTensor([i for i in range(top_k)]).repeat(batch_size, 1)

		next_emb = first_emb
		#nextStep = self.init_forward(batch_size, inp.get("init_h", None))

		for i in range(inp.max_sent_length):
			now = next_emb
			if input_callback:
				now = input_callback(i, now)

			gru_h = nextStep(now, flag, regroup=regroup) # batch_size, top_k, hidden_size
			if isinstance(gru_h, tuple):
				gru_h = gru_h[0]

			w = wLinearLayerCallback(gru_h) # batch_size, top_k, vocab_size

			if no_unk:
				w[:, :, dm.unk_id] = -1e9
			w = w.log_softmax(dim=-1)
			w_pro.append(w.exp())

			new_score = (score.unsqueeze(-1) + w * (1-flag.float()).unsqueeze(-1)) / ((now_length.float() + 1 - flag.float()).unsqueeze(-1) ** length_penalty)
			new_score[:, :, 1:] = new_score[:, :, 1:] - flag.float().unsqueeze(-1) * 1e9
			_, index = new_score.reshape(batch_size, -1).topk(top_k, dim=-1, largest=True, sorted=True) # batch_size, top_k

			new_score = (score.unsqueeze(-1) + w * (1-flag.float()).unsqueeze(-1)).reshape(batch_size, -1)
			# assert (regroup >= new_score.shape[1]).sum().tolist() == 0
			score = torch.gather(new_score, dim=1, index=index)

			vocab_size = w.shape[-1]
			regroup = index / vocab_size # batch_size, top_k
			back_index.append(regroup)
			w = torch.fmod(index, vocab_size) # batch_size, top_k

			# assert (regroup >= flag.shape[1]).sum().tolist() == 0
			flag = torch.gather(flag, dim=1, index=regroup)
			# assert (regroup >= now_length.shape[1]).sum().tolist() == 0
			now_length = torch.gather(now_length, dim=1, index=regroup) + 1 - flag.float()

			w_x = w.clone()
			w_x[w_x >= dm.vocab_size] = dm.unk_id
			#w_x = cuda(w_x)

			next_emb = inp.embLayer(w_x)
			w_o.append(w)
			emb.append(next_emb)

			EOSmet.append(flag)

			flag = flag | (w == dm.eos_id)
			if torch.sum(flag).detach().cpu().numpy() == batch_size * top_k:
				break

		#back tracking
		gen = Storage()
		back_EOSmet = []
		gen.w_o = []
		gen.emb = []
		now_index = LongTensor([i for i in range(top_k)]).repeat(batch_size, 1)

		for i, index in reversed(list(enumerate(back_index))):
			gen.w_o.append(torch.gather(w_o[i], dim=1, index=now_index))
			gen.emb.append(torch.gather(emb[i], dim=1, index=now_index.unsqueeze(-1).expand_as(emb[i])))
			back_EOSmet.append(torch.gather(EOSmet[i], dim=1, index=now_index))
			now_index = torch.gather(index, dim=1, index=now_index)

		back_EOSmet = 1-torch.stack(list(reversed(back_EOSmet)))
		gen.w_o = torch.stack(list(reversed(gen.w_o))) * back_EOSmet.long()
		gen.emb = torch.stack(list(reversed(gen.emb))) * back_EOSmet.float().unsqueeze(-1)
		gen.length = torch.sum(back_EOSmet, 0).detach().cpu().numpy()

		return gen


class SingleGRU(DecoderRNN):
	def __init__(self, input_size, hidden_size, initpara=True):
		super().__init__()

		self.input_size, self.hidden_size = input_size, hidden_size
		self.GRU = GRU(input_size, hidden_size, 1)
		self.initpara = initpara
		if initpara:
			self.h_init = Parameter(torch.Tensor(1, 1, hidden_size))
			stdv = 1.0 / math.sqrt(self.hidden_size)
			self.h_init.data.uniform_(-stdv, stdv)

	def getInitialParameter(self, batch_size):
		return self.h_init.repeat(1, batch_size, 1)

	def forward(self, incoming, length, h_init=None, need_h=False):
		sen_sorted, length_sorted, memo = sortSequence(incoming, length)
		left_batch_size = sen_sorted.shape[-2]
		sen_packed = pack_padded_sequence(sen_sorted, length_sorted)
		if h_init is None:
			h_init = self.getInitialParameter(left_batch_size)
		else:
			h_init = torch.unsqueeze(sortSequenceByMemo(h_init, memo), 0)
		h, h_n = self.GRU(sen_packed, h_init)
		h_n = h_n.transpose(0, 1).reshape(left_batch_size, -1)
		h_n = revertSequence(h_n, memo)
		if need_h:
			h = pad_packed_sequence(h)[0]
			h = revertSequence(h, memo, True)
			return h_n, h
		else:
			return h_n, None

	def init_forward(self, batch_size, h_init=None):
		if h_init is None:
			h_init = self.getInitialParameter(batch_size)
		else:
			h_init = torch.unsqueeze(h_init, 0)
		h_history = h_init
		h = h_init[0]

		def nextStep(incoming, stopmask):
			nonlocal h_history, h
			h = self.cell_forward(incoming, h) * (1 - stopmask).float().unsqueeze(-1)
			return h

		return nextStep

	def cell_forward(self, incoming, h):
		return F_GRUCell( \
				incoming, h, \
				self.GRU.weight_ih_l0, self.GRU.weight_hh_l0, \
				self.GRU.bias_ih_l0, self.GRU.bias_hh_l0, \
		)

	def freerun(self, inp, wLinearLayerCallback, mode='max', input_callback=None, no_unk=True):
		nextStep = self.init_forward(inp.batch_size, inp.get("init_h", None))
		return self._freerun(inp, nextStep, wLinearLayerCallback, mode, input_callback, no_unk)

class SingleAttnGRU(DecoderRNN):
	def __init__(self, input_size, hidden_size, post_size, initpara=True, gru_input_attn=False):
		super().__init__()

		self.input_size, self.hidden_size, self.post_size = \
			input_size, hidden_size, post_size
		self.gru_input_attn = gru_input_attn

		if self.gru_input_attn:
			self.GRU = GRU(input_size + post_size, hidden_size, 1)
		else:
			self.GRU = GRU(input_size, hidden_size, 1)

		self.attn_query = nn.Linear(hidden_size, post_size)

		if initpara:
			self.h_init = Parameter(torch.Tensor(1, 1, hidden_size))
			stdv = 1.0 / math.sqrt(self.hidden_size)
			self.h_init.data.uniform_(-stdv, stdv)

	def getInitialParameter(self, batch_size):
		return self.h_init.repeat(1, batch_size, 1)

	def forward(self, incoming, length, post, post_length, h_init=None):
		batch_size = incoming.shape[1]
		seqlen = incoming.shape[0]
		if h_init is None:
			h_init = self.getInitialParameter(batch_size)
		else:
			h_init = torch.unsqueeze(h_init, 0)
		h_now = h_init[0]
		hs = []
		attn_weights = []
		context = zeros(batch_size, self.post_size)

		for i in range(seqlen):
			if self.gru_input_attn:
				h_now = self.cell_forward(torch.cat([incoming[i], context], last_dim=-1), h_now) \
					* Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)
			else:
				h_now = self.cell_forward(incoming[i], h_now) \
					* Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

			query = self.attn_query(h_now)
			attn_weight = maskedSoftmax((query.unsqueeze(0) * post).sum(-1), post_length)
			context = (attn_weight.unsqueeze(-1) * post).sum(0)

			hs.append(torch.cat([h_now, context], dim=-1))
			attn_weights.append(attn_weight)

		return h_now, hs, attn_weights

	def init_forward(self, batch_size, post, post_length, h_init=None):
		if h_init is None:
			h_init = self.getInitialParameter(batch_size)
		else:
			h_init = torch.unsqueeze(h_init, 0)
		h_now = h_init[0]
		context = zeros(batch_size, self.post_size)

		def nextStep(incoming, stopmask):
			nonlocal h_now, post, post_length, context

			if self.gru_input_attn:
				h_now = self.cell_forward(torch.cat([incoming, context], dim=-1), h_now) \
					* (1 - stopmask).float().unsqueeze(-1)
			else:
				h_now = self.cell_forward(incoming, h_now) * (1 - stopmask).float().unsqueeze(-1)

			query = self.attn_query(h_now)
			attn_weight = maskedSoftmax((query.unsqueeze(0) * post).sum(-1), post_length)
			context = (attn_weight.unsqueeze(-1) * post).sum(0)

			return torch.cat([h_now, context], dim=-1), attn_weight

		return nextStep

	def init_forward_3d(self, batch_size, top_k, post, post_length, h_init=None):
		if h_init is None:
			h_init = self.getInitialParameter(batch_size)
		else:
			h_init = torch.unsqueeze(h_init, 0)
		h_now = h_init[0].unsqueeze(1).expand(-1, top_k, -1) # batch_size * top_k * hidden_size
		context = zeros(batch_size, self.post_size)

		post = post.unsqueeze(-2)
		#post_length = np.tile(np.expand_dims(post_length, 1), (1, top_k, 1))

		def nextStep(incoming, stopmask, regroup=None):
			nonlocal h_now, post, post_length, context
			h_now = torch.gather(h_now, 1, regroup.unsqueeze(-1).repeat(1, 1, h_now.shape[-1]))

			if self.gru_input_attn:
				context = torch.gather(context, 1, regroup.unsqueeze(-1).repeat(1, 1, context.shape[-1]))
				h_now = self.cell_forward(torch.cat([incoming, context], dim=-1), h_now) \
					* (1 - stopmask).float().unsqueeze(-1)
			else:
				h_now = self.cell_forward(incoming, h_now) * (1 - stopmask).float().unsqueeze(-1) # batch_size * top_k * hidden_size

			query = self.attn_query(h_now) # batch_size * top_k * post_size

			mask = generateMask(post.shape[0], post_length).unsqueeze(-1)
			attn_weight = (query.unsqueeze(0) * post).sum(-1).masked_fill(mask==0, -1e9).softmax(0) # post_len * batch_size * top_k
			context = (attn_weight.unsqueeze(-1) * post).sum(0)

			return torch.cat([h_now, context], dim=-1), attn_weight

		return nextStep

	def cell_forward(self, incoming, h):
		shape = h.shape
		return F_GRUCell( \
				incoming.reshape(-1, incoming.shape[-1]), h.reshape(-1, h.shape[-1]), \
				self.GRU.weight_ih_l0, self.GRU.weight_hh_l0, \
				self.GRU.bias_ih_l0, self.GRU.bias_hh_l0, \
		).reshape(*shape)

	def freerun(self, inp, wLinearLayerCallback, mode='max', input_callback=None, no_unk=True, top_k=10):
		nextStep = self.init_forward(inp.batch_size, inp.post, inp.post_length, inp.get("init_h", None))
		return self._freerun(inp, nextStep, wLinearLayerCallback, mode, input_callback, no_unk, top_k=top_k)

	def beamsearch(self, inp, top_k, wLinearLayerCallback, input_callback=None, no_unk=True, length_penalty=0.7):
		nextStep = self.init_forward_3d(inp.batch_size, top_k, inp.post, inp.post_length, inp.get("init_h", None))
		return self._beamsearch(inp, top_k, nextStep, wLinearLayerCallback, input_callback, no_unk, length_penalty)


class SingleSelfAttnGRU(DecoderRNN):
	def __init__(self, input_size, hidden_size, attn_wait=3, initpara=True):
		super().__init__()

		self.input_size, self.hidden_size = \
				input_size, hidden_size
		self.attn_wait = attn_wait
		self.decoderGRU = GRU(input_size + hidden_size, hidden_size, 1)
		self.encoderGRU = GRU(input_size, hidden_size, 1)

		self.attn_query = nn.Linear(hidden_size, hidden_size)

		#self.attn_null = Parameter(torch.Tensor(1, 1, hidden_size))
		#stdv = 1.0 / math.sqrt(self.hidden_size)
		#self.attn_null.data.uniform_(-stdv, stdv)

		if initpara:
			self.eh_init = Parameter(torch.Tensor(1, 1, hidden_size))
			stdv = 1.0 / math.sqrt(self.hidden_size)
			self.eh_init.data.uniform_(-stdv, stdv)
			self.dh_init = Parameter(torch.Tensor(1, 1, hidden_size))
			self.dh_init.data.uniform_(-stdv, stdv)

	def forward(self, incoming, length, eh_init=None, dh_init=None, need_h=False, need_attn_weight=False):
		batch_size = incoming.shape[1]
		seqlen = incoming.shape[0]

		if eh_init is None:
			eh_init = self.eh_init.repeat(1, batch_size, 1)
		else:
			eh_init = torch.unsqueeze(eh_init, 0)
		if dh_init is None:
			dh_init = self.dh_init.repeat(1, batch_size, 1)
		else:
			dh_init = torch.unsqueeze(dh_init, 0)

		h_history = []
		eh = eh_init[0]
		dh = dh_init[0]
		dhs = []
		attn_weights = []
		#attn_null = self.attn_null.repeat(1, batch_size, 1)
		for i in range(seqlen):
			if i <= self.attn_wait:
				context = zeros(batch_size, self.hidden_size)
			else:
				query = self.attn_query(dh)
				h_wait = h_history[:self.attn_wait]
				attn_weight = (query.unsqueeze(0) * h_wait).sum(-1).softmax(0)
				attn_weights.append(attn_weight)
				context = (attn_weight.unsqueeze(-1) * h_wait).sum(0)
			dh = F_GRUCell(
				torch.cat([incoming[i], context], dim=-1), dh,
				self.decoderGRU.weight_ih_l0, self.decoderGRU.weight_hh_l0,
				self.decoderGRU.bias_ih_l0, self.decoderGRU.bias_hh_l0
			) * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

			eh = F_GRUCell(
				incoming[i], eh,
				self.encoderGRU.weight_ih_l0, self.encoderGRU.weight_hh_l0,
				self.encoderGRU.bias_ih_l0, self.encoderGRU.bias_hh_l0
			) * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

			h_history = eh.unsqueeze(0) if not h_history else torch.cat([h_history, eh.unsqueeze(0)], dim=0)
			dhs.append(dh)

		h_n = dh
		if need_h:
			h = torch.stack(dhs, 0)
			if need_attn_weight:
				return h, h_n, attn_weights
			else:
				return h, h_n
		else:
			return h_n

	def init_forward(self, batch_size, eh_init=None, dh_init=None):
		if eh_init is None:
			eh_init = self.eh_init.repeat(1, batch_size, 1)
		else:
			eh_init = torch.unsqueeze(eh_init, 0)
		if dh_init is None:
			dh_init = self.dh_init.repeat(1, batch_size, 1)
		else:
			dh_init = torch.unsqueeze(dh_init, 0)

		h_history = []
		dh = dh_init[0]
		eh = eh_init[0]
		#attn_null = self.attn_null.repeat(1, batch_size, 1)

		def nextStep(incoming, stopmask):
			nonlocal h_history, eh, dh

			if h_history is None or h_history.shape[0] <= self.attn_wait:
				context = zeros(batch_size, self.hidden_size)
			else:
				query = self.attn_query(dh)
				h_wait = h_history[:self.attn_wait]
				attn_weight = (query.unsqueeze(0) * h_wait).sum(-1).softmax(0)
				context = (attn_weight.unsqueeze(-1) * h_wait).sum(0)

			dh = F_GRUCell(
				torch.cat([incoming, context], dim=-1), dh,
				self.decoderGRU.weight_ih_l0, self.decoderGRU.weight_hh_l0,
				self.decoderGRU.bias_ih_l0, self.decoderGRU.bias_hh_l0
			) * (1 - stopmask).float().unsqueeze(-1)

			eh = F_GRUCell(
				incoming, eh,
				self.encoderGRU.weight_ih_l0, self.encoderGRU.weight_hh_l0,
				self.encoderGRU.bias_ih_l0, self.encoderGRU.bias_hh_l0
			) * (1 - stopmask).float().unsqueeze(-1)
			h_history = eh.unsqueeze(0) if not h_history else torch.cat([h_history, eh.unsqueeze(0)], dim=0)
			return dh

		return nextStep

	def freerun(self, inp, wLinearLayerCallback, mode='max', input_callback=None, no_unk=True):
		nextStep = self.init_forward(inp.batch_size, inp.get("eh_init", None), inp.get("dh_init", None))
		return self._freerun(inp, nextStep, wLinearLayerCallback, mode, input_callback, no_unk)
