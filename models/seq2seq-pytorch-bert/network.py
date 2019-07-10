# coding:utf-8
import logging

import torch
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

from utils import zeros, LongTensor,\
			BaseNetwork, MyGRU, Storage, gumbel_max, flattenSequence, SingleAttnGRU, SequenceBatchNorm

# pylint: disable=W0221
class Network(BaseNetwork):
	def __init__(self, param):
		super().__init__(param)

		self.embLayer = EmbeddingLayer(param)
		self.bertEncoder = BERTEncoder(param)
		self.connectLayer = ConnectLayer(param)
		self.genNetwork = GenNetwork(param)

	def forward(self, incoming):
		incoming.result = Storage()

		self.embLayer.forward(incoming)
		self.bertEncoder.forward(incoming)
		self.connectLayer.forward(incoming)
		self.genNetwork.forward(incoming)

		incoming.result.loss = incoming.result.word_loss

		if torch.isnan(incoming.result.loss).detach().cpu().numpy() > 0:
			logging.info("Nan detected")
			logging.info(incoming.result)
			raise FloatingPointError("Nan detected")

	def detail_forward(self, incoming):
		incoming.result = Storage()

		self.embLayer.forward(incoming)
		self.bertEncoder.forward(incoming)
		self.connectLayer.forward(incoming)
		self.genNetwork.detail_forward(incoming)

class EmbeddingLayer(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param
		volatile = param.volatile

		self.embLayer = nn.Embedding(volatile.dm.vocab_size, args.embedding_size)
		self.embLayer.weight = nn.Parameter(torch.Tensor(volatile.wordvec))

	def forward(self, incoming):
		'''
		inp: data
		output: post
		'''
		incoming.resp = Storage()
		incoming.resp.embedding = self.embLayer(incoming.data.resp)
		incoming.resp.embLayer = self.embLayer

class BERTEncoder(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param

		self.bert_exclude = BertModel.from_pretrained(args.bert_model)
		self.drop = nn.Dropout(args.droprate)

	def forward(self, incoming):
		incoming.hidden = hidden = Storage()
		with torch.no_grad():
			h, _ = self.bert_exclude(incoming.data.post_bert)
		hidden.h = h[-1] # [length, batch, hidden]
		hidden.h_n = self.drop(hidden.h[0])
		hidden.h = self.drop(hidden.h)

class ConnectLayer(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param
		self.initLinearLayer = nn.Linear(args.eh_size * 2, args.dh_size)

	def forward(self, incoming):
		incoming.conn = conn = Storage()
		conn.init_h = self.initLinearLayer(incoming.hidden.h_n)

class GenNetwork(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param

		self.GRULayer = SingleAttnGRU(args.embedding_size, args.dh_size, args.eh_size * 2, initpara=False)
		self.wLinearLayer = nn.Linear(args.dh_size + args.eh_size * 2, param.volatile.dm.vocab_size)
		self.lossCE = nn.CrossEntropyLoss(ignore_index=param.volatile.dm.unk_id)
		self.start_generate_id = param.volatile.dm.go_id

		self.drop = nn.Dropout(args.droprate)

	def teacherForcing(self, inp, gen):
		embedding = inp.embedding
		length = inp.resp_length
		embedding = self.drop(embedding)
		_, gen.h, _ = self.GRULayer.forward(embedding, length-1, inp.post, inp.post_length, h_init=inp.init_h)
		gen.h = torch.stack(gen.h, dim=0)
		gen.h = self.drop(gen.h)
		gen.w = self.wLinearLayer(gen.h)


	def freerun(self, inp, gen):
		#mode: beam = beamsearch; max = choose max; sample = random_sampling; sample10 = sample from max 10

		def wLinearLayerCallback(gru_h):
			gru_h = self.drop(gru_h)
			w = self.wLinearLayer(gru_h)
			return w

		def input_callback(i, now):
			return self.drop(now)

		if self.args.decode_mode == "beam":
			new_gen = self.GRULayer.beamsearch(inp, self.args.top_k, wLinearLayerCallback, \
				input_callback=input_callback, no_unk=True, length_penalty=self.args.length_penalty)
			w_o = []
			length = []
			for i in range(inp.batch_size):
				w_o.append(new_gen.w_o[:, i, 0])
				length.append(new_gen.length[i][0])
			gen.w_o = torch.stack(w_o).transpose(0, 1)
			gen.length = length

		else:
			new_gen = self.GRULayer.freerun(inp, wLinearLayerCallback, self.args.decode_mode, \
				input_callback=input_callback, no_unk=True, top_k=self.args.top_k)
			gen.w_o = new_gen.w_o
			gen.length = new_gen.length

	def forward(self, incoming):
		inp = Storage()
		inp.resp_length = incoming.data.resp_length
		inp.embedding = incoming.resp.embedding
		inp.post = incoming.hidden.h
		inp.post_length = incoming.data.post_length
		inp.init_h = incoming.conn.init_h

		incoming.gen = gen = Storage()
		self.teacherForcing(inp, gen)

		w_o_f = flattenSequence(gen.w, incoming.data.resp_length-1)
		data_f = flattenSequence(incoming.data.resp[1:], incoming.data.resp_length-1)
		incoming.result.word_loss = self.lossCE(w_o_f, data_f)
		incoming.result.perplexity = torch.exp(incoming.result.word_loss)

	def detail_forward(self, incoming):
		inp = Storage()
		batch_size = inp.batch_size = incoming.data.batch_size
		inp.init_h = incoming.conn.init_h
		inp.post = incoming.hidden.h
		inp.post_length = incoming.data.post_length
		inp.embLayer = incoming.resp.embLayer
		inp.dm = self.param.volatile.dm
		inp.max_sent_length = self.args.max_sent_length

		incoming.gen = gen = Storage()
		self.freerun(inp, gen)

		dm = self.param.volatile.dm
		w_o = gen.w_o.detach().cpu().numpy()
		incoming.result.resp_str = resp_str = \
				[" ".join(dm.convert_ids_to_tokens(w_o[:, i].tolist())) for i in range(batch_size)]
		incoming.result.golden_str = golden_str = \
				[" ".join(dm.convert_ids_to_tokens(incoming.data.resp[:, i].detach().cpu().numpy().tolist()))\
				for i in range(batch_size)]
		incoming.result.post_str = post_str = \
				[" ".join(dm.convert_bert_ids_to_tokens(incoming.data.post_bert[:, i].detach().cpu().numpy().tolist()))\
				for i in range(batch_size)]
		incoming.result.show_str = "\n".join(["post: " + a + "\n" + "resp: " + b + "\n" + \
				"golden: " + c + "\n" \
				for a, b, c in zip(post_str, resp_str, golden_str)])
