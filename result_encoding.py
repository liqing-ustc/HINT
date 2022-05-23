from utils import START, END, NULL

class ResultEncoding:
	def __init__(self, encoding='decimal', reverse=True):
		super().__init__()

		assert encoding in ['decimal', 'binary', 'sin']
		base = 2 if encoding == 'binary' else 10
		vocab = list(map(str, range(base))) + [START, END, NULL]

		self.reverse = reverse
		self.encoding = encoding
		self.base = base
		self.vocab = vocab
		self.start_idx = vocab.index(START)
		self.end_idx = vocab.index(END)
		self.null_idx = vocab.index(NULL)

	def res2seq(self, r):
		if self.encoding == 'decimal':
			s = list(f'{r:d}')
		elif self.encoding == 'binary':
			s = list(f'{r:b}')
		if self.reverse:
			s = s[::-1]
		s = list(map(self.vocab.index, s))
		return s

	def res2seq_batch(self, res, pad=True):
		seq = [self.res2seq(r) for r in res]
		seq = [[self.start_idx] + s + [self.end_idx] for s in seq]
		if pad:
			max_len = max([len(s) for s in seq])
			seq = [s + [self.null_idx]*(max_len - len(s)) for s in seq]
		return seq

	def seq2res(self, seq):
		res = []
		for x in seq:
			x = self.vocab[x]
			if x in [START, NULL, END]:
				break
			res.append(x)

		if self.reverse:
			res = res[::-1]
		res = int(''.join(res), base=self.base) if len(res) > 0 else -1
		return res
	
	def seq2res_batch(self, seq):
		return [self.seq2res(s) for s in seq]


if __name__ == '__main__':
	res_enc = ResultEncoding('decimal')
	v = 13
	s = res_enc.res2seq(v)
	u = res_enc.seq2res(s)
	print(v, s, u)

	res_enc = ResultEncoding('binary')
	v = 13
	s = res_enc.res2seq(v)
	u = res_enc.seq2res(s)
	print(v, s, u)
