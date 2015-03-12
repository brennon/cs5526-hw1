import re, util, pickle

class MarkovGenerator():
	def __init__(self, textfile=None, history=2):
		self.file_path = textfile
		self.history = history

		self.words = None
		self.ngrams = None
		self.starting_tuples = None

		if textfile is not None:
			self.word_list = self.read_file()
			self.tuples, self.starting_tuples = self.parse_words(self.word_list, self.history)

	def read_file(self):
		"""
			Returns a list of words in the file passed to the constructor.

			>>> m = MarkovGenerator('./test.txt', 3)
			>>> words = m.read_file()
			>>> words
			['This', 'file', 'is', 'a', 'test', 'file.', 'This', 'file', 'contains', 'a', 'couple', 'of', 'sentences.', 'This', 'file', 'contains', 'nothing', 'more', 'than', 'this!']
		"""
		f = open(self.file_path)
		contents = f.read()
		f.close()
		return contents.split()

	def parse_words(self, word_list, history):
		"""
			Returns a dictionary. The keys are tuples of an n-1-grams found in the input file.
			Values are Counters of the final words in the n-grams with. The keys for these
			counters are the final word, and the values are the counts for the respective
			n-gram.

			>>> m = MarkovGenerator('./test.txt', 2)
			>>> words = m.read_file()
			>>> tuples, starters = m.parse_words(words, 2)
			>>> ('This', 'file') in tuples.keys()
			True
			>>> tuples[('This', 'file')]
			{'is': 1, 'contains': 2}
			>>> len(tuples.keys())
			15
			>>> starters
			{('This', 'file'): 3}
		"""
		if len(word_list) < history:
			return

		tuples = dict()

		first_prefix = tuple(word_list[0:history])

		sentence_starters = util.Counter()

		sentence_ended = True

		for i in range(len(word_list) - history):
			prefix = tuple(word_list[i:i + history])
			suffix = word_list[i + history]
			
			if prefix not in tuples.keys():
				tuples[prefix] = util.Counter()

			tuples[prefix][suffix] += 1

			if sentence_ended:
				sentence_starters[prefix] += 1
				sentence_ended = False
			elif self.ends_with_punctuation(prefix[0]):
				sentence_ended = True

		return tuples, sentence_starters

	def ends_with_punctuation(self, word):
		"""
			Returns True if word ends with !, ., ?, or any of these followed by a ", or '.

			>>> m = MarkovGenerator('./test.txt', 3)
			>>> m.ends_with_punctuation('no')
			False
			>>> m.ends_with_punctuation('end.')
			True
			>>> m.ends_with_punctuation('end...')
			True
			>>> m.ends_with_punctuation('end.zone')
			False
			>>> m.ends_with_punctuation('end!')
			True
			>>> m.ends_with_punctuation('end?')
			True
			>>> m.ends_with_punctuation('end."')
			True
			>>> m.ends_with_punctuation("end.'")
			True
		"""
		if re.search("[\.\?\!]{1}[\'\"]?$", word):
			return True
		else:
			return False


	def choose_start_tuple(self, starting_tuples):
		"""
			Returns a start tuple randomly chosen according to the distribution of starting
			tuples.

			>>> m = MarkovGenerator('./test.txt', 3)
			>>> words = m.read_file()
			>>> tuples, starters = m.parse_words(words, 2)
			>>> m.choose_start_tuple(starters)
			('This', 'file')
		"""
		return util.sample(starting_tuples)
	
	def choose_next_word(self, prefix, ngrams):
		"""
			Returns a word to follow the supplied prefix randomly chosen according to 
			the distributions specified in ngrams.

			>>> m = MarkovGenerator('./test.txt', 3)
			>>> words = m.read_file()
			>>> tuples, starters = m.parse_words(words, 2)
			>>> m.choose_next_word(('contains', 'a'), tuples)
			'couple'
		"""
		try:
			sample = util.sample(self.ngrams[prefix])
			return sample
		except:
			return None

	def generate_text(self, length):
		"""
			Generates random text from the sample text, length words long.
		"""

		if self.ngrams is None or self.starting_tuples is None:
			self.words = self.read_file()
			self.ngrams, self.starting_tuples = self.parse_words(self.words, self.history)

		generated_list = list(self.choose_start_tuple(self.starting_tuples))

		for i in range(self.history, length):
			new_prefix = tuple(generated_list[-self.history:])
			next_word = self.choose_next_word(new_prefix, self.ngrams)
			if next_word is None:
				new_start = self.choose_start_tuple(self.starting_tuples)[0]
				generated_list.append(new_start)
			else:
				generated_list.append(next_word)

		return ' '.join(generated_list)

	def serialize(self, filename):
		self.words = self.read_file()
		self.ngrams, self.starting_tuples = self.parse_words(self.words, self.history)
		to_pickle = dict()
		to_pickle['ngrams'] = self.ngrams
		to_pickle['starting_tuples'] = self.starting_tuples
		to_pickle['history'] = self.history
		pickle_file = open(filename, 'wb')
		pickle.dump(to_pickle, pickle_file)

	@classmethod
	def deserialize(cls, filename, history):
		pickle_file = open(filename, 'rb')
		depickled = pickle.load(pickle_file)
		pickle_file.close()
		m = cls(None, 3)
		m.ngrams = depickled['ngrams']
		m.starting_tuples = depickled['starting_tuples']
		m.history = depickled['history']
		return m

if __name__ == '__main__':
	# import doctest
	# doctest.testmod()
	generator = MarkovGenerator('./austen.txt', 3)
	# generator = MarkovGenerator.deserialize('./austen-3.pickle', 3)
	generator.serialize('./austen-3.pickle')
	text = generator.generate_text(500)
	print text