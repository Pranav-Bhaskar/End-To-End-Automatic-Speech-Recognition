import numpy as np
from scipy.io import wavfile

def data_generator(data, params, mode='train'):
	def generator():
		if mode == 'train':
			np.random.shuffle(data)
		# Feel free to add any augmentation
		for (label_id, uid, fname) in data:
			try:
				_, wav = wavfile.read(fname)
				wav = wav.astype(np.float32) / np.iinfo(np.int16).max

				L = 16000  # be aware, some files are shorter than 1 sec!
				if len(wav) < L:
					continue
				# let's generate more silence!
				samples_per_file = 1 if label_id != name2id['silence'] else 20
				for _ in range(samples_per_file):
					if len(wav) > L:
						beg = np.random.randint(0, len(wav) - L)
					else:
						beg = 0
					yield dict(
					target=np.int32(label_id),
					wav=wav[beg: beg + L],
					)
			except Exception as err:
				print(err, label_id, uid, fname)

		return generator
