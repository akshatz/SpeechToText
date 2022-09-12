#!/usr/bin/env python

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import soundfile
import codecs
import fnmatch
import pandas
import tqdm
import tarfile
import unicodedata

from sox import Transformer
from urllib3 import request 

def _maybe_download(fname, data_dir, data_url):
  data_path = os.path.join(data_dir, fname)
  if not os.path.exists(data_path):
    print("Can't find '{}'. Downloading...".format(data_path))
    request.urlretrieve(data_url, filename=data_path + '.tmp')
    os.rename(data_path + '.tmp', data_path)
  else:
    print("Skipping file '{}'".format(data_path))
  return data_path

def _download_and_preprocess_data(data_dir):
  # Conditionally download data to data_dir
  # print("Downloading Librivox data set (55GB) into {} if not already present...".format(data_dir))

  with tqdm.tqdm(total=7) as bar:

    DEV_CLEAN_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"

    def filename_of(x): return os.path.split(x)[1]
	# print(filename_of(x))
    dev_clean = _maybe_download(filename_of(DEV_CLEAN_URL), data_dir, DEV_CLEAN_URL)
    bar.update(1)

  # Conditionally extract LibriSpeech data
  # We extract each archive into data_dir, but test for existence in
  # data_dir/LibriSpeech because the archives share that root.
  # import pdb; pdb.set_trace()
  print("Extracting librivox data if not already extracted...")
  with tqdm.tqdm(total=7) as bar:
    LIBRIVOX_DIR = "LibriSpeech"
    work_dir = os.path.join(data_dir, LIBRIVOX_DIR)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
    bar.update(1)

  '''
  Convert FLAC data to wav, from:
  data_dir/LibriSpeech/split/1/2/1-2-3.flac
  to:
  data_dir/LibriSpeech/split-wav/1-2-3.wav
  
  And split LibriSpeech transcriptions, from:
  data_dir/LibriSpeech/split/1/2/1-2.trans.txt
  to:
  data_dir/LibriSpeech/split-wav/1-2-0.txt
  data_dir/LibriSpeech/split-wav/1-2-1.txt
  data_dir/LibriSpeech/split-wav/1-2-2.txt
  ...
  '''
  print("Converting FLAC to WAV and splitting transcriptions...")
  with tqdm.tqdm(total=7) as bar:

    dev_clean = _convert_audio_and_split_sentences(work_dir, "dev-clean", "dev-clean-wav")
    bar.update(1)
  
  # Write sets to disk as TSV files

  csv_dri = "librispeech_CSV_files/clean"
  if not os.path.exists(csv_dri):
        os.mkdir(csv_dri)

  dev_clean.to_csv(os.path.join(csv_dri, "dev-clean.tsv"), index=False, sep='\t')

def _maybe_extract(data_dir, extracted_data, archive):
  # If data_dir/extracted_data does not exist, extract archive in data_dir
 # if not gfile.Exists(os.path.join(data_dir, extracted_data)):
  if not os.path.exists(os.path.join(data_dir, extracted_data)):
    tar = tarfile.open(archive)
    tar.extractall(data_dir)
    tar.close()

def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir):
	source_dir = os.path.join(extracted_dir, data_set)
	target_dirName = "librispeech_unified"
	if not os.path.exists(target_dirName):
			os.mkdir(target_dirName)

	target_dir = os.path.join(target_dirName, dest_dir)

	if not os.path.exists(target_dir):
		os.makedirs(target_dir)

	'''                                                     
	Loop over transcription files and split each one

	The format for each file 1-2.trans.txt is:
	1-2-0 transcription of 1-2-0.flac
	1-2-1 transcription of 1-2-1.flac
	...
	
	Each file is then split into several files:
	1-2-0.txt (contains transcription of 1-2-0.flac)
	1-2-1.txt (contains transcription of 1-2-1.flac)
	...
	
	We also convert the corresponding FLACs to WAV in the same pass
	'''
	files = []
	for root, dirnames, filenames in os.walk(source_dir):
		for filename in fnmatch.filter(filenames, '*.trans.txt'):
			trans_filename = os.path.join(root, filename)
			with codecs.open(trans_filename, "r", "utf-8") as fin:
				for line in fin:
					# Parse each segment line

					first_space = line.find(" ") # position of first space
					# print(first_space)
					seqid, transcript = line[:first_space], line[first_space+1:]

					# We need to do the encode-decode dance here because encode
					# returns a bytes() object on Python 3, and text_to_char_array
					# expects a string.
					# import pdb; pdb.set_trace()
					transcript1 = unicodedata.normalize("NFD", transcript) \
								.encode("ascii", "ignore")   \
								.decode("ascii", "ignore")

					transcript = transcript1.lower().strip()
					
					# Convert corresponding FLAC to a WAV
					flac_file = os.path.join(root, seqid + ".flac")
					wav_file = os.path.join(target_dir, seqid + ".wav")
					if not os.path.exists(wav_file):
						Transformer().build(flac_file, wav_file)
					wav_filesize = os.path.getsize(wav_file)
					audio_data, samplerate = soundfile.read(wav_file)
					duration =  round((float(len(audio_data)) / samplerate),0)
					details = soundfile.SoundFile(wav_file)
					
					files.append((wav_file,transcript1, transcript,duration, wav_filesize, ((details.format).lower()) ,details.samplerate, details.channels,((details.subtype).split("_")[0]).lower(), ((details.subtype).split("_")[1]), "librispeech",str(data_set)))
  # print((wav_file,transcript1, transcript,duration, wav_filesize, ((details.format).lower()) ,details.samplerate, details.channels,((details.subtype).split("_")[0]).lower(), ((details.subtype).split("_")[1]), "librispeech",str(data_set))
	return pandas.DataFrame(data=files, columns=["path", "transcript_raw", "transcript_transformed", "duration",  "file_size(in bytes)", "format", "sample_rate", "n_channels", "encoding", "bitrate", "corpus" , "corpus_category"])

# if __name__ == "__main__":
_download_and_preprocess_data(sys.argv[1])
