"""

This code is for loading and processing the Twitch data for visual dilog models


"""
import os
import numpy as np
import json
import re
import random
import struct
import h5py
from keras.preprocessing import sequence




PAD_TOKEN = u'\u0c05' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = u'\u0c06' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = u'\u0c07' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = u'\u0c08' # This has a vocab id, which is used at the end of untruncated target sequences
SEGMENT_TOKEN = u'\u0c09'  # This has a vocab id, whic is used to differentiate multiple chats



class Vocab(object):
    def __init__(self,vocab_file,max_size):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
            Args:
                vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. 
                            This code doesn't actually use the frequencies, though.
                max_size: integer. The maximum size of the resulting Vocabulary.
                
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # [PAD], [START], [STOP] and [UNK] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, START_DECODING, STOP_DECODING, UNKNOWN_TOKEN,SEGMENT_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f.read().splitlines():
                pieces = line.split('\t')
                if len(pieces) != 2:
                    print 'Warning: incorrectly formatted line in vocabulary file: %s\n' % line
                    continue
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print "max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count)
                    break
        print "Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1])

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count



class Batch(object):

    def __init__(self):
        self._dict = {}


    def put(self,key,value):
        if self._dict.get(key) is None:
            self._dict[key] = value
        else:
            raise Exception("key:{} already exits".format(key))

    def get(self,key):
       if self._dict.get(key) is not None:
           return self._dict[key]
       else:
           raise Exception("key:{} already exits".format(key))


class Batcher(object):

    def __init__(self,hps,vocab):
        
        self._data_path = hps.data_path
        self._img_dim = hps.img_dim
        self._max_video_enc_steps = hps.max_video_enc_steps
        self._max_context_enc_steps = hps.max_context_enc_steps
        self._max_response_enc_steps = hps.max_response_enc_steps
        self._mode = hps.mode
        self._batch_size = hps.batch_size
        self._vocab = vocab
        self._word_level = hps.word_level
        self._negative_examples = hps.negative_examples
        self._loss_function = hps.loss_function

    def _reorder_list(self, data_list):
        # given [[a,b,c],[d,e,f]] convert to [a,d,b,e,c,f] ...
        reorded_list = []
        split_len = len(data_list[0])
        for i in range(split_len):
            for j in range(len(data_list)):
                reorded_list.append(data_list[j][i])

        return reorded_list

    def _process_data(self):
        """this module extracts data from videos and chat files and creates batches"""
        # load json data which contains all the text information
        if self._word_level:
            if self._loss_function == '3triplet' and self._mode == 'train':
                with open(os.path.join(self._data_path,self._mode+'_triplet_word.json'),'r') as f:
                    json_data = json.load(f)
            else:
                with open(os.path.join(self._data_path,self._mode+'_word.json'),'r') as f:
                    json_data = json.load(f)
        else:
            if self._loss_function == '3triplet':
                with open(os.path.join(self._data_path,self._mode+'_triplet_word.json'),'r') as f:
                    json_data = json.load(f)
            else:
                with open(os.path.join(self._data_path,self._mode+'_word.json'),'r') as f:
                    json_data = json.load(f)


        # load hdf5 data which contain all the video features info
        h5f = h5py.File(os.path.join(self._data_path,self._mode+'_video_feat.h5'),'r')

        data = []
        for sample in json_data:
            responses = sample['response']
            labels = sample['label']
            video_id = sample['video_context_id']
            chat_context = sample['chat_context']
            if self._loss_function == '3triplet' and self._mode == 'train':
                data.append((video_id,chat_context,responses,labels))
            else:
                for i in range(len(responses)):
                    data.append((video_id,chat_context,responses[i],labels[i]))

        print 'total samples:',len(data)        
        if self._mode == 'train':
            np.random.shuffle(data)

        for i in range(0,len(data),self._batch_size):
            start = i
            if i+self._batch_size > len(data): # handling leftovers
                end = len(data)
                current_batch_size = end-start
            else:
                end = i+self._batch_size
                current_batch_size = self._batch_size

            if self._loss_function == '3triplet' and self._mode == 'train':
                current_batch_size = 4*current_batch_size

            original_video_id,original_chat_context,original_response,original_label = zip(*data[start:end])
            if self._loss_function == '3triplet' and self._mode == 'train':
                original_video_id = self._reorder_list(original_video_id)
                original_chat_context = self._reorder_list(original_chat_context)
                original_response = self._reorder_list(original_response)
                original_label = self._reorder_list(original_label)

            original_video_context = [h5f[key][:] for key in original_video_id]
            response = [chat +' '+STOP_DECODING for chat in original_response] 
            chat_context = original_chat_context

            if self._word_level:
                chat_context_ind = map(lambda x: [self._vocab.word2id(word) for word in x.split()[::-1]] , chat_context)             
            else:
                chat_context_ind = map(lambda x: [self._vocab.word2id(word) for word in list(x)[::-1]] , chat_context) 
            
            if self._word_level:
                response_ind = map(lambda x: [self._vocab.word2id(word) for word in x.split()] , response)
            else:
                response_ind = map(lambda x: [self._vocab.word2id(word) for word in list(x)] , response)
            chat_context_batch = sequence.pad_sequences(chat_context_ind,padding='post',maxlen=self._max_context_enc_steps)
            response_batch = sequence.pad_sequences(response_ind,padding='post',maxlen=self._max_response_enc_steps)  
            # masking chats
            chat_context_mask_batch = np.zeros((current_batch_size, self._max_context_enc_steps))
            chat_context_len_batch = np.array( map(lambda x: (x != 0).sum(), chat_context_batch))
            for ind, row in enumerate(chat_context_mask_batch):
                row[:chat_context_len_batch[ind]] = 1
            
            response_mask_batch = np.zeros((current_batch_size, self._max_response_enc_steps))
            response_len_batch = np.array( map(lambda x: (x != 0).sum(), response_batch))
            for ind, row in enumerate(response_mask_batch):
                row[:response_len_batch[ind]] = 1

            # transform/clip frames
            video_batch = np.zeros((current_batch_size,self._max_video_enc_steps,self._img_dim))
            video_mask_batch = np.zeros((current_batch_size,self._max_video_enc_steps))
            for idx,feat in enumerate(original_video_context):
                if len(feat)>self._max_video_enc_steps:
                    video_batch[idx][:] = feat[:self._max_video_enc_steps]
                    video_mask_batch[idx][:] = 1
                else:
                    video_batch[idx][:len(feat)] = feat
                    video_mask_batch[idx][:len(feat)] = 1

            label_batch = np.array(original_label)

            batch = Batch()
            batch.put('original_chat_context',original_chat_context)
            batch.put('original_response',original_response)
            batch.put('chat_context_batch',chat_context_batch)
            batch.put('response_batch',response_batch)
            batch.put('chat_context_mask_batch',chat_context_mask_batch)
            batch.put('response_mask_batch',response_mask_batch)
            batch.put('video_batch',video_batch)
            batch.put('video_mask_batch',video_mask_batch)
            batch.put('video_id',original_video_id)
            batch.put('label_batch',label_batch)

            yield batch
        yield None
        
    def get_batcher(self):
        """
        This module process data and creates batches for train/val/test 
        Also acts as generator
        """
        while(True):
            batch_gen = self._process_data()
            while(True):
                batch = batch_gen.next()
                if batch is None:
                    break
                else:
                    yield batch

            if self._mode != 'train':
                break
    


