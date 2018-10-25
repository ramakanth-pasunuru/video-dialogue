import tensorflow as tf
import numpy as np
import argparse
import time
import os
import sys
from collections import namedtuple
from utils.summary_handler import SummaryHandler
from utils.twitch_retrieval_data_generator import *
from models.tridaf_selfatten_visual_dialog_retrieval import *

def eval_valset(args,sess,model):
    args_dict = vars(args)
    tmp_args = namedtuple('val_args',args_dict.keys())(**args_dict)
    tmp_args = tmp_args._replace(mode='val')
    tmp_args = tmp_args._replace(batch_size=20)
    if args.word_level:
        vocab = Vocab('data/'+args.dataset+'/vocab_word',max_size=args.vocab_size)
    else:
        vocab = Vocab('data/'+args.dataset+'/vocab',max_size=args.vocab_size)

    data_generator = Batcher(tmp_args,vocab)
    batcher = data_generator.get_batcher()

    print 'calculate scores on the validation set'

    recall_1_10 = 0.0
    recall_2_10 = 0.0
    recall_5_10 = 0.0
    counter = 0         
    while(True):
        try:
            batch = batcher.next()
        except:
            break
        print 'I am here'
        start = time.time()
        probs = sess.run(model.probs,
                                        feed_dict={
                                                model.video_enc_batch:batch.get('video_batch'),
                                                model.video_enc_mask_batch:batch.get('video_mask_batch'),
                                                model.context_enc_batch:batch.get('chat_context_batch'),
                                                model.context_enc_mask_batch:batch.get('chat_context_mask_batch'),
                                                model.response_enc_batch:batch.get('response_batch'),
                                                model.response_enc_mask_batch:batch.get('response_mask_batch')
                                                })
        current_batch = len(batch.get('label_batch'))
        for k in range(0,current_batch,10):
            response_probs = probs[k:k+10]
            sorted_index = np.argsort(response_probs)
            sorted_index = sorted_index[::-1]
            if np.any(sorted_index[0:1]== 0):
                recall_1_10 += 1

            if np.any(sorted_index[0:2]==0):
                recall_2_10 += 1

            if np.any(sorted_index[0:5]==0):
                recall_5_10 += 1

        counter += current_batch/10

    print 'recall@k scores:'
    print 'recall@1 in 10:',recall_1_10/counter
    print 'recall@2 in 10:',recall_2_10/counter
    print 'recall@5 in 10:',recall_5_10/counter
    scores = {}
    scores['recall@1in10'] = recall_1_10/counter
    scores['recall@2in10'] = recall_2_10/counter
    scores['recall@5in10'] = recall_5_10/counter

    return scores


def train(args):
    if args.word_level:
        vocab = Vocab('data/'+args.dataset+'/vocab_word',max_size=args.vocab_size)
    else:
        vocab = Vocab('data/'+args.dataset+'/vocab',max_size=args.vocab_size)
    data_generator = Batcher(args,vocab)
    batcher = data_generator.get_batcher()

    if args.use_glove:
        if args.word_level:
            wemb = np.load('data/'+args.dataset+'/Wemb_word.npy')
            wemb = wemb.astype('float32')
    else:
        wemb = None
    
    model = VisualDialogRetrieval( vocab_size=args.vocab_size,
                                    hidden_dim=args.hidden_dim,
                                    max_video_enc_steps=args.max_video_enc_steps,
                                    max_context_enc_steps=args.max_context_enc_steps,
                                    max_response_enc_steps=args.max_response_enc_steps,
                                    emb_dim=args.emb_dim,
                                    img_dim=args.img_dim,
                                    num_layers=args.num_layers,
                                    rand_unif_init_mag=args.rand_unif_init_mag,
                                    trunc_norm_init_std=args.trunc_norm_init_std,
                                    cell_type=args.cell_type,
                                    optimizer_type = args.optimizer_type,
                                    learning_rate = args.lr,
                                    max_grad_clip_norm = args.max_grad_clip_norm,
                                    enable_video_context = args.video_context,
                                    enable_chat_context = args.chat_context,
                                    loss_function = args.loss_function,
                                    wemb = wemb,
                                    enable_dropout=False,
                                    is_training=True)
    
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)


    # print the variables
    for var in tf.global_variables():
        print var

    #create a summary handler
    summary_handler = SummaryHandler(os.path.join(args.summary_save_path,args.model_name),
                                        ['LOSS','recall@1in10','recall@2in10','recall@5in10'])

    saver = tf.train.Saver(max_to_keep=50)
    sess.run(tf.global_variables_initializer())
    if args.load_model != 'None':
            saver.restore(sess,os.path.join(args.model_save_path,args.load_model))
 
    iteration = args.start_iter      
    while(True):
        batch = batcher.next()
        if batch is None:
            batch = batcher.next()
        start = time.time()
        loss,debugger,_ = sess.run([model.loss,model.debugger,model.train_op],
                                        feed_dict={
                                            model.video_enc_batch:batch.get('video_batch'),
                                                model.video_enc_mask_batch:batch.get('video_mask_batch'),
                                                model.context_enc_batch:batch.get('chat_context_batch'),
                                                model.context_enc_mask_batch:batch.get('chat_context_mask_batch'),
                                                model.response_enc_batch:batch.get('response_batch'),
                                                model.response_enc_mask_batch:batch.get('response_mask_batch'),
                                                model.target_label_batch:batch.get('label_batch')
                                                })

        summary = {}
        summary['LOSS'] = loss
        summary['ITERATION'] = iteration
        summary_handler.write_summaries(sess,summary)

        iteration +=1
        print 'iteration:',iteration,'computational time:',time.time()-start,'loss:',loss

        if iteration > args.max_iter:
            break
        if iteration%args.check_point == 0:
            # get validation loss and perplexity
            scores = eval_valset(args,sess,model)
            summary = scores
            summary['ITERATION'] = iteration
            summary_handler.write_summaries(sess,summary)
            saver.save(sess, os.path.join(args.model_save_path, args.model_name+'-'+str(iteration)))

	

def main():

        parser = argparse.ArgumentParser()
        parser.add_argument('--lr',type=float,default=0.00001,help='learning rate used in the optimizer')
        parser.add_argument('--rand_unif_init_mag',type=float,default=0.01,help='learning rate used in the optimizer')
        parser.add_argument('--trunc_norm_init_std',type=float,default=1e-4,help='learning rate used in the optimizer')
        parser.add_argument('--max_grad_clip_norm',type=float,default=2.0,help='learning rate used in the optimizer')
        parser.add_argument('--hidden_dim',type=int,default=256,help='hidden state size of the RNN')
        parser.add_argument('--max_video_enc_steps',type=int,default=60,help='number of time steps of the encoder RNN')
        parser.add_argument('--max_context_enc_steps',type=int,default=70,help='number of time steps of the encoder RNN')
        parser.add_argument('--max_response_enc_steps',type=int,default=10,help='number of time steps of the decoder RNN')
        parser.add_argument('--emb_dim',type=int,default=100,help='words embedding size')
        parser.add_argument('--img_dim',type=int,default=2048,help='image feature embedding projection')
        parser.add_argument('--vocab_size',type=int,default=27000,help='vocabulary size')
        parser.add_argument('--batch_size',type=int,default=16,help='mini-batch training size')
        parser.add_argument('--mode',type=str,default='train',help='mini-batch training size')
        parser.add_argument('--num_layers',type=int,default=1,help='multi-layer RRN (number of layers)') 
        parser.add_argument('--window_size',type=int,default=5,help='size of the window to consider in chats and video; 5 indicates 5 seconds')
        parser.add_argument('--optimizer_type',type=str,default='adam',help='optimizer for training')
        parser.add_argument('--cell_type',type=str,default='lstm',help='cell type like lstm or gru or something else')
        parser.add_argument('--data_path',type=str,default='',help='root location of the dataset')
        parser.add_argument('--dataset',type=str,default='fifa',choices=['twitch','fifa'],help='choose the dataset')
        parser.add_argument('--model_save_path',type=str,default='',help='path where model to be saved')
        parser.add_argument('--summary_save_path',type=str,default='',help='path where training summary is saved')
        parser.add_argument('--eval_save_path',type=str,default='',help='path where the validation results are stored')
        parser.add_argument('--model_name',type=str,help='model name')
        parser.add_argument('--check_point',type=int,default=500,help='after every checkpoint number of iterations calculate the scores')
        parser.add_argument('--saving_metric_type',type=str,default='BLEU-4',choices=['ROUGE-2','METEOR','BLEU-4'],help='choose metric type as criteria for saving best model')
        parser.add_argument('--load_model',type=str,default='None',help='start training from this model')
        parser.add_argument('--video_context',dest='video_context',action='store_false',default=True)
        parser.add_argument('--chat_context',dest='chat_context',action='store_false',default=True)
        parser.add_argument('--word_level',dest='word_level',action='store_true',default=False) # by default models are char level
        parser.add_argument('--negative_examples',type=str,default='soft',help='choose the type of negative examples')
        parser.add_argument('--loss_function',type=str,default='cross-entropy',help='choose from cross-entropy loss or triplet loss')

        parser.add_argument('--max_iter',type=int,default=30000,help='set the maximum number of iterations')
        parser.add_argument('--start_iter',type=int,default=0,help='staring point of the iteration, useful incase starting from a previous saved model')
        parser.add_argument('--use_glove',dest='use_glove',action='store_true',default=False)

	args = parser.parse_args()
        print args

	# set seed 
	np.random.seed(seed=737)
	tf.set_random_seed(seed=737)

	train(args)


if __name__ == '__main__':
	main()
	


