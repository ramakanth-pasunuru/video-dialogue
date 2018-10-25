import tensorflow as tf
import numpy as np
import argparse
import time
import os
import sys
import codecs
from utils.twitch_retrieval_data_generator import *
from models.tridaf_selfatten_visual_dialog_retrieval import *


NUM_NEG_SAMPLES = 9

def write_results(args,scores):
        f = open(os.path.join(args.eval_save_path,args.model_name+'_samples_'+str(NUM_NEG_SAMPLES+1)+'_'+args.negative_examples+'_result'),'w')
        f.write('############ PARAMETERS ############\n')
        f.write(str(args)+'\n')
        f.write('########### SCORES  ###########\n')
        f.write(str(scores))
        f.close()

def write_stat_results(args,scorer):
    f = open(os.path.join(args.eval_save_path,args.model_name+'_samples_'+str(NUM_NEG_SAMPLES+1)+'_'+args.negative_examples+'_stats'),'w')
    f.write('\n'.join(scorer))
    f.close()



def test(args):

    if args.word_level:	
        vocab = Vocab('data/'+args.dataset+'/vocab_word',max_size=args.vocab_size)
    else:
        vocab = Vocab('data/'+args.dataset+'/vocab',max_size=args.vocab_size)

    data_generator = Batcher(args,vocab)
    batcher = data_generator.get_batcher()

    # score the recall score for each input for caluculating the statiscticall scores
    save_scorer = []
    
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
                                    enable_dropout=False,
                                    is_training=False)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)


    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,os.path.join(args.model_save_path,args.model_name))
    
    recall_1_5 = 0.0
    recall_2_5 = 0.0
    recall_3_5 = 0.0
    counter = 0         
    while(True):
        try:
            batch = batcher.next()
        except:
            break
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
        for k in range(0,current_batch,NUM_NEG_SAMPLES+1):
            tmp = [] # for scorer
            response_probs = probs[k:k+NUM_NEG_SAMPLES+1]
            sorted_index = np.argsort(response_probs)
            sorted_index = sorted_index[::-1]
            if np.any(sorted_index[0:1]== 0):
                recall_1_5 += 1
                tmp.append(str(1))
            else:
                tmp.append(str(0))
                
            if np.any(sorted_index[0:2]==0):
                recall_2_5 += 1
                tmp.append(str(1))
            else:
                tmp.append(str(0))

            if np.any(sorted_index[0:5]==0):
                recall_3_5 += 1
                tmp.append(str(1))
            else:
                tmp.append(str(0))

            save_scorer.append(",".join(tmp))


        counter += current_batch/(NUM_NEG_SAMPLES+1)

    print 'recall@k scores:'
    print 'recall@1 in 10:',recall_1_5/counter
    print 'recall@2 in 10:',recall_2_5/counter
    print 'recall@5 in 10:',recall_3_5/counter
    scores = {}
    scores['recall@1in10'] = recall_1_5/counter
    scores['recall@2in10'] = recall_2_5/counter
    scores['recall@5in10'] = recall_3_5/counter

    write_stat_results(args,save_scorer)

    # write results
    write_results(args,scores)


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
    parser.add_argument('--batch_size',type=int,default=100,help='mini-batch training size')
    parser.add_argument('--mode',type=str,default='test',help='mini-batch training size')
    parser.add_argument('--num_layers',type=int,default=1,help='multi-layer RRN (number of layers)') 
    parser.add_argument('--window_size',type=int,default=5,help='size of the window to consider in chats and video; 5 indicates 5 seconds')
    parser.add_argument('--optimizer_type',type=str,default='adam',help='optimizer for training')
    parser.add_argument('--cell_type',type=str,default='lstm',help='cell type like lstm or gru or something else')
    parser.add_argument('--data_path',type=str,default='',help='root location of the dataset')
    parser.add_argument('--dataset',type=str,default='fifa',choices=['twitch','fifa'],help='choice of the datasets')
    parser.add_argument('--model_save_path',type=str,default='',help='path where model to be saved')
    parser.add_argument('--summary_save_path',type=str,default='',help='path where training summary is saved')
    parser.add_argument('--eval_save_path',type=str,default='',help='path where the validation results are stored')
    parser.add_argument('--model_name',type=str,help='model name')
    parser.add_argument('--check_point',type=int,default=500,help='after every checkpoint number of iterations calculate the scores')
    parser.add_argument('--saving_metric_type',type=str,default='BLEU-4',choices=['ROUGE-2','METEOR','BLEU-4'],help='choose metric type as criteria for saving best model')
    parser.add_argument('--load_model',type=str,default='None',help='start training from this model')
    parser.add_argument('--beam_size',type=int,default=1,help='beam search size')
    parser.add_argument('--video_context',dest='video_context',action='store_false',default=True)
    parser.add_argument('--chat_context',dest='chat_context',action='store_false',default=True)
    parser.add_argument('--word_level',dest='word_level',action='store_true',default=False) # by default models are char level
    parser.add_argument('--negative_examples',type=str,default='soft',help='choose the type of negative examples')
    parser.add_argument('--loss_function',type=str,default='cross-entropy',help='choose from cross-entropy loss or triplet loss')


    
    args = parser.parse_args()

    # set seed 
    np.random.seed(seed=111)
    tf.set_random_seed(seed=111)

    test(args)


if __name__ == '__main__':
	main()
	


