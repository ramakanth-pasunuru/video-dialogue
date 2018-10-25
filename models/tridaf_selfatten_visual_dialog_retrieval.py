import tensorflow as tf 
import numpy as np 

'''
Author: Ramakanth Pasunuru
Details:


'''

class VisualDialogRetrieval(object):

    def __init__(self,  vocab_size=1000,
                        hidden_dim=256,
                        max_video_enc_steps=50,
                        max_context_enc_steps=50,
                        max_response_enc_steps=20,
                        emb_dim=128,
                        num_layers=2,
                        img_dim = 1536,
                        rand_unif_init_mag=0.08,
                        trunc_norm_init_std=1e-4,
                        cell_type='lstm',
                        optimizer_type = 'adam',
                        learning_rate = 0.001,
                        max_grad_clip_norm = 10,
                        beam_size = 1,
                        wemb = None,
                        loss_function = 'cross-entropy',
                        enable_video_context=True,
                        enable_chat_context=True,
                        enable_dropout=True,
                        is_training=True):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_video_enc_steps = max_video_enc_steps
        self.max_context_enc_steps = max_context_enc_steps
        self.max_response_enc_steps = max_response_enc_steps
        self.emb_dim = emb_dim
        self.img_dim = img_dim
        self.rand_unif_init_mag = rand_unif_init_mag
        self.trunc_norm_init_std = trunc_norm_init_std
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.max_grad_clip_norm = max_grad_clip_norm
        self.beam_size = beam_size
        self.enable_dropout = enable_dropout
        self.is_training = is_training
        self.enable_video_context = enable_video_context
        self.enable_chat_context = enable_chat_context
        self.loss_function = loss_function
        # create a debugger variable to use it to debug some other variable
        self.debugger = []
        if self.enable_dropout:
                self.keep_prob = tf.placeholder(tf.float32)
        # word embedding look up
        if wemb is None:
            self.wemb = tf.Variable(tf.random_uniform([self.vocab_size,self.emb_dim], -self.rand_unif_init_mag,self.rand_unif_init_mag), name='Wemb')
        else:
            self.wemb = tf.Variable(wemb,name='Wemb')
        self.rand_unif_init = tf.random_uniform_initializer(-self.rand_unif_init_mag, self.rand_unif_init_mag)
        self.trunc_normal_init = tf.truncated_normal_initializer(stddev=self.trunc_norm_init_std)
        # creating rnn cells
        if self.cell_type == 'lstm':
            self.rnn_cell = tf.contrib.rnn.LSTMCell
        elif self.cell_type == 'gru':
            self.rnn_cell = tf.contrib.rnn.GRUCell
        # multi-layer cell setup 
        if self.num_layers > 1:
            self.video_enc_rnn_cell_fw = []
            self.video_enc_rnn_cell_bw = []
            self.context_enc_rnn_cell_fw = []
            self.context_enc_rnn_cell_bw = []
            self.response_enc_rnn_cell_fw = []
            self.response_enc_rnn_cell_bw = []
            for _ in range(self.num_layers):
                self.video_enc_rnn_cell_fw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.video_enc_rnn_cell_bw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.context_enc_rnn_cell_fw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.context_enc_rnn_cell_bw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.response_enc_rnn_cell_fw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))
                self.response_enc_rnn_cell_bw.append(self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init))

            self.video_enc_rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(self.video_enc_rnn_cell_fw,state_is_tuple=True)
            self.video_enc_rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(self.video_enc_rnn_cell_bw,state_is_tuple=True)
            self.context_enc_rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(self.context_enc_rnn_cell_fw,state_is_tuple=True)
            self.context_enc_rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(self.context_enc_rnn_cell_bw,state_is_tuple=True)
            self.response_enc_rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(self.response_enc_rnn_cell_fw,state_is_tuple=True)
            self.response_enc_rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(self.response_enc_rnn_cell_bw,state_is_tuple=True)
        else:
            self.video_enc_rnn_cell_fw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.video_enc_rnn_cell_bw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.context_enc_rnn_cell_fw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.context_enc_rnn_cell_bw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.response_enc_rnn_cell_fw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)
            self.response_enc_rnn_cell_bw = self.rnn_cell(self.hidden_dim,initializer=self.rand_unif_init)

        ## add dropout to lstm
        if self.enable_dropout:
            self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell,output_keep_prob=self.keep_prob)	
        # creating placeholders
        """ this video enc plc has to be updated"""
        self.video_enc_batch = tf.placeholder(tf.float32, [None, None,self.img_dim])
        self.video_enc_mask_batch = tf.placeholder(tf.float32, [None, None])
        # chat encoder placeholders
        self.context_enc_batch = tf.placeholder(tf.int32, [None, self.max_context_enc_steps])
        self.context_enc_mask_batch = tf.placeholder(tf.float32, [None, self.max_context_enc_steps])
        # chat decoder placeholders
        self.response_enc_batch = tf.placeholder(tf.int32, [None, self.max_response_enc_steps])
        self.response_enc_mask_batch = tf.placeholder(tf.float32, [None, self.max_response_enc_steps])
        # target label placeholder
        self.target_label_batch = tf.placeholder(tf.int32,[None])
        # get batch size
        self.batch_size = tf.shape(self.video_enc_batch)[0]
        #self.max_video_enc_steps = tf.shape(self.video_enc_batch)[1]
        # get video encoder

        _,self.video_encoder_hidden_states = self._encoder(self.video_enc_rnn_cell_fw,self.video_enc_rnn_cell_bw,self.video_enc_batch,self.video_enc_mask_batch,scope='video_encoder')



        _,self.context_encoder_hidden_states = self._encoder(self.context_enc_rnn_cell_fw,self.context_enc_rnn_cell_bw,self.context_enc_batch,self.context_enc_mask_batch,scope='context_encoder')


        _,self.response_encoder_hidden_states = self._encoder(self.response_enc_rnn_cell_fw,self.response_enc_rnn_cell_bw,self.response_enc_batch,self.response_enc_mask_batch,scope='response_encoder')

        # bidaf between video and chat context
        self.chat_on_video_states,self.video_on_chat_states = self._cross_attention(self.video_encoder_hidden_states,self.context_encoder_hidden_states,self.video_enc_mask_batch,self.context_enc_mask_batch,self.max_video_enc_steps,self.max_context_enc_steps,scope='bidaf_video_chat')

        # bidaf between response and chat context
        self.chat_on_response_states,self.response_on_chat_states = self._cross_attention(self.response_encoder_hidden_states,self.context_encoder_hidden_states,self.response_enc_mask_batch,self.context_enc_mask_batch,self.max_response_enc_steps,self.max_context_enc_steps,scope='bidaf_response_chat')

        # bidaf between video and response context
        self.response_on_video_states,self.video_on_response_states = self._cross_attention(self.video_encoder_hidden_states,self.response_encoder_hidden_states,self.video_enc_mask_batch,self.response_enc_mask_batch,self.max_video_enc_steps,self.max_response_enc_steps,scope='bidaf_video_response')

        self.video_final_enc_states = tf.concat([self.video_encoder_hidden_states,self.chat_on_video_states,self.response_on_video_states],axis=2)
        self.chat_final_enc_states = tf.concat([self.context_encoder_hidden_states,self.response_on_chat_states,self.video_on_chat_states],axis=2)
        self.response_final_enc_states = tf.concat([self.response_encoder_hidden_states,self.video_on_response_states,self.chat_on_response_states],axis=2)

        
        self.video_encoder_final_state = self._self_attention(self.video_final_enc_states,self.video_enc_mask_batch,self.max_video_enc_steps,scope='video_self_attention')
        self.context_encoder_final_state = self._self_attention(self.chat_final_enc_states,self.context_enc_mask_batch,self.max_context_enc_steps,scope='chat_self_attention')
        self.response_encoder_final_state = self._self_attention(self.response_final_enc_states,self.response_enc_mask_batch,self.max_response_enc_steps,scope='response_self_attention')


        if self.is_training:
            self.loss,self.probs = self._calculate_loss(self.video_encoder_final_state,self.context_encoder_final_state,self.response_encoder_final_state,self.target_label_batch)
            # calculate gradients
            if self.optimizer_type == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer_type == 'sgd':
                    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer_type == 'adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif self.optimizer_type == 'adadelta':
                    self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            else:
                    raise Exception('optimizer type:{} not supported'.format(self.optimizer_type))
    
            self.tvars = tf.trainable_variables()
            self.grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,self.tvars),self.max_grad_clip_norm)
            self.train_op=self.optimizer.apply_gradients(zip(self.grads,self.tvars))
        else:
            logits = self._projection_layer(self.video_encoder_final_state,self.context_encoder_final_state,self.response_encoder_final_state)
            self.probs = tf.sigmoid(logits)



    def _video_embed(self,video_batch):
        """ takes video batch of size batch_size,encoder_steps,features and project it 
            down to lower space: batch_size,encoder_steps,embedding_feature space

        """
        video_batch = tf.reshape(video_batch,[self.batch_size*self.max_video_enc_steps,self.img_dim])

        with tf.variable_scope('video_embeddings',reuse=False):
            weight = tf.get_variable('weights',[self.img_dim,256])
            emb_video_batch = tf.matmul(video_batch,weight)

        emb_video_batch = tf.reshape(emb_video_batch,[self.batch_size,self.max_video_enc_steps,256])
        return emb_video_batch

    def _final_encoder(self,cell,inputs,inputs_mask,scope):
        # find the seqence lengths from the mask placeholders
        seq_len = tf.reduce_sum(inputs_mask,1)
        seq_len = tf.cast(seq_len,dtype=tf.int32)
        outputs,state = tf.nn.dynamic_rnn(cell,inputs,sequence_length=seq_len,dtype=tf.float32,swap_memory=True,scope=scope)
        return state,outputs


    def _encoder(self,cell_fw,cell_bw,inputs,inputs_mask,scope):
        # ENCODER PART
        if 'video' in scope:
            current_embed = self._video_embed(inputs)
        else:
            current_embed = tf.nn.embedding_lookup(self.wemb,inputs)
        # add dropout
        if self.enable_dropout:
            current_embed = tf.nn.dropout(current_embed,self.keep_prob)
        # find the seqence lengths from the mask placeholders
        seq_len = tf.reduce_sum(inputs_mask,1)
        seq_len = tf.cast(seq_len,dtype=tf.int32)

        (outputs,(state_fw,state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,current_embed,sequence_length=seq_len,dtype=tf.float32,swap_memory=True,scope=scope)
        outputs = tf.concat(outputs,axis=2) # hidden fw/bw are in tuple form, concat them
        state = self._reduce_encoder_state(state_fw,state_bw,scope)
        return state,outputs

    def _reduce_encoder_state(self,state_fw,state_bw,scope):
        with tf.variable_scope(scope+'_reduce_encoder_final_state',reuse=False):
            w_reduce_c = tf.get_variable('w_reduce_c', [self.hidden_dim*2, self.hidden_dim], dtype=tf.float32, initializer=self.trunc_normal_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [self.hidden_dim*2, self.hidden_dim], dtype=tf.float32, initializer=self.trunc_normal_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c',[self.hidden_dim],dtype=tf.float32, initializer=self.trunc_normal_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h',[self.hidden_dim],dtype=tf.float32, initializer=self.trunc_normal_init)

        if self.num_layers <=1:
            old_c = tf.concat([state_fw.c, state_bw.c],axis=1) # Concatenation of fw and bw cell
            old_h = tf.concat([state_fw.h, state_bw.h],axis=1) # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
            return tf.nn.rnn_cell.LSTMStateTuple(new_c,new_h)
        else:
            final_state = []
            for i in range(self.num_layers):
                old_c = tf.concat([state_fw[i].c, state_bw[i].c],axis=1) # Concatenation of fw and bw cell
                old_h = tf.concat([state_fw[i].h, state_bw[i].h],axis=1) # Concatenation of fw and bw state
                new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
                new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
                final_state.append(tf.nn.rnn_cell.LSTMStateTuple(new_c,new_h))
            return final_state

    
    def _self_attention(self,hidden_states,mask,step_size,scope):


        compressed_hidden_states = tf.reshape(hidden_states,[self.batch_size*step_size,6*self.hidden_dim])
        with tf.variable_scope(scope,reuse=False):
            Wa = tf.get_variable('Wa',[6*self.hidden_dim,6*self.hidden_dim])
            ba = tf.get_variable('ba',[6*self.hidden_dim],initializer=tf.constant_initializer(0.0))

            output = tf.tanh(tf.matmul(compressed_hidden_states,Wa)+tf.tile(tf.expand_dims(ba,0),[self.batch_size*step_size,1]))
            #print output
            V = tf.get_variable('V',[6*self.hidden_dim,1])

            attn_dist = tf.matmul(output,V)
            #print attn_dist
            attn_dist = tf.reshape(attn_dist,[self.batch_size,step_size])
            attn_dist = tf.nn.softmax(attn_dist)
            attn_dist *= mask
            #print attn_dist
            masked_sums = tf.reduce_sum(attn_dist,1)
            #print masked_sums
            attn_dist = attn_dist/tf.reshape(masked_sums,[-1,1])
            #print attn_dist

        context_vector= tf.reduce_sum(tf.reshape(attn_dist,[self.batch_size,-1,1,1]) * tf.expand_dims(hidden_states,axis=2), [1,2]) # shape: batch_size,2*dim_hidden
        #print context_vector

        return tf.nn.rnn_cell.LSTMStateTuple(context_vector,context_vector)

    def _cross_attention(self,enc1_states,enc2_states,enc1_mask,enc2_mask,max_enc1_steps,max_enc2_steps,scope):

        # first calculate the similarity matrix
        # video state size --> batch_size, max_video_enc_steps, 2*dim_hidden
        # context state size --> batch_size, max_context_enc_steps, 2*dim_hidden
        # size of simialrity matrix = batch_size, max_video_enc_steps,max_context_enc_steps
        similarity_matrix = [] 
        with tf.variable_scope(scope+'/similarity_matrix', reuse=False):
            weight = tf.get_variable('weights',[6*self.hidden_dim,1])

        for i in range(max_enc1_steps):
            repeat_vc = tf.tile(tf.expand_dims(enc1_states[:,i],0),[max_enc2_steps,1,1])
            repeat_vc = tf.transpose(repeat_vc,[1,0,2])
            h = tf.concat([repeat_vc,enc2_states,repeat_vc*enc2_states],axis=2)
            score = tf.matmul(h,tf.tile(tf.expand_dims(weight,0),[self.batch_size,1,1]))
            similarity_matrix.append(score)

        similarity_matrix = tf.stack(similarity_matrix) # size = max_video_enc_steps,batch_size,max_context_enc_steps, 1
        similarity_matrix = tf.reshape(similarity_matrix,[max_enc1_steps,self.batch_size,max_enc2_steps])
        similarity_matrix = tf.transpose(similarity_matrix,[1,0,2])


        '''renormalize attention'''
        # gent enc2 attention weights based on enc1
        enc2_attention_weights = tf.nn.softmax(similarity_matrix)
        # make the mask part to zero weights
        enc2_attention_weights = enc2_attention_weights*tf.tile(tf.expand_dims(enc2_mask,axis=1),[1,max_enc1_steps,1])
        # renormize the attention weights
        masked_sum = tf.tile(tf.expand_dims(tf.reduce_sum(enc2_attention_weights,2),2),[1,1,max_enc2_steps])

        enc2_attention_weights = enc2_attention_weights/masked_sum
        enc2_on_enc1_context = tf.matmul(enc2_attention_weights,enc2_states)

        # gent enc1 attention weights based on enc2
        enc1_attention_weights = tf.nn.softmax(tf.transpose(similarity_matrix,[0,2,1]))
        # make the mask part to zero weights
        enc1_attention_weights = enc1_attention_weights*tf.tile(tf.expand_dims(enc1_mask,axis=1),[1,max_enc2_steps,1])
        # renormize the attention weights
        masked_sum = tf.tile(tf.expand_dims(tf.reduce_sum(enc1_attention_weights,2),2),[1,1,max_enc1_steps])
        enc1_attention_weights = enc1_attention_weights/masked_sum
        enc1_on_enc2_context = tf.matmul(enc1_attention_weights,enc1_states)

        return enc2_on_enc1_context,enc1_on_enc2_context


    def _projection_layer(self,video_state,context_state,response_state,reuse=False):

        # consider only input hidden state information 
        response_state = response_state.h

        if self.enable_video_context:
            video_state = video_state.h
            # calculate projections
            with tf.variable_scope('video_projection',reuse=False):
                W_v = tf.get_variable('weights',[6*self.hidden_dim,6*self.hidden_dim])
            video_proj = tf.matmul(video_state,W_v)
            video_proj = tf.expand_dims(video_proj,[2])
        
        if self.enable_chat_context:
            context_state = context_state.h
            ''' have to correct the name space later '''
            with tf.variable_scope('context_projection',reuse=False):
                W_c = tf.get_variable('weights',[6*self.hidden_dim,6*self.hidden_dim])
            context_proj = tf.matmul(context_state,W_c)
            context_proj = tf.expand_dims(context_proj,[2])

        response_state = tf.expand_dims(response_state,[2])

        with tf.variable_scope('projection_layer',False):
            b = tf.get_variable('b',[1],initializer=tf.constant_initializer(0.0))

        
        if self.enable_video_context and self.enable_chat_context:
            logits = tf.add(tf.matmul(video_proj,response_state,True),tf.matmul(context_proj,response_state,True))+b
        elif self.enable_video_context and not self.enable_chat_context:
            logits = tf.matmul(video_proj,response_state,True)+b
        elif not self.enable_video_context and self.enable_chat_context:
            logits = tf.matmul(context_proj,response_state,True)+b
        else:
            raise Exception('At least one context must be present !!')

        logits = tf.squeeze(logits, [1,2])

        return logits


    def _calculate_loss(self,video_state,context_state,response_state,target_labels):

        logits = self._projection_layer(video_state,context_state,response_state)
        probs = tf.sigmoid(logits)
        self.debugger = logits
        if self.loss_function == 'cross-entropy':
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(target_labels))
            loss = tf.reduce_mean(cross_entropy)

        elif self.loss_function == 'triplet':
            # first half are positive examples and second half are negative examples
            log_prob = tf.log(probs)
            #print log_prob
            log_prob = tf.split(log_prob,2)
            #print log_prob
            loss = tf.maximum(0.0,0.1+log_prob[1]-log_prob[0])
            loss = tf.reduce_mean(loss)

        elif self.loss_function == '3triplet':
            log_prob = tf.log(probs)
            log_prob = tf.split(log_prob,4)
            loss = tf.maximum(0.0,0.1+log_prob[1]-log_prob[0]) 
            if self.enable_video_context: # with video negative example
                loss += tf.maximum(0.0,0.1+log_prob[2]-log_prob[0])
            if self.enable_chat_context: # with chat negative eample
                loss += tf.maximum(0.0,0.1+log_prob[3]-log_prob[0])
            loss = tf.reduce_mean(loss)



        else:
            raise Exception('Unknown loss function:{}'.format(self.loss_function)) 
            

        return loss,probs


    
