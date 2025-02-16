


import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf2
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.layers as layers
from collections import deque, Counter

import SimplePursuringDiscreteEnv5 as spEnv

tf2.config.set_visible_devices([], 'GPU')

#----------------------------------------------------------------------------- 
# PARAMETERS

discount_factor = 0.95

eps_min = 0.2
eps_max = 1
eps_decay_steps = 200000

num_episodes = 500000
batch_size = 50
learning_rate = 0.01
X_shape = (None,10)


global_step = 0
copy_steps = 1000

steps_train = 1
start_steps = 2000

logdir = 'logs_49_DDQN'

gradient_threshold = 3

exp_buffer_length = 100000

using_regularizer = tf2.keras.regularizers.L2(0.0001)
regular_ratio = 0

show_steps = 20
dropout_rate = 0.0

#------------------

envS = spEnv.SimplePursuringDiscreteEn(True)
envS.ShappingGamma = discount_factor
envS.VTgt = 0.004
#MaxAbsAction = float(0.3);
envS.verbose = False
envS.FireReward = float(1)

#-----------------------------------------------------------------------------   

n_outputs = 3

def n_to_float_action(n):
    global MaxAbsAction
    return -MaxAbsAction + 2*MaxAbsAction/(n_outputs-1)*n

#-----------------------------------------------------------------------------   
 
#tf.reset_default_graph()

def Q_Network(X, name_scope) :
    
    initializer = tf.keras.initializers.VarianceScaling()    
    
    with tf.variable_scope(name_scope) as scope:

               

        fc1 = layers.Dense( 
                   750, kernel_initializer=initializer, 
                   activation=tf.keras.activations.relu,
                   kernel_regularizer=using_regularizer, 
                   bias_regularizer=using_regularizer )(X)
        
        
        
        
        fc2 = layers.Dense( 750, kernel_initializer=initializer, 
                   activation=tf.keras.activations.relu,
                   kernel_regularizer=using_regularizer,
                   bias_regularizer=using_regularizer)(fc1)
        
        
        
        
        fc3 = layers.Dense(750, kernel_initializer=initializer, 
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=using_regularizer,
                    bias_regularizer=using_regularizer)(fc2)
        
        
        
        
        fc4 = layers.Dense(750, kernel_initializer=initializer, 
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=using_regularizer,
                    bias_regularizer=using_regularizer)(fc3)
        
        fc5 = layers.Dense( 750, kernel_initializer=initializer, 
                   activation=tf.keras.activations.relu,
                   kernel_regularizer=using_regularizer,
                   bias_regularizer=using_regularizer)(fc4)
        
        
        
        
        fc6 = layers.Dense(750, kernel_initializer=initializer, 
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=using_regularizer,
                    bias_regularizer=using_regularizer)(fc5)
        
        
        
        
        fc7 = layers.Dense(750, kernel_initializer=initializer, 
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=using_regularizer,
                    bias_regularizer=using_regularizer)(fc6)
        
        
        
        output = layers.Dense( n_outputs, 
                    kernel_initializer=initializer,
                    kernel_regularizer=using_regularizer,
                    bias_regularizer=using_regularizer)(fc7)
        
    
        Q_vars = {v.name[len(scope.name):]: v for v in 
                  tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, 
                                    scope=scope.name)} 
        
        return Q_vars, output
 
#-----------------------------------------------------------------------------    

def epsilon_greedy(action,step, episode):
    
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max-(eps_max-eps_min)*step/eps_decay_steps)
    
    if (episode) % 100 == 0: 
        epsilon = 0    
        
    if np.random.rand(1) < epsilon:
        a = np.random.randint(n_outputs)
        #print(a)
        return a, epsilon
    else:
        return action, epsilon

#-----------------------------------------------------------------------------    
    
def sample_memories(batch_size):
    if exp_buffer_full:
        size_buff = exp_buffer_length
    else:
        size_buff = exp_buffer_pos
        
    perm_batch = np.random.randint(0,size_buff,batch_size)
    mem = exp_buffer[perm_batch]
    return mem[:,0],mem[:,1],mem[:,2],mem[:,3],mem[:,4]

#-----------------------------------------------------------------------------    

exp_buffer_pos = 0;
exp_buffer_full = False
exp_buffer = np.zeros(shape=(exp_buffer_length,5), dtype=object)

#-----------------------------------------------------------------------------


#tf2.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()

X = tf.placeholder(float, shape=X_shape,name='X')
in_training_mode = tf.placeholder(bool,name='in_training_mode')
kappa = tf.placeholder(float, shape=(), name='kappa')
step = tf.placeholder(tf.int32, shape=(), name='step')

mainQ, mainQ_outputs = Q_Network(X,'maimQ')
targetQ, targetQ_outputs = Q_Network(X,'targetQ')

X_action = tf.placeholder(tf.int32, shape=(None,),name='X_action')
Q_action = tf.reduce_sum(
    mainQ_outputs * tf.one_hot(X_action, n_outputs),
    axis=-1, keep_dims=True )

copy_op = [tf.assign(main_name,mainQ[var_name]) 
           for var_name, main_name in targetQ.items() ] 
copy_target_to_main = tf.group(*copy_op)

y = tf.placeholder( float, shape=(None,1), name='y' )
loss_net = tf.reduce_mean( tf.square(y-Q_action) )
loss_reg = tf.add_n( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
loss = loss_net + kappa*loss_reg


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=30000,
    decay_rate=1)

learning_rate_tensor = lr_schedule(step)

optimazer = tf.train.AdagradOptimizer(learning_rate=learning_rate_tensor)
grads_and_vars = optimazer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad,-gradient_threshold,gradient_threshold),var)
               for grad, var in grads_and_vars]
#training_op = optimazer.apply_gradients(capped_gvs)
training_op = optimazer.minimize(loss) 

EpisodeReward = tf.placeholder(float, shape=())
EpisodeScore = tf.placeholder(float, shape=())
EpisodeEpsilon = tf.placeholder(float, shape=())
LOSS = tf.placeholder(float,shape=())
loss_summary = tf.summary.scalar('LOSS',LOSS)
episode_reward_summary = tf.summary.scalar('EpisodeReward',EpisodeReward)
episode_score_summary = tf.summary.scalar('EpisodeScore',EpisodeScore)
episode_epsilon_summary = tf.summary.scalar('EpisodeEpsilon',EpisodeEpsilon)
merge_summary = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())

#-----------------------------------------------------------------------------

train_loss = 10
train_loss_net = train_loss_reg = 10
kappa_val = train_loss_net * regular_ratio / train_loss_reg        

init = tf.global_variables_initializer()
saver = tf.train.Saver()

learning_rate_current = 0

with tf.Session() as sess:
    init.run()

    nEpisodesRepeat = 0

    for i in range(num_episodes):
        done = False
        obs, _ = envS.reset()
        epoch = 0
        episodic_reward = 0
        episodic_score = 0
        action_counter = Counter()
        episodic_loss = []
        
        train_loss_reg_sum = train_loss_net_sum = 0.0
        train_loss_sum = 0.0
        train_loss_n = 0
        
        
        while not done:
          
            actions = targetQ_outputs.eval (
                feed_dict={X:[obs], in_training_mode:False}) 
                
            action = np.argmax(actions,axis=-1)
            action_counter[str(action)] += 1

            action, epsilonn = epsilon_greedy(action, global_step, i)
            
            next_obs, reward, done, _, _ = envS.step(action)
            score = 0
        
            #--------------------
        
            exp_buffer[exp_buffer_pos,:] = np.array([obs, action, next_obs, reward, done],dtype=object)
            
            exp_buffer_pos += 1
            if exp_buffer_pos >= exp_buffer_length:
                exp_buffer_pos = 0
                exp_buffer_full = True
                print('Buffer full')
            #--------------------
            
            if global_step % steps_train == 0 and global_step > start_steps:
                o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)
                
                o_obs = [x for x in o_obs]
                o_next_obs = [x for x in o_next_obs]
                
                X_obs = o_next_obs
                
                next_act = targetQ_outputs.eval( 
                    feed_dict={X:o_next_obs,in_training_mode:False}) 
                
                y_batch = o_rew + discount_factor * np.max(next_act,axis=-1) * (1-done)

                train_loss1, _, loss_reg1, loss_net1, learning_rate_current = sess.run(
                    [loss, training_op, loss_reg, loss_net, learning_rate_tensor ],
                    feed_dict={X:np.array(o_obs,dtype=float), 
                               y:np.expand_dims(
                                  np.array(y_batch,dtype=float),axis=-1), 
                               X_action:np.array(o_act,dtype=np.int32), 
                               in_training_mode:True,
                               kappa:kappa_val, 
                               step:global_step} )
                
                if train_loss1 > train_loss*10 and False:
                    print('gluk')
                else:
                    train_loss_sum += train_loss1
                    train_loss_reg_sum += loss_reg1 
                    train_loss_net_sum += loss_net1
                    train_loss_n += 1
                    train_loss = train_loss1
                      
            
            if global_step % copy_steps == 0 and global_step > start_steps:
                copy_target_to_main.run()               
                print('   ')
                print('copy_target_to_main.run()')
                print('   ')
            
            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward
            
        
        episodic_score = envS.MaximumSteps - envS.i_D - 1
        
        if train_loss_n > 0: 
            train_loss = train_loss_sum/train_loss_n
            train_loss_net = train_loss_net_sum/train_loss_n
            train_loss_reg = train_loss_reg_sum/train_loss_n
        
        mrg_summary = merge_summary.eval( 
            feed_dict={X:np.zeros(dtype=float,shape=(1,10)),
                       in_training_mode:False,
                       LOSS:train_loss, 
                       EpisodeReward:episodic_reward,
                       EpisodeScore:episodic_score,
                       EpisodeEpsilon:epsilonn} )
        file_writer.add_summary(mrg_summary, i)

        kappa_val = train_loss_net * regular_ratio / train_loss_reg        

        print(logdir,'Episode', i, 'Reward', episodic_reward, 'Score', episodic_score, 'epsilon', epsilonn )
        print('mean loss', train_loss, 'n', train_loss_n )        
        print('loss_net', train_loss_net, 'loss_reg*kappa', 
              kappa_val*train_loss_reg, 'kappa', kappa_val )
                    
        print('learning_rate', learning_rate_current, 'global_step', global_step )
            
        if (i) % show_steps == 0:            
            envS.show(i)
   
        # if (i+1) % 100 == 0:
        #     save_path = saver.save(sess, logdir + "\my_model_" + str(i+1) + ".ckpt");
        
#-----------------------------------------------------------------------------



    
