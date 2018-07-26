
import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.examples.tutorials.mnist import input_data
line = "======================================================================"


parameter_servers = ["192.168.148.12:2222"]
workers = ["192.168.148.12:2223","192.168.148.12:2224"]
cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker":workers})
tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")
FLAGS = tf.app.flags.FLAGS
server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index, protocol='grpc+gdr')


start_time = time.time()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print("--- read_data: %s seconds ---" % (time.time() - start_time))
print(line)


trainX = np.reshape(mnist.train.images, (-1, 28, 28, 1))
train_lb = mnist.train.labels
testX = np.reshape(mnist.test.images, (-1, 28, 28, 1))
test_lb = mnist.test.labels


start_time = time.time()
#reformat the data so it's not flat
trainX=trainX.reshape(len(trainX),28,28,1)
testX = testX.reshape(len(testX),28,28,1)
print("--- %s seconds ---" % (time.time() - start_time))
print(line)


start_time = time.time()
#get a validation set and remove it from the train set
trainX,valX,train_lb,val_lb=trainX[0:(len(trainX)-500),:,:,:],trainX[(len(trainX)-500):len(trainX),:,:,:],                            train_lb[0:(len(trainX)-500),:],train_lb[(len(trainX)-500):len(trainX),:]
print("--- %s seconds ---" % (time.time() - start_time))
print(line)



start_time = time.time()

#make sure the images are alright
# plt.imshow(trainX.reshape(len(trainX),28,28)[0],cmap="Greys")
# print("--- %s seconds ---" % (time.time() - start_time))


#need to batch the test data because running low on memory
class test_batchs:
    def __init__(self,data):
        self.data = data
        self.batch_index = 0
    def nextBatch(self,batch_size):
        if (batch_size+self.batch_index) > self.data.shape[0]:
            print ("batch sized is messed up")
        batch = self.data[self.batch_index:(self.batch_index+batch_size),:,:,:]
        self.batch_index= self.batch_index+batch_size
        return batch

#set the test batchsize
test_batch_size = 100


#returns accuracy of model
def accuracy(target,predictions):
    return(100.0*np.sum(np.argmax(target,1) == np.argmax(predictions,1))/target.shape[0])




batch_size = 48
map1 = 32
map2 = 64
num_fc1 = 700 #1028
num_fc2 = 10
reduce1x1 = 16
dropout=0.5
graph = tf.Graph()

if FLAGS.job_name == "ps":
    server.join() 
elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
        global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0),trainable=False)   
        with graph.as_default():
            #train data and labels
            X = tf.placeholder(tf.float32,shape=(batch_size,28,28,1))
            y_ = tf.placeholder(tf.float32,shape=(batch_size,10))

            #validation data
            tf_valX = tf.placeholder(tf.float32,shape=(len(valX),28,28,1))

            #test data
            tf_testX=tf.placeholder(tf.float32,shape=(test_batch_size,28,28,1))


            def createWeight(size,Name):
                return tf.Variable(tf.truncated_normal(size, stddev=0.1),
                                  name=Name)


            start_time = time.time()
            def createBias(size,Name):
                return tf.Variable(tf.constant(0.1,shape=size),
                                  name=Name)



            def conv2d_s1(x,W):
                return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


            def max_pool_3x3_s1(x):
                return tf.nn.max_pool(x,ksize=[1,3,3,1],
                                     strides=[1,1,1,1],padding='SAME')

            #Inception Module1
            #
            #follows input
            W_conv1_1x1_1 = createWeight([1,1,1,map1],'W_conv1_1x1_1')
            b_conv1_1x1_1 = createWeight([map1],'b_conv1_1x1_1')

            #follows input
            W_conv1_1x1_2 = createWeight([1,1,1,reduce1x1],'W_conv1_1x1_2')
            b_conv1_1x1_2 = createWeight([reduce1x1],'b_conv1_1x1_2')

            #follows input
            W_conv1_1x1_3 = createWeight([1,1,1,reduce1x1],'W_conv1_1x1_3')
            b_conv1_1x1_3 = createWeight([reduce1x1],'b_conv1_1x1_3')

            #follows 1x1_2
            W_conv1_3x3 = createWeight([3,3,reduce1x1,map1],'W_conv1_3x3')
            b_conv1_3x3 = createWeight([map1],'b_conv1_3x3')

            #follows 1x1_3
            W_conv1_5x5 = createWeight([5,5,reduce1x1,map1],'W_conv1_5x5')
            b_conv1_5x5 = createBias([map1],'b_conv1_5x5')

            #follows max pooling
            W_conv1_1x1_4= createWeight([1,1,1,map1],'W_conv1_1x1_4')
            b_conv1_1x1_4= createWeight([map1],'b_conv1_1x1_4')



            #Inception Module2
            #
            #follows inception1
            W_conv2_1x1_1 = createWeight([1,1,4*map1,map2],'W_conv2_1x1_1')
            b_conv2_1x1_1 = createWeight([map2],'b_conv2_1x1_1')

            #follows inception1
            W_conv2_1x1_2 = createWeight([1,1,4*map1,reduce1x1],'W_conv2_1x1_2')
            b_conv2_1x1_2 = createWeight([reduce1x1],'b_conv2_1x1_2')

            #follows inception1
            W_conv2_1x1_3 = createWeight([1,1,4*map1,reduce1x1],'W_conv2_1x1_3')
            b_conv2_1x1_3 = createWeight([reduce1x1],'b_conv2_1x1_3')

            #follows 1x1_2
            W_conv2_3x3 = createWeight([3,3,reduce1x1,map2],'W_conv2_3x3')
            b_conv2_3x3 = createWeight([map2],'b_conv2_3x3')

            #follows 1x1_3
            W_conv2_5x5 = createWeight([5,5,reduce1x1,map2],'W_conv2_5x5')
            b_conv2_5x5 = createBias([map2],'b_conv2_5x5')

            #follows max pooling
            W_conv2_1x1_4= createWeight([1,1,4*map1,map2],'W_conv2_1x1_4')
            b_conv2_1x1_4= createWeight([map2],'b_conv2_1x1_4')



            #Fully connected layers
            #since padding is same, the feature map with there will be 4 28*28*map2
            W_fc1 = createWeight([28*28*(4*map2),num_fc1],'W_fc1')
            b_fc1 = createBias([num_fc1],'b_fc1')

            W_fc2 = createWeight([num_fc1,num_fc2],'W_fc2')
            b_fc2 = createBias([num_fc2],'b_fc2')

            def model(x,train=True):
                #Inception Module 1

                start_time = time.time()
                conv1_1x1_1 = conv2d_s1(x,W_conv1_1x1_1)+b_conv1_1x1_1
                print("--- conv1_1x1_1:  %s seconds ---" % (time.time() - start_time))
                print(line)


                start_time = time.time()
                conv1_1x1_2 = tf.nn.relu(conv2d_s1(x,W_conv1_1x1_2)+b_conv1_1x1_2)
                print("--- conv1_1x1_2:  %s seconds ---" % (time.time() - start_time))
                print(line)


                start_time = time.time()
                conv1_1x1_3 = tf.nn.relu(conv2d_s1(x,W_conv1_1x1_3)+b_conv1_1x1_3)
                print("--- conv1_1x1_3:  %s seconds ---" % (time.time() - start_time))
                print(line)

                start_time = time.time()
                conv1_3x3 = conv2d_s1(conv1_1x1_2,W_conv1_3x3)+b_conv1_3x3
                print("--- conv1_3x3:  %s seconds ---" % (time.time() - start_time))
                print(line)

                start_time = time.time()
                conv1_5x5 = conv2d_s1(conv1_1x1_3,W_conv1_5x5)+b_conv1_5x5
                print("--- conv1_5x5:  %s seconds ---" % (time.time() - start_time))
                print(line)

                start_time = time.time()
                maxpool1 = max_pool_3x3_s1(x)
                print("--- maxpool1:  %s seconds ---" % (time.time() - start_time))
                print(line)


                start_time = time.time()
                conv1_1x1_4 = conv2d_s1(maxpool1,W_conv1_1x1_4)+b_conv1_1x1_4
                print("--- conv1_1x1_4:  %s seconds ---" % (time.time() - start_time))
                print(line)

                #concatenate all the feature maps and hit them with a relu
                inception1 = tf.nn.relu(tf.concat([conv1_1x1_1,conv1_3x3,conv1_5x5,conv1_1x1_4],3))

                #Inception Module 2
                start_time = time.time()
                conv2_1x1_1 = conv2d_s1(inception1,W_conv2_1x1_1)+b_conv2_1x1_1
                print("--- conv2_1x1_1:  %s seconds ---" % (time.time() - start_time))
                print(line)


                start_time = time.time()
                conv2_1x1_2 = tf.nn.relu(conv2d_s1(inception1,W_conv2_1x1_2)+b_conv2_1x1_2)
                print("--- conv2_1x1_2:  %s seconds ---" % (time.time() - start_time))
                print(line)

                start_time = time.time()
                conv2_1x1_3 = tf.nn.relu(conv2d_s1(inception1,W_conv2_1x1_3)+b_conv2_1x1_3)
                print("--- conv2_1x1_3:  %s seconds ---" % (time.time() - start_time))
                print(line)

                start_time = time.time()
                conv2_3x3 = conv2d_s1(conv2_1x1_2,W_conv2_3x3)+b_conv2_3x3
                print("--- conv2_3x3:  %s seconds ---" % (time.time() - start_time))
                print(line)

                start_time = time.time()
                conv2_5x5 = conv2d_s1(conv2_1x1_3,W_conv2_5x5)+b_conv2_5x5
                print("--- conv2_5x5:  %s seconds ---" % (time.time() - start_time))
                print(line)


                start_time = time.time()
                maxpool2 = max_pool_3x3_s1(inception1)
                print("--- maxpool2:  %s seconds ---" % (time.time() - start_time))
                print(line)

                start_time = time.time()
                conv2_1x1_4 = conv2d_s1(maxpool2,W_conv2_1x1_4)+b_conv2_1x1_4
                print("--- conv2_1x1_4:  %s seconds ---" % (time.time() - start_time))
                print(line)


                #concatenate all the feature maps and hit them with a relu
                start_time = time.time()
                inception2 = tf.nn.relu(tf.concat([conv2_1x1_1,conv2_3x3,conv2_5x5,conv2_1x1_4],3))
                print("--- inception2:  %s seconds ---" % (time.time() - start_time))
                print(line)

                #flatten features for fully connected layer
                start_time = time.time()
                inception2_flat = tf.reshape(inception2,[-1,28*28*4*map2])
                print("--- inception2_flat:  %s seconds ---" % (time.time() - start_time))
                print(line)

                #Fully connected layers
                if train:
                    h_fc1 =tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1),dropout)
                else:
                    h_fc1 = tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1)

                return tf.matmul(h_fc1,W_fc2)+b_fc2

        #     tf.nn.softmax_cross_entropy_with_logits(logits = yPredbyNN, labels=Y)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=model(X),labels=y_))
            opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

            predictions_val = tf.nn.softmax(model(tf_valX,train=False))
            predictions_test = tf.nn.softmax(model(tf_testX,train=False))

            #initialize variable
            init = tf.initialize_all_variables()

            #use to save variables so we can pick up later
            saver = tf.train.Saver()


    #set use_previous=1 to use file_path model
    #set use_previous=0 to start model from scratch
    use_previous = 0


    BATCH_SIZE = 50 

    TRAINING_STEPS = 20000 

    PRINT_EVERY = 100 

    LOG_DIR = "/tmp/log"
    with tf.device('/device:GPU:1'):
        num_steps = 20000

        start_time = time.time()
        config=tf.ConfigProto(log_device_placement=True)
        #maximun alloc gpu 10% of MEM
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        config.gpu_options.allow_growth = True #allocate dynamically
        sess = tf.Session(config = config)

   
    
        with tf.Session(graph=graph) as sess:

                #initialize variables
                sess.run(init)
                print("Model initialized.")

                #use the previous model or don't and initialize variables
                # if use_previous:
                #     saver.restore(sess,file_path)
                #     print("Model restored.")

                #training
                for s in range(num_steps):
                    start_time = time.time()

                    offset = (s*batch_size) % (len(trainX)-batch_size)
                    batch_x,batch_y = trainX[offset:(offset+batch_size),:],train_lb[offset:(offset+batch_size),:]
                    feed_dict={X : batch_x, y_ : batch_y}

                    _,loss_value = sess.run([opt,loss],feed_dict=feed_dict)

                    print("step",s)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print(line)


                    if s%100 == 0:
                        feed_dict = {tf_valX : valX}
                        preds=sess.run(predictions_val,feed_dict=feed_dict)

                        print ("step: "+str(s))
                        print ("validation accuracy: "+str(accuracy(val_lb,preds)))
                        print (" ")
                        print("--- %s seconds ---" % (time.time() - start_time))
                        print(line)

                    #get test accuracy and save model
                    if s == (num_steps-1):
                        #create an array to store the outputs for the test
                        result = np.array([]).reshape(0,10)

                        #use the batches class
                        batch_testX=test_batchs(testX)

                        start_time = time.time()
                        for i in range(len(testX)/test_batch_size):
                            feed_dict = {tf_testX : batch_testX.nextBatch(test_batch_size)}
                            preds=sess.run(predictions_test, feed_dict=feed_dict)
                            result=np.concatenate((result,preds),axis=0)
                        print("--- loop_time %s seconds ---" % (time.time() - start_time))


                        print ("test accuracy: "+str(accuracy(test_lb,result)))

                        # save_path = saver.save(sess,file_path)
                        # print("Model saved.")
        sess.close()       

        

