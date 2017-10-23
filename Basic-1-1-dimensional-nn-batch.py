from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
# Parameters
learning_rate = 0.01  #The "length" of the step while descending the gradient
epoch = 500  #How many times the network is trained with all the data
display_step = 100 #We are going to display information each every 10 epoch

# Network Parameters
n_hidden_1 = 6 # 1st layer number of neurons
n_hidden_2 = 6 # 2nd layer number of neurons
num_input = 1 # The input data has 1 dimension
num_classes = 1 # The output has 1 dimension
data_set_size=6000 #The data set size
BATCH_SIZE=100

#We create a simple input data numbers between -6 and 6
def create_data_input(dim,amount):
    matrix=np.ndarray([dim,amount])
    for i in range(dim):
        matrix[:][i]=np.random.randint(-600,600,[1,amount])/100.0
        #matrix[:][i].sort(0)
    return matrix    
        
#We create a simple output data: the sin(x) of the input numbers
def create_data_label(dim,amount,input_data):
    
    matrix=np.ndarray([dim,amount])
    for i in range(dim):
         #matrix[:][i]=input_data[:][i]**2
         matrix[:][i]=np.sin(input_data[:][i])+np.random.randn(dim,amount)/16
    return matrix
    
    
#We create the input data and the output data
input_matrix=create_data_input(num_input,data_set_size).T
output_matrix=create_data_label(num_classes,data_set_size,input_matrix.T).T
#These 2 data set below are created to plot the data progress
input_test=input_matrix.copy()
input_test.sort(0)
output_test=create_data_label(num_classes,data_set_size,input_test.T).T
#output_matrix=output_matrix/float(90000.0)



# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 6 neurons       
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
#prediction =tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.pow(logits - Y, 2)) #Loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op) #This is what we want to minimize, the mean of the elements of the 
#difference between neural_net(X) and Y i.e the predictions minus the label
#optimizer: we need to choose which algorithms are we going to use to optimize the loss function
#train_op: we execute the minimize function (.minimize(loss_op)) given the function to minimize (loss_op) and the optimizer (optimizer)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    results = sess.run(logits, feed_dict={X:input_test})
    plt.plot(input_test,output_test.T[:][0],'b--')
    plt.plot(input_test,results.T[:][0],'ro')
    plt.show()

    for step in range(1, epoch+1):

        #We are going to go through every point of the data set once per each epoch. Ideally we should shuffle the data
        #each time we go through all of it. The more random the better.
        #random_seed=random.randint(1,epoch+1)        
        #random.Random(random_seed).shuffle(input_matrix) #We shuffle the data each time we start a new epoch
        #random.Random(random_seed).shuffle(output_matrix)
        #output_matrix=create_data_label(num_classes,data_set_size,input_matrix.T).T
        #I dont know why if you shuffle the data the outcome is bad, if you know please tell me.
        for start, end in zip(range(0, data_set_size, BATCH_SIZE),
                              range(BATCH_SIZE, data_set_size + 1,BATCH_SIZE)):

            sess.run(train_op, feed_dict={X: input_matrix[start:end],Y: output_matrix[start:end]})        
        
        
        
        
        
#        for f in range(len(output_matrix)):
#            batch_x= input_matrix[f]
#            batch_y= output_matrix[f]
#            #As we are doing an online training (i.e batch_x has 1 data point)
#            #we need to reshape to get a shape=[1,1] or else we will get a shape
#            # equal to [1,] which is not compatible with the shapes we have given to the
#            #tensorflow variables, in particular to the placeholders. The place holders
#            #accept any size in the first coordinate of the shape [any,1]
#            batch_x=batch_x.reshape([1,num_input])
#            batch_y=batch_y.reshape([1,num_classes])
##           #Run optimization op (backprop)
#            #Remember that sess.run(fetches,feed_dict=None,some other param) receives fetches, and
#            # feed_dict as parameters, as long with others that will not be used now.
#            #fetches can be a singleton or 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
#            #feed_dict: A dictionary that maps graph elements to values (described above).
#            #In summary: fetches can be a list (among other containers) of graph_elements and
#            #feed_dict is the data that will be passed to the graph elements when we run the graph.
#            #the idea will be more clear if you see the concepts in an example like the one below.
#            #train_op has as a placeholder the "unit" X and Y, in order to run the graph we need to replace 
#            #these placeholders with some data, these data are found in batch_x and batch_y respectively, and the
#            #key of the dictionary must be the same as the name of the placeholder.
#            #as we initialized the placeholder with [none,1] we can give 1 data point at a time or chunks of data.
#            #in this case, we will give 1 data point at a time, this way of training is called one-step or online training.
#            #If we give chunks of data it would be called batch training.
#            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # Here we show the plot every 10 epochs and the corresponding loss value
        if step % display_step == 0 or step == 1:
            results = sess.run(logits, feed_dict={X:input_test})
            plt.plot(input_test,output_test.T[:][0],'b--')
            plt.plot(input_test,results.T[:][0],'ro')
            plt.show()

            loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
            print("Epoch " + str(step) + ", Epoch Loss= " + "{:.4f}".format(loss) )
    
    #This last part is just to print data outside the domain, to see if the nn is able to extrapolate    
    matrix=np.ndarray([1,40000])
    matrix[:][0]=np.random.randint(-6000,6000,[1,40000])/300.0
    matrix.sort(1)
    resultsfinal = sess.run(logits, feed_dict={X:matrix.T})
    matrix_final=create_data_label(1,40000,matrix).T
    plt.plot(matrix.T,matrix_final.T[:][0],'b-')
    plt.plot(matrix.T,resultsfinal,'ro')
    print("Extrapolation error:{0}".format(sum(np.sqrt((resultsfinal-matrix_final)**2))))
