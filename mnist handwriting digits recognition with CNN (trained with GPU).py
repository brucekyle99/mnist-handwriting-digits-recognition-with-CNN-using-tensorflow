import tensorflow as tf
import time


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


def weight_new(shape):
    initial=tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_new(length):
    initial=tf.constant(0.05, shape=length)
    return tf.Variable(initial)

def conv_layer(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x=tf.placeholder(tf.float32, [None, 784])

#真实照片label
y_true=tf.placeholder(tf.float32, [None, 10])

# 定义dropout占位符
keep_prob=tf.placeholder(tf.float32)

# 黑白照片是1个channel
# 把照片从1x784转成28x28
x_image=tf.reshape(x, [-1, 28, 28, 1])

#第一个卷积层尺寸为5x5，一个channel，32个核
w_conv1=weight_new([5, 5, 1, 32])
b_conv1=bias_new([32])

conv1_activated=tf.nn.relu(conv_layer(x_image,w_conv1)+b_conv1)
conv1_pooled=max_pooling(conv1_activated)
#输出为14x14x32

#第二个卷积层尺寸为5x5，64个核
w_conv2=weight_new([5, 5, 32, 64])
b_conv2=bias_new([64])

conv2_activated=tf.nn.relu(conv_layer(conv1_pooled,w_conv2)+b_conv2)
conv2_pooled=max_pooling(conv2_activated)
#输出尺寸变为7x7x64

# 连接全连接层#1
# w_fc1=weight_new([7*7*64, 1024])
w_fc1=weight_new([7*7*64, 128])
# b_fc1=bias_new([1024])
b_fc1=bias_new([128])
conv2_pooled_flat=tf.reshape(conv2_pooled, [-1, 7*7*64])
fc1_output=tf.nn.relu(tf.matmul(conv2_pooled_flat, w_fc1)+b_fc1)

#加入dropout
fc1_output_drop=tf.nn.dropout(fc1_output, keep_prob)

#连接全连接层#2
# w_fc2=weight_new([1024, 10])
# b_fc2=bias_new([10])
w_fc2=weight_new([128, 10])
b_fc2=bias_new([10])
y_predicted=tf.nn.softmax(tf.matmul(fc1_output_drop, w_fc2)+b_fc2)

# 定义损失函数和优化器
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predicted))
train_step=tf.train.AdamOptimizer(0.0001).minimize(loss)
# train_step=tf.train.AdadeltaOptimizer(0.1).minimize(loss)

# 定义如何评判accuracy
right_prediction=tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_true, 1))
accuracy=tf.reduce_mean(tf.cast(right_prediction, tf.float32))


# 初始化
init=tf.global_variables_initializer()


# 开始训练
with tf.Session() as sess:
    sess.run(init)
    #计时开始
    start_time = time.time()

    #用minibatch
    # for epoch in range(30):
    #     # batch=data.train.next_batch(50)
    #     for batch in range(1100):
    #         batch_x, batch_y_true = data.train.next_batch(50)
    #         sess.run(train_step, feed_dict={x: batch_x, y_true: batch_y_true, keep_prob: 0.8})
    #
    #     test_acc = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels, keep_prob: 1.0})
    #     print('epoch ' + str(epoch) + ': test accuracy ' + str(test_acc))

    #不用minibatch, 直接迭代
    for i in range(20000):
        batch_x, batch_y_true = data.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch_x, y_true: batch_y_true, keep_prob: 0.8})

        if i%100==0:
            test_acc = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels, keep_prob: 1.0})
            print('iteration: ' + str(i) + ': test accuracy ' + str(test_acc))


    #计时结束
    end_time = time.time()
    time_dif=(end_time-start_time)/60
    print('total time is:'+str(time_dif)+' mins')
    # print('test accuracy:'+str(accuracy.eval({x: data.test.images, y_true: data.test.labels, keep_prob:1.0})))

'''
#30 epochs, 0.8drop out, fc layer 1024 神经元
epoch 0: test accuracy 0.9626
epoch 1: test accuracy 0.9772
epoch 2: test accuracy 0.9825
epoch 3: test accuracy 0.986
epoch 4: test accuracy 0.9846
epoch 5: test accuracy 0.9884
epoch 6: test accuracy 0.9878
epoch 7: test accuracy 0.9892
epoch 8: test accuracy 0.9893
epoch 9: test accuracy 0.9895
epoch 10: test accuracy 0.9918
epoch 11: test accuracy 0.9914
epoch 12: test accuracy 0.9905
epoch 13: test accuracy 0.9907
epoch 14: test accuracy 0.9914
epoch 15: test accuracy 0.9911
epoch 16: test accuracy 0.992
epoch 17: test accuracy 0.9927
epoch 18: test accuracy 0.9909
epoch 19: test accuracy 0.9917
epoch 20: test accuracy 0.9924
epoch 21: test accuracy 0.992
epoch 22: test accuracy 0.9913
epoch 23: test accuracy 0.9916
epoch 24: test accuracy 0.9913
epoch 25: test accuracy 0.9916
epoch 26: test accuracy 0.9918
epoch 27: test accuracy 0.9916
epoch 28: test accuracy 0.9904
epoch 29: test accuracy 0.9912
total time is:3.5239071011543275 mins
'''

'''
30 epochs, 0.8drop out, 128 fc layer 神经元
epoch 0: test accuracy 0.9281
epoch 1: test accuracy 0.9508
epoch 2: test accuracy 0.9625
epoch 3: test accuracy 0.9715
epoch 4: test accuracy 0.9743
epoch 5: test accuracy 0.9784
epoch 6: test accuracy 0.9794
epoch 7: test accuracy 0.9814
epoch 8: test accuracy 0.9851
epoch 9: test accuracy 0.9855
epoch 10: test accuracy 0.9846
epoch 11: test accuracy 0.9864
epoch 12: test accuracy 0.9879
epoch 13: test accuracy 0.9868
epoch 14: test accuracy 0.9879
epoch 15: test accuracy 0.9888
epoch 16: test accuracy 0.9885
epoch 17: test accuracy 0.9884
epoch 18: test accuracy 0.9901
epoch 19: test accuracy 0.989
epoch 20: test accuracy 0.9894
epoch 21: test accuracy 0.9909
epoch 22: test accuracy 0.9903
epoch 23: test accuracy 0.9896
epoch 24: test accuracy 0.9901
epoch 25: test accuracy 0.9907
epoch 26: test accuracy 0.9902
epoch 27: test accuracy 0.9914
epoch 28: test accuracy 0.9907
epoch 29: test accuracy 0.9908
total time is:3.0051090836524965 mins
'''

'''
不用minibatch,直接迭代， fc 128
iteration: 0: test accuracy 0.1026
iteration: 100: test accuracy 0.3889
iteration: 200: test accuracy 0.7499
iteration: 300: test accuracy 0.8114
iteration: 400: test accuracy 0.8294
iteration: 500: test accuracy 0.8359
iteration: 600: test accuracy 0.8457
iteration: 700: test accuracy 0.849
iteration: 800: test accuracy 0.8498
iteration: 900: test accuracy 0.8533
iteration: 1000: test accuracy 0.8572
iteration: 1100: test accuracy 0.855
iteration: 1200: test accuracy 0.9074
iteration: 1300: test accuracy 0.932
iteration: 1400: test accuracy 0.9408
iteration: 1500: test accuracy 0.9463
iteration: 1600: test accuracy 0.9458
iteration: 1700: test accuracy 0.9451
iteration: 1800: test accuracy 0.9521
iteration: 1900: test accuracy 0.9526
iteration: 2000: test accuracy 0.9544
iteration: 2100: test accuracy 0.9567
iteration: 2200: test accuracy 0.9581
iteration: 2300: test accuracy 0.9603
iteration: 2400: test accuracy 0.9602
iteration: 2500: test accuracy 0.9621
iteration: 2600: test accuracy 0.9618
iteration: 2700: test accuracy 0.9617
iteration: 2800: test accuracy 0.9637
iteration: 2900: test accuracy 0.9672
iteration: 3000: test accuracy 0.9682
iteration: 3100: test accuracy 0.9693
iteration: 3200: test accuracy 0.9681
iteration: 3300: test accuracy 0.9685
iteration: 3400: test accuracy 0.9686
iteration: 3500: test accuracy 0.972
iteration: 3600: test accuracy 0.9724
iteration: 3700: test accuracy 0.9726
iteration: 3800: test accuracy 0.9728
iteration: 3900: test accuracy 0.9733
iteration: 4000: test accuracy 0.9707
iteration: 4100: test accuracy 0.9724
iteration: 4200: test accuracy 0.975
iteration: 4300: test accuracy 0.9714
iteration: 4400: test accuracy 0.9753
iteration: 4500: test accuracy 0.9745
iteration: 4600: test accuracy 0.9757
iteration: 4700: test accuracy 0.9765
iteration: 4800: test accuracy 0.9783
iteration: 4900: test accuracy 0.9782
iteration: 5000: test accuracy 0.9776
iteration: 5100: test accuracy 0.978
iteration: 5200: test accuracy 0.9794
iteration: 5300: test accuracy 0.9774
iteration: 5400: test accuracy 0.9783
iteration: 5500: test accuracy 0.9791
iteration: 5600: test accuracy 0.9775
iteration: 5700: test accuracy 0.9797
iteration: 5800: test accuracy 0.9805
iteration: 5900: test accuracy 0.9807
iteration: 6000: test accuracy 0.9811
iteration: 6100: test accuracy 0.9803
iteration: 6200: test accuracy 0.981
iteration: 6300: test accuracy 0.9815
iteration: 6400: test accuracy 0.9806
iteration: 6500: test accuracy 0.9798
iteration: 6600: test accuracy 0.9808
iteration: 6700: test accuracy 0.9802
iteration: 6800: test accuracy 0.9817
iteration: 6900: test accuracy 0.9825
iteration: 7000: test accuracy 0.9812
iteration: 7100: test accuracy 0.9829
iteration: 7200: test accuracy 0.9839
iteration: 7300: test accuracy 0.9823
iteration: 7400: test accuracy 0.983
iteration: 7500: test accuracy 0.9824
iteration: 7600: test accuracy 0.9829
iteration: 7700: test accuracy 0.9819
iteration: 7800: test accuracy 0.9825
iteration: 7900: test accuracy 0.9809
iteration: 8000: test accuracy 0.9837
iteration: 8100: test accuracy 0.9819
iteration: 8200: test accuracy 0.9825
iteration: 8300: test accuracy 0.9849
iteration: 8400: test accuracy 0.9827
iteration: 8500: test accuracy 0.9842
iteration: 8600: test accuracy 0.9832
iteration: 8700: test accuracy 0.9839
iteration: 8800: test accuracy 0.9842
iteration: 8900: test accuracy 0.9843
iteration: 9000: test accuracy 0.984
iteration: 9100: test accuracy 0.9823
iteration: 9200: test accuracy 0.9851
iteration: 9300: test accuracy 0.9853
iteration: 9400: test accuracy 0.9841
iteration: 9500: test accuracy 0.9837
iteration: 9600: test accuracy 0.9855
iteration: 9700: test accuracy 0.9829
iteration: 9800: test accuracy 0.9843
iteration: 9900: test accuracy 0.9846
iteration: 10000: test accuracy 0.9868
iteration: 10100: test accuracy 0.9865
iteration: 10200: test accuracy 0.9862
iteration: 10300: test accuracy 0.9868
iteration: 10400: test accuracy 0.9856
iteration: 10500: test accuracy 0.9848
iteration: 10600: test accuracy 0.9874
iteration: 10700: test accuracy 0.9853
iteration: 10800: test accuracy 0.9852
iteration: 10900: test accuracy 0.9857
iteration: 11000: test accuracy 0.9863
iteration: 11100: test accuracy 0.9854
iteration: 11200: test accuracy 0.9878
iteration: 11300: test accuracy 0.9862
iteration: 11400: test accuracy 0.9871
iteration: 11500: test accuracy 0.9863
iteration: 11600: test accuracy 0.9861
iteration: 11700: test accuracy 0.9847
iteration: 11800: test accuracy 0.9876
iteration: 11900: test accuracy 0.9871
iteration: 12000: test accuracy 0.9878
iteration: 12100: test accuracy 0.9869
iteration: 12200: test accuracy 0.9862
iteration: 12300: test accuracy 0.9875
iteration: 12400: test accuracy 0.9884
iteration: 12500: test accuracy 0.987
iteration: 12600: test accuracy 0.9873
iteration: 12700: test accuracy 0.9881
iteration: 12800: test accuracy 0.9869
iteration: 12900: test accuracy 0.9873
iteration: 13000: test accuracy 0.9871
iteration: 13100: test accuracy 0.9878
iteration: 13200: test accuracy 0.9889
iteration: 13300: test accuracy 0.9882
iteration: 13400: test accuracy 0.9882
iteration: 13500: test accuracy 0.9868
iteration: 13600: test accuracy 0.9879
iteration: 13700: test accuracy 0.9889
iteration: 13800: test accuracy 0.9887
iteration: 13900: test accuracy 0.9897
iteration: 14000: test accuracy 0.9892
iteration: 14100: test accuracy 0.9894
iteration: 14200: test accuracy 0.9883
iteration: 14300: test accuracy 0.9884
iteration: 14400: test accuracy 0.9877
iteration: 14500: test accuracy 0.9887
iteration: 14600: test accuracy 0.9884
iteration: 14700: test accuracy 0.9901
iteration: 14800: test accuracy 0.9891
iteration: 14900: test accuracy 0.9899
iteration: 15000: test accuracy 0.9895
iteration: 15100: test accuracy 0.9886
iteration: 15200: test accuracy 0.988
iteration: 15300: test accuracy 0.9889
iteration: 15400: test accuracy 0.9888
iteration: 15500: test accuracy 0.9888
iteration: 15600: test accuracy 0.9898
iteration: 15700: test accuracy 0.9885
iteration: 15800: test accuracy 0.9886
iteration: 15900: test accuracy 0.9895
iteration: 16000: test accuracy 0.9888
iteration: 16100: test accuracy 0.9893
iteration: 16200: test accuracy 0.9895
iteration: 16300: test accuracy 0.99
iteration: 16400: test accuracy 0.9887
iteration: 16500: test accuracy 0.99
iteration: 16600: test accuracy 0.9899
iteration: 16700: test accuracy 0.9894
iteration: 16800: test accuracy 0.9892
iteration: 16900: test accuracy 0.9894
iteration: 17000: test accuracy 0.99
iteration: 17100: test accuracy 0.9899
iteration: 17200: test accuracy 0.9898
iteration: 17300: test accuracy 0.9903
iteration: 17400: test accuracy 0.9892
iteration: 17500: test accuracy 0.9893
iteration: 17600: test accuracy 0.9889
iteration: 17700: test accuracy 0.9897
iteration: 17800: test accuracy 0.9903
iteration: 17900: test accuracy 0.9901
iteration: 18000: test accuracy 0.9901
iteration: 18100: test accuracy 0.9887
iteration: 18200: test accuracy 0.9898
iteration: 18300: test accuracy 0.9888
iteration: 18400: test accuracy 0.99
iteration: 18500: test accuracy 0.991
iteration: 18600: test accuracy 0.9897
iteration: 18700: test accuracy 0.9914
iteration: 18800: test accuracy 0.9899
iteration: 18900: test accuracy 0.9912
iteration: 19000: test accuracy 0.9905
iteration: 19100: test accuracy 0.9904
iteration: 19200: test accuracy 0.9909
iteration: 19300: test accuracy 0.9901
iteration: 19400: test accuracy 0.9906
iteration: 19500: test accuracy 0.9907
iteration: 19600: test accuracy 0.9896
iteration: 19700: test accuracy 0.9902
iteration: 19800: test accuracy 0.9898
iteration: 19900: test accuracy 0.9909
total time is:2.2654181838035585 mins

'''
























