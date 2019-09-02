def inference_varn(_X, _Y, _dropout, BATCH_SIZE):
    ### video net ####################################################################################################
    with tf.variable_scope('custom') as scope:
        cnt = 0
        conv1_custom = tf.nn.conv3d(_X, get_conv_weight('firstconv1', [1, 7, 7, RGB_CHANNEL, 64]), strides=[1, 1, 2, 2, 1], padding='SAME')
        conv1_custom_bn = tf.layers.batch_normalization(conv1_custom, training=IS_TRAIN)
        conv1_custom_bn_relu = tf.nn.relu(conv1_custom_bn)
        video_layer = tf.nn.max_pool3d(conv1_custom_bn_relu,[1, 2, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer1') as scope:
        b1 = make_block(video_layer, 64, 3, 64, cnt)
        video_layer = b1.infer()
        cnt = b1.cnt
        video_layer = tf.nn.max_pool3d(video_layer, [1, 2, 1, 1, 1], strides=[1, 2, 1, 1, 1], padding='SAME')
    with tf.variable_scope('layer2') as scope:
        b2 = make_block(video_layer, 128, 8, 256, cnt, stride=2)
        video_layer = b2.infer()
        cnt = b2.cnt
        video_layer = tf.nn.max_pool3d(video_layer, [1, 2, 1, 1, 1], strides=[1, 2, 1, 1, 1], padding='SAME')
    with tf.variable_scope('layer3') as scope:
        b3 = make_block(video_layer, 256, 36, 512, cnt, stride=2)
        video_layer = b3.infer()
        cnt = b3.cnt
        video_layer = tf.nn.max_pool3d(video_layer, [1, 2, 1, 1, 1], strides=[1, 2, 1, 1, 1], padding='SAME')
    with tf.variable_scope('full') as scope:
        shape = video_layer.shape.as_list()
        video_layer = tf.reshape(video_layer, shape=[-1, shape[2], shape[3], shape[4]])
    
        video_layer = make_block(video_layer, 512, 3, 1024, cnt, stride=2).infer()
    
        # Caution:make sure avgpool on the input which has the same shape as kernelsize has been setted padding='VALID'
        video_layer = tf.nn.avg_pool(video_layer, [1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID')
    
        video_layer = tf.reshape(video_layer, shape=[-1, 2048])
        if(IS_TRAIN):
            video_layer = tf.nn.dropout(video_layer, keep_prob=0.5)
        else:
            video_layer = tf.nn.dropout(video_layer, keep_prob=1)
    ### video net ####################################################################################################
    
    ### audio net ####################################################################################################
    with tf.variable_scope("audio_layer") as scope:
        conv1_custom = tf.layers.conv1d(inputs=_Y, filters=4, kernel_size=16, strides=1, padding="valid")
        conv1_custom_bn = tf.layers.batch_normalization(conv1_custom, training=IS_TRAIN)
        conv1_custom_bn_relu = tf.nn.relu(conv1_custom_bn)
        audio_layer = tf.layers.MaxPooling1D(pool_size=4, strides=1, padding="same")(conv1_custom_bn_relu)

        audio_layer = tf.reshape(audio_layer, shape=[8, 452])
    ### audio net ####################################################################################################

    ### fusion net ###################################################################################################
    with tf.variable_scope("fusion_layer") as scope:
        fusion_layer = tf.concat([video_layer, audio_layer], 1)
        fusion_layer = tf.layers.dense(inputs=fusion_layer, units=1024)
    ### fusion net ###################################################################################################
    
    ### reason net ###################################################################################################
    with tf.variable_scope("reason_layer") as scope:
        reason_layer = tf.layers.dense(inputs=fusion_layer, units=512, activation=tf.nn.relu)
        # reason_layer = tf.nn.dropout(reason_layer, keep_prob=0.5)
        reason_layer = tf.layers.dense(inputs=reason_layer, units=128, activation=tf.nn.relu)
        # reason_layer = tf.nn.dropout(reason_layer, keep_prob=0.5)
        reason_layer = tf.layers.dense(inputs=reason_layer, units=32, activation=tf.nn.relu)
        # reason_layer = tf.nn.dropout(reason_layer, keep_prob=0.5)
        reason_layer = tf.layers.dense(inputs=reason_layer, units=NUM_CLASS, activation=tf.nn.relu)
        # reason_layer = tf.nn.dropout(reason_layer, keep_prob=0.5)
    ### reason net ###################################################################################################

    ### predict net ###################################################################################################
    with tf.variable_scope("predict_layer") as scope:
        predict_layer = tf.layers.dense(inputs=fusion_layer, units=512, activation=tf.nn.relu)
        # predict_layer = tf.nn.dropout(reason_layer, keep_prob=0.5)
        predict_layer = tf.layers.dense(inputs=predict_layer, units=128, activation=tf.nn.relu)
        # predict_layer = tf.nn.dropout(reason_layer, keep_prob=0.5)
        predict_layer = tf.layers.dense(inputs=predict_layer, units=32, activation=tf.nn.relu)
        # predict_layer = tf.nn.dropout(reason_layer, keep_prob=0.5)
        predict_layer = tf.layers.dense(inputs=predict_layer, units=NUM_CLASS, activation=tf.nn.relu)
        # predict_layer = tf.nn.dropout(reason_layer, keep_prob=0.5)
    ### predict net ###################################################################################################

    return reason_layer, predict_layer