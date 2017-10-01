
from core.imagedb import imagedb
from core.imageReader import imageReader
import tensorflow as tf
import datetime
import numpy as np
from core.model import P_Net,O_Net,R_Net
import os

def train_model(base_lr,loss,data_num,batch_size):
    lr_factor=0.1
    global_step = tf.Variable(0, trainable=False)
    lr_epoch = [8,14]
    boundaries=[int(epoch*data_num/batch_size) for epoch in lr_epoch]
    lr_values=[base_lr*(lr_factor**x) for x in range(0,len(lr_epoch)+1)]
    lr_op=tf.train.piecewise_constant(global_step, boundaries, lr_values)

    optimizer=tf.train.MomentumOptimizer(lr_op,0.9)
    train_op=optimizer.minimize(loss,global_step)
    return train_op,lr_op

def compute_accuracy(cls_prob,label):
    keep=(label>=0)
    pred=np.zeros_like(cls_prob)
    pred[cls_prob>0.5]=1
    return np.sum(pred[keep]==label[keep])*1.0/np.sum(keep)


def train_p_net(prefix,end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,):

    train_data=imageReader(imdb,12,batch_size,shuffle=True)

    input_image=tf.placeholder(tf.float32,shape=[batch_size,12,12,3],name='input_image')
    label=tf.placeholder(tf.float32,shape=[batch_size],name='label')
    bbox_target=tf.placeholder(tf.float32,shape=[batch_size,4],name='bbox_target')

    cls_prob_op,bbox_pred_op,cls_loss_op,bbox_loss_op=P_Net(input_image,label,bbox_target)

    train_op,lr_op=train_model(base_lr,cls_loss_op+bbox_loss_op,len(imdb),batch_size)

    model_dir=prefix.rsplit('/',1)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    sess=tf.Session()
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()
        accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]
        for batch_idx,(image_x,(label_y,bbox_y))in enumerate(train_data):
            sess.run(train_op,feed_dict={input_image:image_x,label:label_y,bbox_target:bbox_y})
            if batch_idx%frequent==0:
                cls_pred,cls_loss,bbox_loss,lr=sess.run([cls_prob_op,cls_loss_op,bbox_loss_op,lr_op],feed_dict={input_image:image_x,label:label_y,bbox_target:bbox_y})
                accuracy=compute_accuracy(cls_pred,label_y)
                print "%s : Epoch: %d, Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,lr:%f "%(datetime.datetime.now(),cur_epoch,batch_idx,accuracy,cls_loss,bbox_loss,lr)
                accuracy_list.append(accuracy)
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)

        print "Epoch: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f "%(cur_epoch,np.mean(accuracy_list),np.mean(cls_loss_list),np.mean(bbox_loss_list))
        saver.save(sess,prefix,cur_epoch)


def train_o_net(prefix,end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,):

    train_data=imageReader(imdb,48,batch_size,shuffle=True)

    input_image=tf.placeholder(tf.float32,shape=[batch_size,48,48,3],name='input_image')
    label=tf.placeholder(tf.float32,shape=[batch_size],name='label')
    bbox_target=tf.placeholder(tf.float32,shape=[batch_size,4],name='bbox_target')

    cls_prob_op,bbox_pred_op,cls_loss_op,bbox_loss_op=O_Net(input_image,label,bbox_target)

    train_op,lr_op=train_model(base_lr,cls_loss_op+bbox_loss_op,len(imdb),batch_size)

    model_dir=prefix.rsplit('/',1)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    sess=tf.Session()
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()
        accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]
        for batch_idx,(image_x,(label_y,bbox_y))in enumerate(train_data):
            sess.run(train_op,feed_dict={input_image:image_x,label:label_y,bbox_target:bbox_y})
            if batch_idx%frequent==0:
                cls_pred,cls_loss,bbox_loss,lr=sess.run([cls_prob_op,cls_loss_op,bbox_loss_op,lr_op],feed_dict={input_image:image_x,label:label_y,bbox_target:bbox_y})
                accuracy=compute_accuracy(cls_pred,label_y)
                print "%s : Epoch: %d, Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,lr:%f "%(datetime.datetime.now(),cur_epoch,batch_idx,accuracy,cls_loss,bbox_loss,lr)
                accuracy_list.append(accuracy)
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)

        print "Epoch: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f "%(cur_epoch,np.mean(accuracy_list),np.mean(cls_loss_list),np.mean(bbox_loss_list))
        saver.save(sess,prefix,cur_epoch)


def train_r_net(prefix,end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,):

    train_data=imageReader(imdb,24,batch_size,shuffle=True)

    input_image=tf.placeholder(tf.float32,shape=[batch_size,24,24,3],name='input_image')
    label=tf.placeholder(tf.float32,shape=[batch_size],name='label')
    bbox_target=tf.placeholder(tf.float32,shape=[batch_size,4],name='bbox_target')

    cls_prob_op,bbox_pred_op,cls_loss_op,bbox_loss_op=R_Net(input_image,label,bbox_target)

    train_op,lr_op=train_model(base_lr,cls_loss_op+bbox_loss_op,len(imdb),batch_size)

    model_dir=prefix.rsplit('/',1)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    sess=tf.Session()
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()
        accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]
        for batch_idx,(image_x,(label_y,bbox_y))in enumerate(train_data):
            sess.run(train_op,feed_dict={input_image:image_x,label:label_y,bbox_target:bbox_y})
            if batch_idx%frequent==0:
                cls_pred,cls_loss,bbox_loss,lr=sess.run([cls_prob_op,cls_loss_op,bbox_loss_op,lr_op],feed_dict={input_image:image_x,label:label_y,bbox_target:bbox_y})
                accuracy=compute_accuracy(cls_pred,label_y)
                print "%s : Epoch: %d, Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,lr:%f "%(datetime.datetime.now(),cur_epoch,batch_idx,accuracy,cls_loss,bbox_loss,lr)
                accuracy_list.append(accuracy)
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)

        print "Epoch: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f "%(cur_epoch,np.mean(accuracy_list),np.mean(cls_loss_list),np.mean(bbox_loss_list))
        saver.save(sess,prefix,cur_epoch)