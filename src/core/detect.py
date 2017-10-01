import tensorflow as tf
from model import P_Net,R_Net,O_Net
import os


def create_mtcnn(sess, model_path, data_batch_size):
    if not model_path:
        model_path, _ = os.path.split(os.path.realpath(__file__))

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    p_cls_prob, p_bbox_pred = P_Net(tf.placeholder(tf.float32,[data_batch_size,12,12,3]), training=False)
    o_cls_prob, o_bbox_pred = O_Net(tf.placeholder(tf.float32,[data_batch_size,48,48,3]), training=False)
    r_cls_prob, r_bbox_pred = R_Net(tf.placeholder(tf.float32,[data_batch_size,24,24,3]), training=False)

    pnet_fun = lambda img_batch: sess.run([p_cls_prob,p_bbox_pred], feed_dict={tf.placeholder(tf.float32,[data_batch_size,12,12,3]): img_batch})
    rnet_fun = lambda img_batch: sess.run([o_cls_prob,o_bbox_pred], feed_dict={tf.placeholder(tf.float32,[data_batch_size,24,24,3]): img_batch})
    onet_fun = lambda img_batch: sess.run([r_cls_prob,r_bbox_pred], feed_dict={tf.placeholder(tf.float32,[data_batch_size,48,48,3]): img_batch})
    return pnet_fun, rnet_fun, onet_fun




