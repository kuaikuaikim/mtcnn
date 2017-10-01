import numpy as np
import cv2


class imageReader:
    def __init__(self, imdb, im_size, batch_size=128, shuffle=False):

        self.imdb = imdb
        self.batch_size = batch_size
        self.im_size = im_size
        self.shuffle = shuffle

        self.cur = 0
        self.size = len(imdb)
        self.index = np.arange(self.size)
        self.num_classes = 2

        self.batch = None
        self.data = None
        self.label = None

        self.label_names= ['label', 'bbox_target']
        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data,self.label
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = get_minibatch(imdb)
        self.data = data['data']
        self.label = [label[name] for name in self.label_names]


def get_minibatch(imdb):
    # im_size: 12, 24 or 48
    num_images = len(imdb)
    processed_ims = list()
    cls_label = list()
    bbox_reg_target = list()
    for i in range(num_images):
        im = cv2.imread(imdb[i]['image'])
        cls = imdb[i]['label']
        bbox_target = imdb[i]['bbox_target']

        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        im_tensor = im/127.5
        processed_ims.append(im_tensor)
        cls_label.append(cls)
        bbox_reg_target.append(bbox_target)

    im_array = np.asarray(processed_ims)
    label_array = np.array(cls_label)
    bbox_target_array = np.vstack(bbox_reg_target)
    '''
    bbox_reg_weight = np.ones(label_array.shape)
    invalid = np.where(label_array == 0)[0]
    bbox_reg_weight[invalid] = 0
    bbox_reg_weight = np.repeat(bbox_reg_weight, 4, axis=1)
    '''

    data = {'data': im_array}
    label = {'label': label_array,
             'bbox_target': bbox_target_array}

    return data, label