#!encoding: utf-8
import os
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import resnet as nn
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

import ipdb
pdb = ipdb.set_trace


# ###############加入日志
import logging
import logging.handlers
import datetime
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# set all.log
rf_handle = logging.handlers.TimedRotatingFileHandler('logs/all.log',
        when='midnight', interval=10,
                                                   backupCount=7, atTime=datetime.time(0, 0, 0,0 ))
rf_handle.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

# set error.log
f_handler = logging.FileHandler('logs/error.log')
f_handler.setLevel(logging.ERROR)
f_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s")
)
logger.addHandler(rf_handle)
logger.addHandler(f_handler)

logger.info("\n\nlog start...")


BaseDataset = os.environ['BaseDataset']
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path",
                  help="Path to training data.", default=BaseDataset + 'crackDetection')
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc 选择要训练的数据集",
                  default="simple"),
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--hf", dest="horizontal_flips",
                  help="Augment with horizontal flips in training. (Default=true).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips",
                  help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--num_epochs", dest="num_epochs",
                  help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path",
                  help="模型存储在：Output path for weights.",
                  default='./Models/model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path",
        default='./Models',
        help="预训练模型。Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
    parser.error(
        'Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError(
        "Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

# 32
C.num_rois = int(options.num_rois)
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

### 寻找model文件
path = options.input_weight_path
file_ = os.listdir(path)
file_.sort()
options.input_weight_path =os.path.join(path, file_[0])
print('模型输入的名称为： ',options.input_weight_path)


C.model_path = options.output_weight_path
print('模型输出路径：', C.model_path)


if options.input_weight_path:
    C.base_net_weights = options.input_weight_path

#########################
#数据产生
#########################
all_imgs, classes_count, class_mapping = get_data(options.train_path,
        cache=True)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(
    train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')

data_gen_val = data_generators.get_anchor_gt(
    val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

print('tf or th')
if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)


##########################
#模型构建
##########################
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4)) ### C.num_rois=4
#  由gpu能力决定一次处理几个 后面的4是（x,y,w,h)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios) # 9

rpn = nn.rpn(shared_layers, num_anchors)
# 分别是用于后面的分类和回归
print('打印出图片生成用于分类和回归的shape')
print(rpn[0].shape, rpn[1].shape)


# roi pooling layer
# 下面是roi的分类器 输出的是 最后我们需要的cls reg
# C.roi_input 代表每次网络处理几个 根据gpu的能力设定
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(
    classes_count), trainable=True)

# 代表在rpn中的分类和回归模型
model_rpn = Model(img_input, rpn[:2])
# 代表的是roi中的分类和回归模型
model_classifier = Model([img_input, roi_input], classifier)

##########
# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
# 这是总的模型  输入的是rpn_cls rpn_reg roi_cls roi_reg
##########
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

print('预训练模型 loading weights from {}'.format(C.base_net_weights))
model_rpn.load_weights(C.base_net_weights, by_name=True)
model_classifier.load_weights(C.base_net_weights, by_name=True)

optimizer = Adam(lr=1e-4)
optimizer_classifier = Adam(lr=1e-4)

##########
# 分三个模型 model_rpn model_classfier(roi的分类和回归) model_all 进行编译
##########
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(
    num_anchors), losses.rpn_loss_regr(num_anchors)])

model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(
    len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 100
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))  #shape (1000, 5) 5 代表5种loss值
rpn_accuracy_rpn_monitor = []      ### ???

# 每一次epoch清零一次 计算 Mean number of bounding boxes from RPN overlapping ground truth boxes
rpn_accuracy_for_epoch = []   ### ???
start_time = time.time()

#############
#通过文件名找出之前模型的loss值
#############
origin_loss = options.input_weight_path.split('_')[-2]
try:
    best_loss = float(origin_loss)
    print('原模型loss值为：', best_loss)
except:
    print('原模型loss值为无限大')
    best_loss = np.Inf


class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

for epoch_num in range(num_epochs):

    # 展示训练进程bar
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(
                    sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                # 每1000次之后， 平均每张图找到的重叠的框是多少
                print('Average number of 重叠overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')


            #分别为 np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug
            X, Y, img_data = data_gen_train.__next__() ### 每次生成训练的数据

            loss_rpn = model_rpn.train_on_batch(X, Y)

            # list: include 9 anchors to 判断 bg or fg  and 36 anchors to
            # discriminate bboxes
            P_rpn = model_rpn.predict_on_batch(X)

            # R.sahpe (300, 4) 选出300个框
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(
            ), use_regr=True, overlap_thresh=0.7, max_boxes=300)

            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)
            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            # 选择正样本和负样本
            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois/2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(
                        pos_samples, int(C.num_rois/2), replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(
                        neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(
                        neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch(
                [X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num,
                [('rpn_cls',np.mean(losses[:iter_num,0])),
                ('rpn_regr', np.mean(losses[:iter_num, 1])),
                ('detector_cls', np.mean(losses[:iter_num, 2])),
                ('detector_regr', np.mean(losses[:iter_num, 0]))])

            # 一次epoch 结束的时候
            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(
                    sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print(
                        'Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(round(loss_rpn_cls,
                        3)))
                    print('Loss RPN regression: {}'.format(round(loss_rpn_regr, 3)))
                    print('Loss Detector classifier: {}'.format(round(loss_class_cls, 3)))
                    print('Loss Detector regression: {}'.format(round(loss_class_regr, 3)))
                    print('总损失值为: {}'.format(round(loss_class_cls+loss_rpn_cls+loss_rpn_regr+loss_class_regr,
                                3)))

                    spend_time = time.time() - start_time
                    time_m = int(spend_time/60)
                    time_s = int(spend_time%60)
                    print('Elapsed time: {} min {} second'.format(time_m,
                        time_s))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    # 根据loss值改变model name
                    loss_value = str(curr_loss)[:4]
                    loss_ = '_' + loss_value + '_'
                    path_list = C.model_path.split('/')
                    path = 'model_frcnn' + loss_ + '.hdf5'
                    path_list[-1] = path

                    C.model_path = '/'.join(path_list)
                    print(C.model_path)
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(
                            best_loss, curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)

                    # logger
                    logger.info('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    logger.info(
                        'Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    logger.info('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    logger.info('Loss RPN regression: {}'.format(loss_rpn_regr))
                    logger.info('Loss Detector classifier: {}'.format(loss_class_cls))
                    logger.info('Loss Detector regression: {}'.format(loss_class_regr))
                    logger.info('Elapsed time: {}\n'.format(time.time() - start_time))
                    logger.info('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))

                break

        except Exception as e:
            print('     Exception: {}'.format(e))
            continue

print('Training complete, exiting.')
