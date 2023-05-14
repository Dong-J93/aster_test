from __future__ import absolute_import

from PIL import Image
import os
import numpy as np
from collections import OrderedDict
#from scipy.misc import imresize
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from io import BytesIO
from multiprocessing import Pool
import math
import sys

import torch
from torch.nn import functional as F

from . import to_torch, to_numpy
from ..evaluation_metrics.metrics import get_str_list


def recognition_vis(images, preds, targets, scores, dataset, vis_dir):
  images = images.permute(0,2,3,1)
  images = to_numpy(images)
  images = (images * 0.5 + 0.5)*255
  pred_list, targ_list = get_str_list(preds, targets, dataset)
  for id, (image, pred, target, score) in enumerate(zip(images, pred_list, targ_list, scores)):
    if pred.lower() == target.lower():
      flag = 'right'
    else:
      flag = 'error'
    file_name = '{:}_{:}_{:}_{:}_{:.3f}.jpg'.format(flag, id, pred, target, score)
    file_path = os.path.join(vis_dir, file_name)
    image = Image.fromarray(np.uint8(image))
    image.save(file_path)


# save to disk sub process
def _save_plot_pool(vis_image, save_file_path):
  vis_image = Image.fromarray(np.uint8(vis_image))
  vis_image.save(save_file_path)


def stn_vis(raw_images, rectified_images, ctrl_points, preds, targets, real_scores, pred_scores, dataset, vis_dir):
  """
    raw_images: images without rectification
    rectified_images: rectified images with stn
    ctrl_points: predicted ctrl points
    preds: predicted label sequences
    targets: target label sequences
    real_scores: scores of recognition model
    pred_scores: predicted scores by the score branch
    dataset: xxx
    vis_dir: xxx
  """
  if raw_images.ndimension() == 3:
    raw_images = raw_images.unsqueeze(0)
    rectified_images = rectified_images.unsqueeze(0)
  batch_size, _, raw_height, raw_width = raw_images.size()

  # translate the coordinates of ctrlpoints to image size
  ctrl_points = to_numpy(ctrl_points)
  ctrl_points[:,:,0] = ctrl_points[:,:,0] * (raw_width-1)
  ctrl_points[:,:,1] = ctrl_points[:,:,1] * (raw_height-1)
  ctrl_points = ctrl_points.astype(np.int)

  # tensors to pil images
  raw_images = raw_images.permute(0,2,3,1)
  raw_images = to_numpy(raw_images)
  raw_images = (raw_images * 0.5 + 0.5)*255
  rectified_images = rectified_images.permute(0,2,3,1)
  rectified_images = to_numpy(rectified_images)
  rectified_images = (rectified_images * 0.5 + 0.5)*255

  # draw images on canvas
  vis_images = []
  num_sub_plot = 3
  raw_images = raw_images.astype(np.uint8)
  rectified_images = rectified_images.astype(np.uint8)

  pred_list, targ_list = get_str_list(preds, targets, dataset)

  for i in range(batch_size):
    plt.ion()
    fig = plt.figure()
    ax = [fig.add_subplot(num_sub_plot,1,i+1) for i in range(num_sub_plot)]
    for a in ax:
      a.set_xticklabels([])
      a.set_yticklabels([])
      a.axis('off')
    ax[0].imshow(raw_images[i])
    ax[0].scatter(ctrl_points[i,:,0], ctrl_points[i,:,1], marker='+', s=20, c='r')
    ax[1].imshow(rectified_images[i])
    #ax[2].set_title("{}".format(pred_list[i]))
    ax[2].text(x=0.5,y=0.8,s="{}".format(pred_list[i]),transform=ax[2].transAxes,
               fontdict=dict(fontsize=20, color='b',#字体属性设置
                       family='monospace',#字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                       weight='bold',#磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                      ),
                bbox = {'facecolor': '#74C476',  # 填充色
                        'edgecolor': 'b',  # 外框色
                        'alpha': 0.5,  # 框透明度
                        'pad': 8,  # 本文与框周围距离
                        }
              )
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    buffer_ = BytesIO()
    plt.savefig(buffer_, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buffer_.seek(0)
    dataPIL = Image.open(buffer_)
    data = np.asarray(dataPIL).astype(np.uint8)
    buffer_.close()

    vis_images.append(data)

  # save to disk
  if vis_dir is None:
    return vis_images
  else:
    #pred_list, targ_list = get_str_list(preds, targets, dataset)
    file_path_list = []
    for id, (image, pred, target, real_score) in enumerate(zip(vis_images, pred_list, targ_list, real_scores)):
      if pred.lower() == target.lower():
        flag = 'right'
      else:
        flag = 'error'
      if pred_scores is None:
        file_name = '{:}_{:}_{:}_{:}_{:.3f}.png'.format(flag, id, pred, target, real_score)
      else:
        file_name = '{:}_{:}_{:}_{:}_{:.3f}_{:.3f}.png'.format(flag, id, pred, target, real_score, pred_scores[id])
      file_path = os.path.join(vis_dir, file_name)
      file_path_list.append(file_path)

    with Pool(os.cpu_count()) as pool:
      pool.starmap(_save_plot_pool, zip(vis_images, file_path_list))