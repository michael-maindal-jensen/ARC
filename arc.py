
import os
import sys
import csv
import json
import math
import enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from os import listdir
from os.path import isfile, join
from skimage import measure
from skimage import filters
from scipy import ndimage

class OutputShapeType(enum.Enum):
   Constant = 1
   Input = 2
   Unknown = 3

# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def plot_one(task,ax, i,train_or_test,input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)
    

def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    num_train = len(task['train'])
    print('num_train',num_train)
    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
    for i in range(num_train):     
        plot_one(task,axs[0,i],i,'train','input')
        plot_one(task,axs[1,i],i,'train','output')        
    plt.tight_layout()
    plt.show()        
        
    num_test = len(task['test'])
    print('num_test',num_test)
    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))
    if num_test==1: 
        plot_one(task,axs[0],0,'test','input')
        plot_one(task,axs[1],0,'test','output')     
    else:
        for i in range(num_test):      
            plot_one(task,axs[0,i],i,'test','input')
            plot_one(task,axs[1,i],i,'test','output')  
    plt.tight_layout()
    plt.show() 

def plot_components(blobs, all_labels, blobs_labels):
  plt.figure(figsize=(9, 3.5))
  plt.subplot(131)
  plt.imshow(blobs, cmap='gray')
  plt.axis('off')
  plt.subplot(132)
  plt.imshow(all_labels, cmap='nipy_spectral')
  plt.axis('off')
  plt.subplot(133)
  plt.imshow(blobs_labels, cmap='nipy_spectral')
  plt.axis('off')

  plt.tight_layout()
  plt.show()

def find_output_shape(input_shapes,output_shapes):
  num_shapes = len(input_shapes)
  assert(num_shapes > 0)

  # constant shape
  h0 = output_shapes[0][0]
  w0 = output_shapes[0][1]

  # all hypotheses are true until proven false
  constant_shape = True
  input_shape = True
  for i in range(0,num_shapes):
    h = output_shapes[i][0]
    w = output_shapes[i][1]
    if (h != h0) or (w != w0):
      constant_shape = False
    #print('w/h',w,h)
    hi = input_shapes[i][0]
    wi = input_shapes[i][1]
    if (h != hi) or (w != wi):
      input_shape = False

  if constant_shape:
    return OutputShapeType.Input, None
  elif input_shape:
    return OutputShapeType.Constant, None
  return OutputShapeType.Unknown, None

def get_percentage(n, total):
  return 100.0*(float(n)/float(total))

# Colours
# 0 and 5 seem to have special meanings. 0 is background.
# 5 may be some sort of separator structure.
# How preserved is this?
#      0           1         2        3        4
# ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
#      5           6         7          8          9
#  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

def rgb_2_grey(rgb):
  """A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). 
  The smallest possible grid size is 1x1 and the largest is 30x30."""
  # symbol (integer between 0 and 9, which are visualized as colors).

  #rgb_weights = [0.2989, 0.5870, 0.1140]
  rgb_weights = [0.333, 0.333, 0.333]
  grey = np.dot(rgb[...,:3], rgb_weights)
  return grey

def symbol_2_grey(symbols):
  """A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). 
  The smallest possible grid size is 1x1 and the largest is 30x30."""
  # symbol (integer between 0 and 9, which are visualized as colors).

  # all symbol values are equally different. So to convert, make a series of masks.
  # e.g. * --> 1 
  for symbol in range(0,9):
    is_true = 1.0
    is_false = 0.0
    x = symbols.where(symbols == 0, is_true, is_false)

def symbol_2_edges(symbols):
  # 0 1 2
  #  a b
  # 0 1 2 3  
  #  a b c
  h = symbols.shape[0]
  w = symbols.shape[1]
  eh = h + h -1
  ew = w + w -1
  edges = np.zeros((eh,ew))
  for y in range(0,h):
    for x in range(0,w):
      #is_edge = 0.0
      s = symbols[y][x]
      for dy in range(-1,2):
        for dx in range(-1,2):
          y2 = y+dy
          x2 = x+dx
          if (x2 == x) and (y2 == y):
            continue  # Non edge
          
          if (y2 < 0) or (x2 < 0) or (y2 >= h) or (x2 >= w):
            continue  # ignore image edges - non edge

          s2 = symbols[y2][x2]
          if s2 != s:
            #is_edge = 1.0
            #       1 2  <    3*2=6  
            #     1 2  <    2*2=4  
            # 0 1 2 3 4 5 
            #  a b c d e
            # 0123456789 
            ey = y * 2 + dy
            ex = x * 2 + dx
            edges[ey][ex] = 1.0
  return edges

def find_symmetry(image):
  plt.figure()
  plt.imshow(image, cmap='gray')
  plt.tight_layout()
  plt.show()

def process_file(file_path, do_plot=False, e_sel=None):
  print('Reading file: ', file_path)
  with open(file_path) as json_file:
    js = json.load(json_file)
    #print(js)
  if do_plot:
    plot_task(js)

  # https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
  example = 2
  train = js['train']
  num_train_examples = len(train)
  input_shapes = []
  output_shapes = []
  for e in range(0,num_train_examples):
    if e_sel is not None:
      if e != e_sel:
        continue
    example_input = train[e]['input']
    #print('example_input', example_input)
    example_input = np.array(example_input)
    input_shapes.append(example_input.shape)

    find_symmetry(example_input)
    edges = symbol_2_edges(example_input)
    find_symmetry(edges)

    example_output = train[e]['output']
    #print('example_output', example_output)
    example_output = np.array(example_output)
    print('output.shape', example_output.shape)
    output_shapes.append(example_output.shape)
    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label
    connectivity = 1
    #connectivity = 2
    # TODO separate objects by colour
    all_labels = measure.label(example_input, connectivity=connectivity)
    blobs_labels = measure.label(example_input, connectivity=connectivity, background=0)
    if do_plot:
      plot_components(example_input, all_labels, blobs_labels)

  output_shape_type, output_shape = find_output_shape(input_shapes,output_shapes)
  return output_shape_type

# https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines/notebook
print('hello world')

#file_path = './data/training/ff28f65a.json'
#data_dir = sys.argv[1]
data_dir = './data/training'
files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
num_files = len(files)
num_constant = 0
num_input = 0
num_unknown = 0
for i in range(0, num_files):
  file_name = files[i]
  file_path = os.path.join(data_dir,file_name)
  output_shape_type = process_file(file_path)
  if output_shape_type == OutputShapeType.Constant:
    num_constant += 1
  elif output_shape_type == OutputShapeType.Input:
    num_input += 1   
  else:
    num_unknown += 1

total = num_constant + num_input + num_unknown
print('Constant:', num_constant, get_percentage(num_constant, total),'%')
print('Input:', num_input, get_percentage(num_input, total),'%')
print('Unknown:', num_unknown, get_percentage(num_unknown, total),'%')

