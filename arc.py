
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

class OutputShapeType(enum.Enum):
   Constant = 1
   Input = 2
   Unknown = 3

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

    example_output = train[e]['output']
    #print('example_output', example_output)
    example_output = np.array(example_output)
    print('output.shape', example_output.shape)
    output_shapes.append(example_output.shape)
    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label
    connectivity = 1
    #connectivity = 2
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

