
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

class HypothesisType(enum.Enum):
   SymbolTx = 1

BG_SYMBOL = 0
NUM_SYMBOLS = 10

def get_symbol_cmap_and_norm():
  cmap = colors.ListedColormap(
      ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
       '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
  norm = colors.Normalize(vmin=0, vmax=9)
  return cmap, norm

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
  cmap, norm = get_symbol_cmap_and_norm()
  input_matrix = task[train_or_test][i][input_or_output]
  ax.imshow(input_matrix, cmap=cmap, norm=norm)
  ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
  ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
  ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_title(train_or_test + ' '+input_or_output)
    
def plot_ans(ans,ax):
  cmap, norm = get_symbol_cmap_and_norm()
  ax.imshow(ans, cmap=cmap, norm=norm)
  ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
  ax.set_yticks([x-0.5 for x in range(1+len(ans))])
  ax.set_xticks([x-0.5 for x in range(1+len(ans[0]))])     
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_title('Hypothesis')

def plot_task(task, ans=None):
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
  num_subplots = 2
  if ans is not None:
    num_subplots = 3
  fig, axs = plt.subplots(num_subplots, num_test, figsize=(3*num_test,3*num_subplots))
  if num_test==1: 
    plot_one(task,axs[0],0,'test','input')
    plot_one(task,axs[1],0,'test','output')     
  else:
    for i in range(num_test):      
      plot_one(task,axs[0,i],i,'test','input')
      plot_one(task,axs[1,i],i,'test','output')  
  if ans is not None:
    plot_ans(ans,axs[2])
  plt.tight_layout()
  plt.show() 

def plot_components(symbols, component_labels, bb_image):
  # https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.subplot.html
  # Either a 3-digit integer or three separate integers describing the position of the subplot. 
  # If the three integers are nrows, ncols, and index in order, the subplot will take the index
  # position on a grid with nrows rows and ncols columns. index starts at 1 in the upper 
  # left corner and increases to the right.
  cmap, norm = get_symbol_cmap_and_norm()
  plt.figure(figsize=(9, 3.5))
  ax1 = plt.subplot(131)
  plt.imshow(symbols, cmap=cmap, norm=norm)
  ax1.title.set_text('symbols')
  plt.axis('off')
  ax2 = plt.subplot(132)
  plt.imshow(component_labels, cmap='nipy_spectral')
  ax2.title.set_text('all labels')
  plt.axis('off')
  ax3 = plt.subplot(133)
  plt.imshow(bb_image, cmap='nipy_spectral')
  ax3.title.set_text('bounding boxes')
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
    return OutputShapeType.Constant, None
  elif input_shape:
    return OutputShapeType.Input, None
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

def find_density(symbols):
  h = symbols.shape[0]
  w = symbols.shape[1]
  mass = 0.0
  for y in range(0,h):
    for x in range(0,w):
      #is_edge = 0.0
      s = symbols[y][x]
      if s != 0:
        mass += 1
  area = h * w
  density = mass / area
  return density

def find_bounding_boxes(component_labels, num_labels):

  bounding_boxes = {}

  bb_image = np.zeros(component_labels.shape)
  h = component_labels.shape[0]
  w = component_labels.shape[1]
  mass = 0.0
  symbols = []
  for y in range(0,h):
    for x in range(0,w):
      label_value = component_labels[y][x]
      # if label_value == 0:
      # print('has bg')
      # has_background = True
      if label_value in bounding_boxes.keys():
        bounding_box = bounding_boxes[label_value]
        x_min = bounding_box[0]
        y_min = bounding_box[1]
        x_max = bounding_box[2]
        y_max = bounding_box[3]

        x_min = min(x,x_min)
        y_min = min(y,y_min)
        x_max = max(x,x_max)
        y_max = max(y,y_max)

        bounding_box[0] = x_min
        bounding_box[1] = y_min
        bounding_box[2] = x_max
        bounding_box[3] = y_max
      else:
        symbols.append(label_value)
        bounding_box = [x,y,x,y]
        bounding_boxes[label_value] = bounding_box

  # if has_background:
  #   num_labels += 1
  print('all BBs ', bounding_boxes)
  num_symbols = len(symbols)
  for i in range(0, num_symbols):
    label = symbols[i]
    if label == 0:
      continue  # don't draw
    bounding_box = bounding_boxes[label]
    #print('bb of label', label, bounding_box)
    x_min = bounding_box[0]
    y_min = bounding_box[1]
    x_max = bounding_box[2]
    y_max = bounding_box[3]
    bw = x_max - x_min +1
    bh = y_max - y_min +1
    for x in range(0,bw):
      bb_image[y_min][x+x_min] = 1.0
      bb_image[y_max][x+x_min] = 1.0
    for y in range(0,bh):
      bb_image[y+y_min][x_min] = 1.0
      bb_image[y+y_min][x_max] = 1.0

  return bounding_boxes, bb_image

def find_symmetry(image):
  plt.figure()
  plt.imshow(image, cmap='gray')
  plt.tight_layout()
  plt.show()

class Hypothesis:
  def __init__(self):
    pass

  def apply(self, example_input, output_shape_type, output_shape):
    output = np.zeros(output_shape)
    return output

class SymbolTxHypo(Hypothesis):
  def __init__(self, s1, s2):
    self.s1 = s1
    self.s2 = s2

  def apply(self, example_input, output_shape_type, output_shape):
    if output_shape_type != OutputShapeType.Input:
      print('shape mismatch')
      return

    output = np.zeros(example_input.shape)
    h = example_input.shape[0]
    w = example_input.shape[1]
    for y in range(0,h):
      for x in range(0,w):
        s1 = example_input[y][x]
        s2 = s1
        if s1 == float(self.s1):
          #print('$$$', self.s2)
          s2 = int(self.s2)
        output[y][x] = s2
    return output

def evaluate_output(example_output, hypo_output):
  errors = 0
  h = example_output.shape[0]
  w = example_output.shape[1]
  if hypo_output is None:
    return h*w  # all wrong

  for y in range(0,h):
    for x in range(0,w):
      s1 = example_output[y][x]
      s2 = hypo_output[y][x]
      if s1 != s2:
        errors += 1
  return errors

def find_hypothesis(train, test, output_shape_type, output_shape):
  hypo_type = HypothesisType.SymbolTx

  # Build hypotheses
  hypos = []

  for s1 in range(0,NUM_SYMBOLS):
    for s2 in range(0,NUM_SYMBOLS):
      hypo = SymbolTxHypo(s1, s2)
      #print('*******************',hypo.s1, hypo.s2)
      hypos.append(hypo)

  # Evaluate hypotheses
  num_hypotheses = len(hypos)
  num_train_examples = len(train)
  best_hypo_errors = 0
  best_hypo = None

  for h in range(0,num_hypotheses):
    hypo = hypos[h]
    hypo_errors = 0

    for e in range(0,num_train_examples):
      example_input = train[e]['input']
      example_input = np.array(example_input)

      example_output = train[e]['output']
      example_output = np.array(example_output)

      hypo_output = hypo.apply(example_input, output_shape_type, output_shape)
      #print('hypo:',hypo_output)
      errors = evaluate_output(example_output, hypo_output)
      hypo_errors += errors
      #print('- - -> Train Eg Errors ', errors)

    if (best_hypo is None) or (hypo_errors < best_hypo_errors):
      best_hypo = hypo
      best_hypo_errors = hypo_errors

    #print('-----> Train Errors for H', h,'are',hypo_errors, 'best=', best_hypo_errors)

  # Keep the simplest hypo that has no error, or failing that, with min error
  test_input = test[0]['input']
  test_input = np.array(test_input)
  test_output = test[0]['output']
  test_output = np.array(test_output)
  hypo_output = best_hypo.apply(test_input, output_shape_type, output_shape)
  test_errors = evaluate_output(test_output, hypo_output)
  print('=====> Test Errors ', test_errors)
  return hypo_output, test_errors

def process_file(file_path, do_plot=False, do_hypo=False, e_sel=None):
  print('Reading file: ', file_path)
  with open(file_path) as json_file:
    js = json.load(json_file)
    #print(js)
  if do_plot:
    plot_task(js)

  # https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
  example = 2
  train = js['train']
  test = js['test']
  num_train_examples = len(train)
  input_shapes = []
  output_shapes = []
  sum_densities = 0.0
  for e in range(0,num_train_examples):
    if e_sel is not None:
      if e != e_abeL:
        continue
    example_input = train[e]['input']
    #print('example_input', example_input)
    example_input = np.array(example_input)
    input_shapes.append(example_input.shape)
    print('example_input.shape', example_input.shape)

    density = find_density(example_input)
    sum_densities += density
    
    #find_symmetry(example_input)
    #edges = symbol_2_edges(example_input)
    #find_symmetry(edges)

    example_output = train[e]['output']
    #print('example_output', example_output)
    example_output = np.array(example_output)
    print('output.shape', example_output.shape)
    output_shapes.append(example_output.shape)

    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label
    #connectivity = 1
    connectivity = 2
    # TODO separate objects by colour
    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label
    # "background: Consider all pixels with this value as background pixels, and label them as 0."
    #all_labels = measure.label(example_input, connectivity=connectivity)
    # Returns: Labeled array, where all connected regions are assigned the same integer value.
    component_labels, num_labels = measure.label(example_input, connectivity=connectivity, background=0, return_num=True)
    bounding_boxes, bb_image = find_bounding_boxes(component_labels, num_labels)
    if False: #do_plot:
      plot_components(example_input, component_labels, bb_image)

  output_shape_type, output_shape = find_output_shape(input_shapes,output_shapes)
  mean_density = sum_densities / float(num_train_examples)

  correct = False
  if do_hypo:
    if output_shape_type != OutputShapeType.Unknown:
      hypo_output, test_errors = find_hypothesis(train, test, output_shape_type, output_shape)
      #plot_task(js,hypo_output)
      if test_errors == 0:
        correct = True

  return output_shape_type, mean_density, correct

# Priors:
# Connected components
# Channels == symbol value
# 0 = background = free space
# Bounding boxes
# Area of components
# Centroids of components
# https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines/notebook
print('hello world')

file_path = './data/training/b1948b0a.json'
#file_path = './data/training/ff28f65a.json'
#data_dir = sys.argv[1]
data_dir = './data/training'
files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
num_files = len(files)
num_constant = 0
num_input = 0
num_unknown = 0
num_correct = 0
densities = []
density_threshold = 0.7
do_hypo = True

for i in range(0, num_files):
  file_name = files[i]
  file_path = os.path.join(data_dir,file_name)
  output_shape_type, density, correct = process_file(file_path, do_hypo=do_hypo)
  densities.append(density)
  if correct:
    num_correct += 1
  # if density >= density_threshold:
  #   process_file(file_path, do_plot=True, do_hypo=do_hypo)  # debug it

  if output_shape_type == OutputShapeType.Constant:
    num_constant += 1
  elif output_shape_type == OutputShapeType.Input:
    num_input += 1   
  else:
    num_unknown += 1

density_bins = 30
plt.hist(densities, bins = density_bins)
plt.show()

total = num_constant + num_input + num_unknown
print('Constant:', num_constant, get_percentage(num_constant, total),'%')
print('Input:', num_input, get_percentage(num_input, total),'%')
print('Unknown:', num_unknown, get_percentage(num_unknown, total),'%')

print('Correct:', num_correct, get_percentage(num_correct, total),'%')
