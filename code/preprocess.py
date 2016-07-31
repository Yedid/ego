import os
import numpy as np

# Use test sequences recorded on the same day.
# False means use sequences recorded at least a week later.
next_week = False
# Optical flow frame rate is 15 FPS.
fps = 15
# Length of subsequence use for training in seconds. 
duration = 4
fpf = fps * duration
# Number of optical flow features per frame
# Number of cells (5 rows * 10 cols) * 2 (x and y)
n_features = 100
# Compressive root is a heuristic that compressed the feature by their square root.
compressive_root = True
# Subtract the mean value from each OF component. Simulates simple 2D stabilization.
do_stabilize = False 
# Load dataset from directory tree.
def load_dataset(dir_name):
  dataset = {}
  user_files = os.listdir(dir_name)
  for user_id in user_files:
    dataset[user_id] = []
    seq_files = os.listdir('%s/%s/'%(dir_name,user_id))
    for seq_id in seq_files:
      seq_data = np.loadtxt('%s/%s/%s'%(dir_name,user_id,seq_id), delimiter = ',').T
      dataset[user_id].append(seq_data)
  return dataset 
# Convert all the sequences of a user into a list of fixed length (4s) subsequences.
def seq2examples(seq, id, fpf):
  n_examples = max(0,seq.shape[0]*2/fpf -1)
  data = np.zeros((n_examples,fpf,n_features))
  labels = np.ones((n_examples)) * id
  for i in range(n_examples):
    data[i,:,:] = seq[i*fpf/2 : i*fpf/2 + fpf,:]
  return data, labels
# Convert dataset into a list of fixed length (4s) subsequences for each person.
def dataset2examples(dataset):
  data = np.zeros((0,fpf,n_features))
  labels = np.zeros((0))
  for person_id in dataset.keys():
    person_data = dataset[person_id]
    for seq in person_data:
      seq_data, seq_labels = seq2examples(seq, int(person_id), fpf)
      data = np.concatenate((data, seq_data), axis = 0)
      labels = np.concatenate((labels, seq_labels), axis = 0)
  return data, labels
# Find user with most examples.
def get_mode(my_array):
  ul = np.unique(my_array)
  max_count = 0
  for l in ul:
    max_count = max(max_count, np.sum(my_array == l))
  return ul, max_count
# Replicate examples so that all users have the same number of examples.
def get_max_data(input_data, input_labels):  
  class_labels, n_examples_per_class = get_mode(input_labels)
  data =  np.zeros((len(class_labels) * n_examples_per_class ,input_data.shape[1], input_data.shape[2]))
  labels = np.zeros((len(class_labels) * n_examples_per_class))
  for (i, label) in enumerate(class_labels):
    idx = np.where(input_labels == label)[0]
    ratio = n_examples_per_class / len(idx)
    myrem =  n_examples_per_class - ratio * len(idx)
    for j in range(ratio):
      data[i*n_examples_per_class+j*len(idx):i*n_examples_per_class+(j+1)*len(idx),:,:] = input_data[idx,:,:]
    data[i*n_examples_per_class+ratio*len(idx):(i+1)*n_examples_per_class,:,:] = input_data[idx[:myrem],:,:]   
    labels[i*n_examples_per_class: (i+1)*n_examples_per_class] = label
  return data, labels
# Clear users who do not appear in both train and test
def clear_orphan_label(train_x, train_y, test_x, test_y):
  labels = np.intersect1d(train_y, test_y)
  train_idx = np.in1d(train_y, labels)
  test_idx = np.in1d(test_y, labels)
  return train_x[train_idx], train_y[train_idx], test_x[test_idx], test_y[test_idx] 

def main():
  # Retrieve datasets.
  black = load_dataset('../datasets/black')
  black_data, black_labels = dataset2examples(black)
  grey = load_dataset('../datasets/grey')
  grey_data, grey_labels = dataset2examples(grey)
  last = load_dataset('../datasets/last')
  last_data, last_labels = dataset2examples(last)
  # Select the dataset to be used.
  # Next week means test sequences recorded at least a 
  # week after the train sequences.
  if not next_week:
    train_data, train_labels = black_data, black_labels
    test_data, test_labels = grey_data, grey_labels
  else:
    train_data, train_labels = black_data, black_labels
    test_data, test_labels = last_data, last_labels
  # Remove labels that do not appear in both train and test data sets.
  train_data, train_labels, test_data, test_labels = clear_orphan_label(train_data, train_labels, test_data, test_labels)  
  # Replicate examples in data poor classes to ensure the dataset in balanced.  
  train_data, train_labels = get_max_data(train_data, train_labels)
  test_data, test_labels = get_max_data(test_data, test_labels)
  # Reshape data so that X and Y components are in different channels.
  train_data = np.reshape(train_data,(train_data.shape[0],train_data.shape[1],train_data.shape[2]/2,2))
  test_data = np.reshape(test_data,(test_data.shape[0],test_data.shape[1],test_data.shape[2]/2,2))
  # Subtract the mean motion from each OF component.
  # Simulates simple 2D stabilization.
  if do_stabilize:
    train_data -= train_data.mean(axis=2)[:,:,None,:]
    test_data -= test_data.mean(axis=2)[:,:,None,:]
  # Subtract the mean.
  mu = np.mean(train_data, axis = 0)[None,:,:,:]
  train_data -= mu
  test_data -= mu
  # Root compress the data. Empirically shown to work, probably due to anomaly supression.
  if compressive_root:
    train_data = np.sign(train_data)*np.sqrt(np.abs(train_data))
    test_data = np.sign(test_data)*np.sqrt(np.abs(test_data))
  # Shuffle train examples.
  rp = np.random.permutation(train_data.shape[0])
  train_data = train_data[rp,:,:,:]
  train_labels = train_labels[rp]
  # Save datasets.
  np.save('../npy/train_X',train_data)
  np.save('../npy/train_y',train_labels)
  np.save('../npy/test_X',test_data)
  np.save('../npy/test_y',test_labels)

if __name__ == "__main__":
  main()