
from utils import * 




def read_data(data_dir, original_labels, original_features, seq_len, offset, limit_len) : 

  subjects_dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

  all_subjects_files = [] 
  all_labels = [] 
  for subject_dir in subjects_dirs :
      subject_files = [f for f in os.listdir(subject_dir) if f.endswith('.txt') and any(f.startswith(label) for label in original_labels)]
      labels = [get_label(f) for f in subject_files]
      all_subjects_files.extend([os.path.join(subject_dir, f) for f in subject_files])
      all_labels.extend(labels)

  print("Number of files: ", len(all_subjects_files))
  data_batches, data_labels = [], []
  timestamps = [0] * len(original_labels)
  for i in range(len(all_subjects_files)) : 
      data_df = get_data(all_subjects_files[i], as_df = True, label = all_labels[i])
      data_duration = data_df['timestamp'].values[-1] - data_df['timestamp'].values[0]
      timestamps[data_df['label'][0]] += data_duration
      data_df, label_df = data_df[original_features], data_df[['label']]
      print("file:{} has {} samples".format(all_subjects_files[i], len(data_df)))
      data_df, label_df = data_df[:limit_len], label_df[:limit_len]
      print("file:{} has {} samples".format(all_subjects_files[i], len(data_df)))
      data_batches.extend([data_df.iloc[i:i+seq_len] for i in range(0, len(data_df)-seq_len, offset)])
      data_labels.extend([label_df.iloc[i+seq_len] for i in range(0, len(data_df)-seq_len, offset)])


  data_batches = np.array(data_batches)
  data_labels = np.array(data_labels)
  # data_labels = to_categorical(data_labels, num_classes = len(original_labels))

  # shuffle data
  perm = np.random.permutation(len(data_batches))
  data_batches = data_batches[perm]
  data_labels = data_labels[perm]

  return data_batches, data_labels 


def split_data_KD(data_batches, data_labels) : 
  # data_batches shape: (NS, seqlen, n_features)
  # NS: Number of samples
  # data_labels shape: (NS, n_classes) 

  perm = np.random.permutation(len(data_batches))
  data_batches = data_batches[perm]
  data_labels = data_labels[perm]
  
  # split data into train and test
  train_size = int(len(data_batches) * 0.7)
  pub_data_size = int(len(data_batches) * 0.1)
  test_size = len(data_batches) - train_size - pub_data_size

  train_batches = data_batches[:train_size]
  train_labels = data_labels[:train_size]

  pub_data_batches = data_batches[train_size:train_size+pub_data_size]
  pub_data_labels = data_labels[train_size:train_size+pub_data_size]

  test_batches = data_batches[train_size+pub_data_size:]
  test_labels = data_labels[train_size+pub_data_size:]


  # scaling_data = MinMaxScaler()
  # scaled_train_batches = scaling_data.fit_transform(train_batches)
  # scaled_pub_batches = scaling_data.transform(pub_data_batches)
  # scaled_test_batches = scaling_data.transform(test_batches) 

  return (train_batches, train_labels), (pub_data_batches, pub_data_labels), (test_batches, test_labels) 


def get_label(filename) : 
    filename = filename.split(".")[0]
    if filename.startswith('study') :
        return 0
    elif filename.startswith('walk') : 
        return 1
    elif filename.startswith('sleep') : 
        return 2
    elif filename.startswith('idle') : 
        return 3
    else :
        return -1



def get_data (data_dir, label, as_np = False, as_df = False) : 

    experiments = [] 
    data = [] 
    with open(data_dir, 'r') as f : 
        for line in f.readlines() : 
            if line.startswith('month') : 
                if len(data) : 
                    experiments.append(data) 
                    data = [] 
            data.append(line) 
        if len(data) : 
            experiments.append(data) 
    
    if as_np or as_df : 
        np_exps = [] 
        for i, exp in enumerate(experiments) : 
            _, data = put_experiment_data_to_np(exp, label = label)  
            np_exps.append(data) 
        np_exp = np.concatenate(np_exps) 

        if as_df : 
            dataframe_dict = {
                'hr': np_exp[:, 0],
                'gryo_x': np_exp[:, 1], 
                'gyro_y': np_exp[:, 2], 
                'gyro_z': np_exp[:, 3],
                'timestamp': np_exp[:, 4], 
                'label': np_exp[:, 5]}
            df = pd.DataFrame(dataframe_dict)
            return df 
         
        return np_exp 


    return experiments 



def put_experiment_data_to_np(exp, label = None) : 

    def get_first_hr(exp) : 
        for i in range(1, len(exp)):
            line = exp[i]
            vars = line.split(',') 
            if len(vars) == 2 : 
                return int(vars[0])
    hr = get_first_hr(exp) 
    np_data = []
    for i in range(1, len(exp)):
        line = exp[i]
        vars = line.split(',') 
        if len(vars) == 2 : 
            if (int(vars[0])  < 0) : continue 
            hr = (int(vars[0]) + hr) // 2
        elif len(vars) == 4 : 
            gryo_vars = list(map(int, vars[:3])) 
            if label is not None : 
                data = np.array([hr, *gryo_vars, int(vars[3].split('.')[0]), label])
            else : 
                data = np.array([hr, *gryo_vars, int(vars[3].split('.')[0])])
            np_data.append(data)      
    return exp[0], np.array(np_data) 



class HARDataset(Dataset):
  def __init__(self, data, labels, transform=None, target_transform=None):
    self.data = data
    self.labels = labels
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    data = torch.from_numpy(self.data[idx].astype(np.float32))
    label = torch.from_numpy(self.labels[idx].astype(np.float32))
    if self.transform:
        data = self.transform(data)
    if self.target_transform:
        label = self.target_transform(label)
    return data, label