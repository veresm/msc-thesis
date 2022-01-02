import numpy as np
import torch

def get_affinity_merge_layer(affinity_merge_layer,
                              dim1, dim2,dim3,dim4):
  if affinity_merge_layer=='default':
    return MergeLayer(dim1,dim2,dim3,dim4)
  # elu
  elif affinity_merge_layer=='extra_layers':
    return AffinityMergeLayer_extra(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_had':
    return AffinityMergeLayer_extra_had(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_sincos':
    return AffinityMergeLayer_extra_sincos(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_extra6':
    return AffinityMergeLayer_extra6(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_extra6_sincoshad':
    return AffinityMergeLayer_extra6_sincoshad(dim1,dim2,dim3,dim4)
  # relu
  elif affinity_merge_layer=='extra_layers_relu':
    return AffinityMergeLayer_extra_relu(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_had_relu':
    return AffinityMergeLayer_extra_had_relu(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_sincos_relu':
    return AffinityMergeLayer_extra_sincos_relu(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_extra6_relu':
    return AffinityMergeLayer_extra6_relu(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_extra6_sincoshad_relu':
    return AffinityMergeLayer_extra6_sincoshad_relu(dim1,dim2,dim3,dim4)
  # tanh
  elif affinity_merge_layer=='extra_layers_tanh':
    return AffinityMergeLayer_extra_tanh(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_had_tanh':
    return AffinityMergeLayer_extra_had_tanh(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_sincos_tanh':
    return AffinityMergeLayer_extra_sincos_tanh(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_extra6_tanh':
    return AffinityMergeLayer_extra6_tanh(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_extra6_sincoshad_tanh':
    return AffinityMergeLayer_extra6_sincoshad_tanh(dim1,dim2,dim3,dim4)
  # sigmoid
  elif affinity_merge_layer=='extra_layers_sigmoid':
    return AffinityMergeLayer_extra_sigmoid(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_had_sigmoid':
    return AffinityMergeLayer_extra_had_sigmoid(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_sincos_sigmoid':
    return AffinityMergeLayer_extra_sincos_sigmoid(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_extra6_sigmoid':
    return AffinityMergeLayer_extra6_sigmoid(dim1,dim2,dim3,dim4)
  elif affinity_merge_layer=='extra_layers_extra6_sincoshad_sigmoid':
    return AffinityMergeLayer_extra6_sincoshad_sigmoid(dim1,dim2,dim3,dim4)
  else:
    print("Load default merging layer")
    return MergeLayer(dim1,dim2,dim3,dim4)

class MergeLayer(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)

#############################################
###################  ELU  ###################
#############################################

class AffinityMergeLayer_extra(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ELU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra_had(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*3, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ELU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, x1*x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra_sincos(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*6, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ELU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, torch.sin(x1), torch.cos(x1), torch.sin(x2), torch.cos(x2)], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra6(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1+dim2, dim3*2)
    self.fc2 = torch.nn.Linear(dim3*2, dim3*4)
    self.fc3 = torch.nn.Linear(dim3*4, dim3*8)
    self.fc4= torch.nn.Linear(dim3*8, dim3*4)
    self.fc5 = torch.nn.Linear(dim3*4, dim3*2)
    self.fc6 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ELU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.tanh(self.fc2(h1))
    h3 = self.tanh(self.fc3(h2))
    h4 = self.tanh(self.fc4(h3))
    h5 = self.relu(self.fc5(h4))
    return self.fc6(h5)

class AffinityMergeLayer_extra6_sincoshad(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*4, dim3*10)
    self.fc2 = torch.nn.Linear(dim3*10, dim3*6)
    self.fc3 = torch.nn.Linear(dim3*6, dim3*4)
    self.fc4= torch.nn.Linear(dim3*4, dim3*2)
    self.fc5 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ELU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, torch.sin(x1*x2), torch.cos(x1*x2)], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.tanh(self.fc2(h1))
    h3 = self.tanh(self.fc3(h2))
    h4 = self.tanh(self.fc4(h3))
    return self.fc5(h4)

#############################################
###################  ReLU  ###################
#############################################

class AffinityMergeLayer_extra_relu(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra_had_relu(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*3, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, x1*x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra_sincos_relu(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*6, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, torch.sin(x1), torch.cos(x1), torch.sin(x2), torch.cos(x2)], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra6_relu(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1+dim2, dim3*2)
    self.fc2 = torch.nn.Linear(dim3*2, dim3*4)
    self.fc3 = torch.nn.Linear(dim3*4, dim3*8)
    self.fc4= torch.nn.Linear(dim3*8, dim3*4)
    self.fc5 = torch.nn.Linear(dim3*4, dim3*2)
    self.fc6 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.tanh(self.fc2(h1))
    h3 = self.tanh(self.fc3(h2))
    h4 = self.tanh(self.fc4(h3))
    h5 = self.relu(self.fc5(h4))
    return self.fc6(h5)

class AffinityMergeLayer_extra6_sincoshad_relu(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*4, dim3*10)
    self.fc2 = torch.nn.Linear(dim3*10, dim3*6)
    self.fc3 = torch.nn.Linear(dim3*6, dim3*4)
    self.fc4= torch.nn.Linear(dim3*4, dim3*2)
    self.fc5 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, torch.sin(x1*x2), torch.cos(x1*x2)], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.tanh(self.fc2(h1))
    h3 = self.tanh(self.fc3(h2))
    h4 = self.tanh(self.fc4(h3))
    return self.fc5(h4)

#############################################
###################  Tanh  ###################
#############################################

class AffinityMergeLayer_extra_tanh(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Tanh()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra_had_tanh(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*3, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Tanh()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, x1*x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra_sincos_tanh(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*6, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Tanh()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, torch.sin(x1), torch.cos(x1), torch.sin(x2), torch.cos(x2)], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra6_tanh(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1+dim2, dim3*2)
    self.fc2 = torch.nn.Linear(dim3*2, dim3*4)
    self.fc3 = torch.nn.Linear(dim3*4, dim3*8)
    self.fc4= torch.nn.Linear(dim3*8, dim3*4)
    self.fc5 = torch.nn.Linear(dim3*4, dim3*2)
    self.fc6 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Tanh()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.tanh(self.fc2(h1))
    h3 = self.tanh(self.fc3(h2))
    h4 = self.tanh(self.fc4(h3))
    h5 = self.relu(self.fc5(h4))
    return self.fc6(h5)

class AffinityMergeLayer_extra6_sincoshad_tanh(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*4, dim3*10)
    self.fc2 = torch.nn.Linear(dim3*10, dim3*6)
    self.fc3 = torch.nn.Linear(dim3*6, dim3*4)
    self.fc4= torch.nn.Linear(dim3*4, dim3*2)
    self.fc5 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Tanh()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, torch.sin(x1*x2), torch.cos(x1*x2)], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.tanh(self.fc2(h1))
    h3 = self.tanh(self.fc3(h2))
    h4 = self.tanh(self.fc4(h3))
    return self.fc5(h4)

#############################################
#################  Sigmoid  #################
#############################################

class AffinityMergeLayer_extra_sigmoid(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Sigmoid()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra_had_sigmoid(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*3, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Sigmoid()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, x1*x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra_sincos_sigmoid(torch.nn.Module):
  #
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*6, dim3*8)
    self.fc2 = torch.nn.Linear(dim3*8, dim3*2)
    self.fc3 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Sigmoid()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, torch.sin(x1), torch.cos(x1), torch.sin(x2), torch.cos(x2)], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    return self.fc3(h2)

class AffinityMergeLayer_extra6_sigmoid(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1+dim2, dim3*2)
    self.fc2 = torch.nn.Linear(dim3*2, dim3*4)
    self.fc3 = torch.nn.Linear(dim3*4, dim3*8)
    self.fc4= torch.nn.Linear(dim3*8, dim3*4)
    self.fc5 = torch.nn.Linear(dim3*4, dim3*2)
    self.fc6 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Sigmoid()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.tanh(self.fc2(h1))
    h3 = self.tanh(self.fc3(h2))
    h4 = self.tanh(self.fc4(h3))
    h5 = self.relu(self.fc5(h4))
    return self.fc6(h5)

class AffinityMergeLayer_extra6_sincoshad_sigmoid(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4): # 2,2,2,1
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*4, dim3*10)
    self.fc2 = torch.nn.Linear(dim3*10, dim3*6)
    self.fc3 = torch.nn.Linear(dim3*6, dim3*4)
    self.fc4= torch.nn.Linear(dim3*4, dim3*2)
    self.fc5 = torch.nn.Linear(dim3*2, dim4)
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Sigmoid()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2, torch.sin(x1*x2), torch.cos(x1*x2)], dim=1)
    h1 = self.tanh(self.fc1(x))
    h2 = self.tanh(self.fc2(h1))
    h3 = self.tanh(self.fc3(h2))
    h4 = self.tanh(self.fc4(h3))
    return self.fc5(h4)

##########################################################
##########################################################
##########################################################

class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    # print(src_idx)
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times