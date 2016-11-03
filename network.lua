function create_network()

require 'torch'
require 'nn'
require 'image'
require 'cutorch'
require 'cunn'
require "optim"
require "weight-init"


  mlp = nn.Sequential()

  input_patchsz = 16
  output_patchsz = 8
  sx = 1+math.floor((input_patchsz - output_patchsz) / 2)
  sy = 1+math.floor((input_patchsz - output_patchsz) / 2)
  ex = math.floor((input_patchsz + output_patchsz) / 2)
  ey = math.floor((input_patchsz + output_patchsz) / 2)

  input_nodes = input_patchsz*input_patchsz*2
  output_nodes = output_patchsz*output_patchsz

  hidden_layer_1 = 1024
  hidden_layer_2 = 512
  hidden_layer_3 = 256
  hidden_layer_4 = 128

  mlp:add(nn.Linear(input_nodes,hidden_layer_1))
  mlp:add(nn.RReLU())
  mlp:add(nn.Linear(hidden_layer_1,hidden_layer_2))
  mlp:add(nn.RReLU())
  mlp:add(nn.Linear(hidden_layer_2,hidden_layer_3))
  mlp:add(nn.RReLU())
  mlp:add(nn.Linear(hidden_layer_3,hidden_layer_4))
  mlp:add(nn.RReLU())
  mlp:add(nn.Linear(hidden_layer_4,output_nodes))

  mlp = require('weight-init')(mlp,'kaiming')
  mlp:cuda()

  return mlp
end
