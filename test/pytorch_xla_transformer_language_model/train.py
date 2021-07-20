# Copyright (c) 2019, Bryan McCann
# All rights reserved.

import os
import time
import math

import numpy
import torch
import torch.utils.data

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# from test.pytorch-xla-transformer-language-model.transformer import Transformer
def gelu(x):
  return x * torch.sigmoid(1.702 * x)


class Feedforward(torch.nn.Module):

  def __init__(self, outer_dimension, inner_dimension):
    super().__init__()
    self.linear_in = torch.nn.Linear(outer_dimension, inner_dimension)
    self.linear_out = torch.nn.Linear(inner_dimension, outer_dimension)

  def forward(self, input_sequence):
    return self.linear_out(gelu(self.linear_in(input_sequence)))


class Attention(torch.nn.Module):

  def __init__(self, dimension, num_heads):
    super().__init__()
    self.projection = torch.nn.Linear(dimension, 3 * dimension, bias=False)
    self.softmax = torch.nn.Softmax(dim=-1)
    self.scale = dimension**-0.5
    self.num_heads = num_heads
    self.batch_mask = None
    self.sequence_masks = None
    self.out = torch.nn.Linear(dimension, dimension, bias=False)

  def apply_masks(self, x, bs, nh, sl, hd):
    neginf = (-torch.from_numpy(numpy.array(numpy.inf)))
    if x.dtype == torch.float16:
      neginf = neginf.half()
    if self.batch_mask is not None:
      return x.masked_fill(self.batch_mask.bool(), neginf)
    else:
      return x

  def batch_heads(self, x, bs, sl, nh, hd):
    return x.view(bs, sl, nh,
                  hd).transpose(1, 2).contiguous().view(bs * nh, sl, hd)

  def unbatch_heads(self, x, bs, sl, od):
    return x.transpose(0, 1).contiguous().view(sl, bs, od).transpose(0, 1)

  def forward(self, input_sequence):
    bs, sl, od = input_sequence.size()
    nh = self.num_heads
    hd = od // nh
    projection_output = self.projection(input_sequence)
    q = projection_output[:,:,:512].clone()
    k = projection_output[:,:,512:1024].clone()
    v = projection_output[:,:,1024:].clone()
    # q, k, v = self.projection(input_sequence).clone().chunk(3, dim=-1)
    # import pdb;pdb.set_trace()
    q *= self.scale
    q, k, v = [self.batch_heads(x, bs, sl, nh, hd) for x in [q, k, v]]
    attention_weights = self.softmax(
        self.apply_masks(torch.bmm(q, k.transpose(1, 2)), bs, nh, sl, hd))
    return self.out(
        self.unbatch_heads(torch.bmm(attention_weights, v), bs, sl, od))


class Residual(torch.nn.Module):

  def __init__(self, function, dimension, dropout=0.2):
    super().__init__()
    self.operations = torch.nn.Sequential(
        torch.nn.LayerNorm(dimension), function, torch.nn.Dropout(dropout))

  def forward(self, input_sequence):
    return self.operations(input_sequence) + input_sequence


class Layer(torch.nn.Module):

  def __init__(self, outer_dimension, inner_dimension, num_heads=8,
               dropout=0.2):
    super().__init__()
    self.operations = torch.nn.Sequential(
        Residual(
            Attention(outer_dimension, num_heads),
            outer_dimension,
            dropout=dropout),
        Residual(
            Feedforward(outer_dimension, inner_dimension),
            outer_dimension,
            dropout=dropout))

  def forward(self, input_sequence):
    return self.operations(input_sequence)


class Transformer(torch.nn.Module):

  def __init__(self, max_sequence_length, num_layers, outer_dimension,
               inner_dimension, num_heads, dropout):
    super().__init__()
    self.embed = torch.nn.Embedding(256, outer_dimension)
    self.position = torch.nn.Embedding(max_sequence_length, outer_dimension)
    layers = [
        Layer(
            outer_dimension,
            inner_dimension,
            num_heads=num_heads,
            dropout=dropout) for i in range(num_layers)
    ]
    self.layers = torch.nn.Sequential(*layers)
    self.norm = torch.nn.LayerNorm(outer_dimension)
    self.out = torch.nn.Linear(outer_dimension, 256)
    self.xent = torch.nn.CrossEntropyLoss()

  def set_batch_mask(self, mask):
    for layer in self.layers:
      layer.operations[0].operations[1].batch_mask = mask

  def forward(self, input, positions, target=None, batch_mask=None):
    if batch_mask is not None:
      self.set_batch_mask(batch_mask)
    embedding = self.embed(input) + self.position(positions)
    scores = self.out(self.norm(self.layers(embedding)))
    import pdb;pdb.set_trace()
    loss = self.xent(scores.view(-1, 256), target.view(-1))
    return loss


class LazyDataset:

  def __init__(self, path, sequence_length):
    self.path = path
    self.size = os.stat(path).st_size - sequence_length
    self.sequence_length = sequence_length

  def __getitem__(self, index):
    with open(self.path, 'rb') as f:
      f.seek(index)
      chunk = f.read(self.sequence_length)
    return torch.ByteTensor(numpy.frombuffer(chunk, dtype=numpy.uint8))
    # return torch.BoolTensor(numpy.frombuffer(chunk, dtype=numpy.uint8))

  def __len__(self):
    return self.size


def get_dataloader(path, batch_size, sequence_length, num_workers):
  dataset = LazyDataset(path, sequence_length + 1)
  if xm.xrt_world_size() > 1:
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
  else:
    sampler = torch.utils.data.RandomSampler(dataset)
  return torch.utils.data.DataLoader(
      dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)

def loop_with_amp(model, input, positions, target, causal_mask, optimizer, xla_enabled, autocast, scaler):
  with autocast():
    loss = model(input, positions, target, batch_mask=causal_mask)
  if xla_enabled:
    scaler.scale(loss).backward()
    gradients = xm._fetch_gradients(optimizer)
    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
    scaler.step(optimizer)
    scaler.update()
    xm.mark_step()
  else:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

  return loss

def get_autocast_and_scaler(xla_enabled): 
    if xla_enabled: 
        from torch_xla.amp import autocast, GradScaler
        return autocast, GradScaler()
    
    from torch.cuda.amp import autocast, GradScaler
    return autocast, GradScaler()

def main(index):
  BATCH_SIZE = 64
  LOG_STEPS = 10
  METRICS_STEP = 50
  NUM_EPOCHS = 8
  SEQUENCE_LENGTH = 256
  if amp_enabled:
    autocast, scaler = get_autocast_and_scaler(xla_enabled)
  if xla_enabled:
    device = xm.xla_device()
  else:
    device = "cuda:0"
  model = Transformer(256, 12, 512, 2048, 8, 0.2).to(device)
  # import pdb;pdb.set_trace()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

  def train_loop_fn(loader):
    tracker = xm.RateTracker()

    positions = torch.arange(SEQUENCE_LENGTH).long().view(
        1, SEQUENCE_LENGTH).to(device)
    causal_mask = torch.triu(
        torch.ones(
            SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=torch.uint8, device=device),
        diagonal=1).unsqueeze(0)

    model.train()
    for iteration, batch in enumerate(loader):
      optimizer.zero_grad()
      input = batch[:, :-1].long()
      target = batch[:, 1:].long()
      if not xla_enabled:
        input = input.to(device)
        target = target.to(device)

      if amp_enabled:
        loss = loop_with_amp(model, input, positions, target, causal_mask, optimizer, xla_enabled, autocast, scaler)
      else:
        loss = model(input, positions, target, batch_mask=causal_mask)
        loss.backward()
        if xla_enabled:
          xm.optimizer_step(optimizer)
        else:
          optimizer.step()
      tracker.add(BATCH_SIZE)
      if iteration % LOG_STEPS == 0:
        print('[{}]({}) Loss={:.5f} Rate={:.2f}'.format(
            device, iteration,
            loss.item() / math.log(2), tracker.rate()))
      # if iteration % METRICS_STEP == 0:
      #   xm.master_print(met.metrics_report())

  if xla_enabled:
    train_loader = get_dataloader('/pytorch/xla/test/pytorch_xla_transformer_language_model/datasets/enwik8/train/train.txt.raw',
                                  BATCH_SIZE, SEQUENCE_LENGTH, 0)    
  else:
    train_loader = get_dataloader('test/pytorch_xla_transformer_language_model/datasets/enwik8/train/train.txt.raw',
                                BATCH_SIZE, SEQUENCE_LENGTH, 0)

  for epoch in range(0, NUM_EPOCHS):
    if xla_enabled:
      import torch_xla.distributed.parallel_loader as pl
      para_loader = pl.ParallelLoader(train_loader, [device])
      train_loop_fn(para_loader.per_device_loader(device))
    else:
      train_loop_fn(train_loader)


if __name__ == '__main__':
  xla_enabled= False
  amp_enabled= True
  xmp.spawn(main, args=())
