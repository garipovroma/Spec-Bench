import torch
import time
import os
import numpy as np
import shortuuid

class BatchStats:
    """
    Statistics of one generation function call of speculative decoding method
    """
    def __init__(self, batch_size, accepted_length=None):
        if accepted_length is None:
            self.accepted_length = torch.empty((batch_size, 0))
        self.start_time = None
        self.end_time = None
        self.wall_time = None
        self.finished = None
    
    def set_timer(self, start_time=None):
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

    def stop_timer(self):
        if self.start_time is None:
            assert "Timer is not started"
        self.end_time = time.time()
        self.wall_time =  self.end_time - self.start_time
    
    def add_accept(self, accepts: torch.tensor):
        self.accepted_length = torch.cat([self.accepted_length, accepts.cpu()], dim=-1)
    
    def calculate_stats(self):
        # self.steps = self.accepted_length.shape[0]
        self.steps = (self.accepted_length > 0).sum(dim=1)
        self.mean_accept = self.accepted_length.sum(dim=1) / self.steps
    
    def calculate_finished(self, ids, eos_id):
        self.finished = torch.any(ids == eos_id, dim=-1)

    def get_new_tokens(self):
        return self.accepted_length.sum(-1)

class Choice:
    """
    Batched version of original choices
    """
    def __init__(self):
        self.index = []
        self.batch_id = []
        self.turns = []
        self.steps = []
        self.new_tokens = []
        self.wall_time = []
        self.cur_accept_lengths_tree = []
    
    def append(self, output, index, new_tokens, steps, wall_time, cur_accept_lengths_tree, batch_id):
        self.index.append(index)
        self.batch_id.append(batch_id)
        self.turns.append(output)
        self.new_tokens.append(new_tokens)
        self.steps.append(steps)
        self.wall_time.append(wall_time)
        self.cur_accept_lengths_tree.append(cur_accept_lengths_tree)
    
    def __dict__(self):
        return {
                "index": self.index,
                "turns": self.turns,
                "decoding_steps": self.steps,
                "new_tokens": self.new_tokens,
                "wall_time": self.wall_time,
                "accept_lengths": self.cur_accept_lengths_tree,
                "batch_index": self.batch_id,
                }

class ExperimentStats:
    def __init__(self, batch_size):
        self.accept_lengths_tree = torch.empty(batch_size, 0 )
        self.batch_size = batch_size
    
    def new_batch(self):
        self.choices = [Choice() for _ in range(self.batch_size)]
        
    def update_exp_stats(self, stats, output, index=1):
        if not stats is None:
            stats.calculate_stats()
            accept_length_tree = stats.accepted_length
            for b, ch in enumerate(self.choices):
                ch.append(  output[b],
                            index,
                            stats.mean_accept[b].item(),
                            stats.steps[b].item(), 
                            stats.wall_time, 
                            accept_length_tree[b].tolist(),
                            shortuuid.uuid(),) 
            self.accept_lengths_tree = torch.cat([self.accept_lengths_tree, accept_length_tree], dim=-1)
        return [ch.__dict__() for ch in self.choices]
    
    def get_total_accept_mean(self):
        return self.accept_lengths_tree.mean().item()
