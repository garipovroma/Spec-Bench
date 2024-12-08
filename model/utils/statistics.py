import torch
import time

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
        self.wall_time = self.start_time - self.end_time
    
    def add_accept(self, accepts: torch.tensor):
        self.accepted_length = torch.cat([self.accepted_length.to(device=accepts.device), accepts], dim=-1)
    
    def calculate_stats(self):
        self.steps = self.accepted_length.shape[0]
        self.mean_accept = self.accepted_length.mean()
        self.mean_accept_per_sample = self.accepted_length.mean(dim=1)
    
    def calculate_finished(self, ids, eos_id):
        self.finished = torch.any(ids == eos_id, dim=-1)

    def get_new_tokens(self):
        return self.accepted_length.sum(-1)

    def __call__(self, *args, **kwds):
        self.calculate_stats()
        return self
