import sys
from tqdm import tqdm
import torch

__all__ = ['progressMeter']

'''class progressMeter:
    def __init__(self, desc, writer, logger, total_data_len, epoch, use_pbar):
        self.states = torch.zeros(1, device='cuda', dtype=torch.float32)
        self.states_name_idx = {'batch_size':0}
        self.num_states = 0
        
        self.writer = writer
        self.logger = logger
        self.use_pbar = use_pbar
        self.epoch = epoch
        self.desc = f'Epoch {epoch} | {desc}: '
        self.total_data_len = total_data_len
        if use_pbar:
            self.pbar = tqdm(total=total_data_len, ncols=70, file=sys.stdout, desc=self.desc)
        
    def update(self, state_dict):
        assert 'batch_size' in state_dict.keys(), "Batch size should be provided in states! (as 'batch_size')"
        assert 'loss' in state_dict.keys(), "Loss should be provided in states! (as 'loss')"
        with torch.no_grad():
            self.state_size = state_dict.pop('batch_size')
            self.states[0] += self.state_size
            for name, val in state_dict.items():
                if name not in self.states_name_idx.keys():
                    self.num_states += 1
                    self.states_name_idx[name] = self.num_states
                    self.states = torch.concat((self.states, torch.zeros((1), device='cuda')))
                idx = self.states_name_idx[name]
                if torch.is_tensor(val):
                    val = val.data
                self.states[idx] += val * self.state_size
    
    def verbose_states(self):
        avg_states = self.states / self.states[0]
        if self.use_pbar:
            print('\r', end='')
            self.pbar.update(int(self.state_size))
        else:
            print(self.desc, end='')
        out = ''
        for name, idx in self.states_name_idx.items():
            if name != 'batch_size':
                out += f" | {name}: {avg_states[idx]:.5f}"
        print(out, end='')
        epoch_prog = f"({self.states[0]:4.0f}/{self.total_data_len:4.0f})"
        self.logger.debug(self.desc+epoch_prog+out)
        if not self.use_pbar:
            print("")
    
    def write_summary(self):
        avg_states = self.states / self.states[0]
        for name, idx in self.states_name_idx.items():
            if name != 'batch_size':
                self.writer.add_scalar(name, avg_states[idx], self.epoch)
        if self.use_pbar:
            self.pbar.close()
        return avg_states[self.states_name_idx['loss']] # Average Loss
    
'''
class progressMeter:
    def __init__(self, desc, writer, logger, total_data_len, epoch, use_pbar):
        self.states = torch.zeros(1, device='cuda', dtype=torch.float32)
        self.states_name_idx = {'batch_size':0}
        self.num_states = 0
        
        self.writer = writer
        self.logger = logger
        self.use_pbar = use_pbar
        self.epoch = epoch
        self.desc = f'Epoch {epoch} | {desc}: '
        self.total_data_len = total_data_len
        if use_pbar:
            self.pbar = tqdm(total=total_data_len, ncols=70, file=sys.stdout, desc=self.desc)
        
    def update(self, state_dict):
        assert 'batch_size' in state_dict.keys(), "Batch size should be provided in states! (as 'batch_size')"
        assert 'loss' in state_dict.keys(), "Loss should be provided in states! (as 'loss')"
        with torch.no_grad():
            self.state_size = state_dict.pop('batch_size')
            self.states[0] += self.state_size
            for name, val in state_dict.items():
                if name not in self.states_name_idx.keys():
                    self.num_states += 1
                    self.states_name_idx[name] = self.num_states
                    self.states = torch.concat((self.states, torch.zeros((1), device='cuda')))
                idx = self.states_name_idx[name]
                if torch.is_tensor(val):
                    val = val.data
                self.states[idx] += val * self.state_size
    
    def verbose_states(self):
        # 1. 로그 파일 업데이트 로직 제거
        avg_states = self.states / self.states[0]
        if self.use_pbar:
            print('\r', end='')
            self.pbar.update(int(self.state_size))
        else:
            print(self.desc, end='')
        out = ''
        for name, idx in self.states_name_idx.items():
            if name != 'batch_size':
                out += f" | {name}: {avg_states[idx]:.5f}"
        print(out, end='')
        # self.logger.debug(self.desc+epoch_prog+out) # <--- 이 부분이 제거되었습니다.
        if not self.use_pbar:
            print("")
    
    def write_summary(self):
        avg_states = self.states / self.states[0]
        
        # 2. 로깅 로직을 write_summary로 이동 (에포크 요약 로깅)
        out = ''
        for name, idx in self.states_name_idx.items():
            if name != 'batch_size':
                self.writer.add_scalar(name, avg_states[idx], self.epoch)
                out += f" | {name}: {avg_states[idx]:.5f}" # 로그 메시지 생성을 위해 추가
                
        epoch_prog = f"({self.states[0]:4.0f}/{self.total_data_len:4.0f})"
        self.logger.debug(f'Epoch Summary | {self.desc}{epoch_prog}{out}') # <--- 이 부분이 추가되었습니다.
        
        if self.use_pbar:
            self.pbar.close()
        return avg_states[self.states_name_idx['loss']] # Average Loss