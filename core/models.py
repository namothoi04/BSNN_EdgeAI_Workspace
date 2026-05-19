# File: core/models.py
import torch
import torch.nn as nn
from core.layers import BinaryWeightConv2d, BinaryActivation
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*13*13, 10)
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class BNN(nn.Module):
    def __init__(self, activation_type="binary"):
        super(BNN, self).__init__()
        
        self.conv1 = BinaryWeightConv2d(1, 32, 3)
        self.bn =  nn.BatchNorm2d(32)

        if activation_type == "binary":
            self.act = BinaryActivation()
        elif activation_type == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("Chỉ hỗ trợ 'binary' hoặc 'relu'")
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*13*13, 10)
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class BaseSNN(nn.Module):
    def __init__(self, beta: float = 0.9) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Khối trích xuất đặc trưng
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Khối phân loại
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 13 * 13, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x_steps: torch.Tensor, return_mem: bool = False):
        # x_steps: [num_steps, Batch, 1, 28, 28]
        num_steps = x_steps.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec, mem2_rec = [], []

        for step in range(num_steps):
            x_t = x_steps[step]

            # Layer 1
            cur1 = self.pool(self.conv1(x_t))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc(self.flatten(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            if return_mem:
                mem2_rec.append(mem2)

        if return_mem:
            return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
        return torch.stack(spk2_rec, dim=0)
class SNNGenericWrapper(nn.Module):
    def __init__(self, snn_model, coding_type="rate", num_steps=25, tau_latency=5.0):
        super().__init__()
        self.model = snn_model
        self.coding_type = coding_type.lower()
        self.num_steps = num_steps
        self.tau_latency = tau_latency
        
        assert self.coding_type in ["rate", "temporal"], "Chỉ hỗ trợ dạng coding 'rate' hoặc 'temporal'"

    def forward(self, x):
        # --- BƯỚC 1: MÃ HÓA (ENCODING) ---
        # Chuyển ảnh tĩnh (Batch, C, H, W) -> Chuỗi xung (num_steps, Batch, C, H, W)
        if self.coding_type == "rate":
            spike_data = spikegen.rate(x, num_steps=self.num_steps)
            
            # --- BƯỚC 2 & 3: CHẠY LÕI & GIẢI MÃ (DECODING) ---
            spk_out = self.model(spike_data, return_mem=False)
            # Rate decoding: Tổng hợp số lượng xung
            logits = spk_out.sum(dim=0)
            
        elif self.coding_type == "temporal":
            spike_data = spikegen.latency(
                x, num_steps=self.num_steps, tau=self.tau_latency, threshold=0.01, bypass=True
            )
            
            # --- BƯỚC 2 & 3: CHẠY LÕI & GIẢI MÃ (DECODING) ---
            _, mem_out = self.model(spike_data, return_mem=True)
            # Temporal decoding: Tổng hợp tích lũy điện thế màng
            logits = mem_out.sum(dim=0)
            
        return logits