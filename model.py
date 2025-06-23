import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class TradingSNN(nn.Module):
    def __init__(self, num_inputs=5, num_hidden=100, num_outputs=3, beta=0.95):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, time_steps=20):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_count = torch.zeros((x.size(0), 3), device=x.device)

        for _ in range(time_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_count += spk2

        return spk_count