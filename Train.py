import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from CNN import CNN

def treinar_modelo(modelo, dataloader, funcaoDePerda, otimizador, device, epocas=1):
    modelo.train()
    perdas_epoca = []
    
    for _ in range(epocas):
        perda_total = 0.0
        for entradas, rotulos in dataloader:
            entradas, rotulos = entradas.to(device), rotulos.to(device)
            
            otimizador.zero_grad()
            saidas = modelo(entradas)
            perda = funcaoDePerda(saidas, rotulos)
            perda.backward()
            otimizador.step()
            
            perda_total += perda.item()
        
        perdas_epoca.append(perda_total / len(dataloader))
    
    return perdas_epoca