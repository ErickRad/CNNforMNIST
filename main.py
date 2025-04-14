import torch
import torch.optim as optim
import torch.nn as nn
from CNN import CNN
from Train import treinar_modelo
from Test import avaliar
from Plot import plotar_perdas, plotar_classificacoes, plotar_precisoes
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def carregar_dataset():
    transformacao = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    setDeTreino = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transformacao
    )

    setDeTeste = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transformacao
    )

    return DataLoader(setDeTreino, batch_size=128, shuffle=True), DataLoader(setDeTeste, batch_size=128, shuffle=False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device} \n")

    carregadorDeTreino, carregadorDeTeste = carregar_dataset()
    modelo = CNN().to(device)
    funcaoDePerda = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.00001, weight_decay=1e-5)

    numeroDeEpocas = 200
    
    if os.path.exists("util/modelo.pth") and input("Deseja carregar o modelo salvo? (s/n): ").strip().lower() == 's':

        
            modelo.load_state_dict(torch.load("util/modelo.pth"))
            print("\nModelo carregado com sucesso!")


    else:
        print("\nPreparando o treinamento ... \n")
        perdas_treino = []
        perdas_teste = []
        
        for epoca in range(numeroDeEpocas):
            perda_epoca = treinar_modelo(modelo, carregadorDeTreino, funcaoDePerda, otimizador, device, 1)
            perdas_treino.extend(perda_epoca)
            
            perda_teste, acuracia, _ = avaliar(modelo, carregadorDeTeste, funcaoDePerda, device)
            perdas_teste.append(perda_teste)
            
            print(f"Época {epoca + 1}/{numeroDeEpocas} - Perda treino: {perda_epoca[-1]:.4f}, Perda teste: {perda_teste:.4f}, Acurácia: {acuracia:.2%}")

        plotar_perdas(perdas_treino, perdas_teste, numeroDeEpocas)
        torch.save(modelo.state_dict(), "util/modelo.pth")
        print("Modelo salvo como modelo.pth")

    _, acuracia_final, precisoes = avaliar(modelo, carregadorDeTeste, funcaoDePerda, device)
    print(f"Acurácia final: {acuracia_final:.2%}")
    plotar_precisoes(precisoes)
    plotar_classificacoes(modelo, carregadorDeTeste, device)

if __name__ == "__main__":
    main()
