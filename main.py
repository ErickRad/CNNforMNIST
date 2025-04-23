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

    treino = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transformacao
    )

    teste = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transformacao
    )

    return DataLoader(treino, batch_size=128, shuffle=True), DataLoader(teste, batch_size=128, shuffle=False)

def quantizar_modelo(modelo, dataloader):
    modelo.eval()
    modelo.cpu()
    modelo.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(modelo, inplace=True)

    with torch.no_grad():
        for imagens, _ in dataloader:
            modelo(imagens)
            break

    torch.quantization.convert(modelo, inplace=True)
    return modelo

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device} \n")

    carregadorDeTreino, carregadorDeTeste = carregar_dataset()
    modelo = CNN()

    funcaoDePerda = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.000001, weight_decay=1e-5)
    numeroDeEpocas = 100

    if input("Deseja carregar algum modelo? (s/n): ").strip().lower() == 's':
        tipo = input("Qual modelo? Normal/Quantizado (n/q): ").strip().lower()

        if tipo == 'n' and os.path.exists("util/modelo.pth"):
            modelo.load_state_dict(torch.load("util/modelo.pth"))
            print("Modelo normal carregado com sucesso!")

        elif tipo == 'q' and os.path.exists("util/quantizado.pth"):
            modelo.load_state_dict(torch.load("util/quantizado.pth"))
            print("Modelo quantizado carregado com sucesso!")

        else:
            print("Modelo não encontrado.")

    else:
        print("\nTreinando modelo ...\n")
        perdas_treino = []
        perdas_teste = []

        modelo.to(device)
        for epoca in range(numeroDeEpocas):
            perda_epoca = treinar_modelo(modelo, carregadorDeTreino, funcaoDePerda, otimizador, device, 1)
            perdas_treino.extend(perda_epoca)

            perda_teste, acuracia, _ = avaliar(modelo, carregadorDeTeste, funcaoDePerda, device)
            perdas_teste.append(perda_teste)

            print(f"Época {epoca + 1}/{numeroDeEpocas} - Perda treino: {perda_epoca[-1]:.4f}, Perda teste: {perda_teste:.4f}, Acurácia: {acuracia:.2%}")

        plotar_perdas(perdas_treino, perdas_teste, numeroDeEpocas)

    if input("Deseja quantizar esse modelo? (s/n): ").strip().lower() == 's':
        modelo = quantizar_modelo(modelo, carregadorDeTreino)
        print("Modelo quantizado com sucesso!")

    if input("Deseja salvar esse modelo? (s/n): ").strip().lower() == 's':
        nome = "quantizado.pth" if isinstance(modelo.quant, torch.quantization.QuantStub) else "modelo.pth"
        torch.save(modelo.state_dict(), f"util/{nome}")
        print(f"Modelo salvo como {nome}")
    

    modelo.eval()
    modelo.to(device)
    _, acuracia_final, precisoes = avaliar(modelo, carregadorDeTeste, funcaoDePerda, device)
    print(f"Acurácia final: {acuracia_final:.2%}")
    plotar_precisoes(precisoes)
    plotar_classificacoes(modelo, carregadorDeTeste, device)

if __name__ == "__main__":
    main()
