import numpy as np
import matplotlib.pyplot as plt

A = 1
f_0 = 1

def square_n(N_max):
    # --- Vetor de Tempo ---
    # Gera 1000 pontos de tempo para plotar 2 ciclos completos da onda
    t = np.linspace(0, 1, 1000)

    # --- Construção da Série de Fourier ---
    # Inicializa o sinal de saída com zeros
    x_t = np.zeros_like(t)

    # Loop para somar os harmônicos de n=1 até N_max
    for n in range(1, N_max + 1):
        # A série de Fourier para uma onda quadrada só tem componentes ímpares
        if n % 2 != 0:
        # Calcula o coeficiente b_n para o harmônico n atual
            b_n = (4 * A) / (n * np.pi)

            # Calcula o termo da série (harmônico) e soma ao sinal total
            termo = b_n * np.sin(2 * np.pi * n * f_0 * t)
            x_t += termo
    
    return max(x_t)-1

error = []
terms = []
for n in range(1,10000):
    e = square_n(n)
    if not n%100 or e < 0.05:
        print(f"{n} Termos, com erro de {e:.2%}")
        error.append(100*e)
        terms.append(n)
        if e < 0.05:
            break

plt.figure(figsize=(14,6), layout="tight")
plt.stem(terms,error)
plt.xlabel("Número de termos (-)", fontsize=14)
plt.ylabel("Erro máximo (%)", fontsize=14)
plt.grid(True)
plt.savefig("Figures/Erro vs n.png")