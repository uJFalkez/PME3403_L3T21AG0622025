import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sig_DIR = "Signals\\"

def load(filename: str, coord: str) -> tuple[tuple[int, np.float64], pd.DataFrame]:
    df = pd.read_csv(f"{sig_DIR}{filename}.csv")
    
    df.columns = ["t", "x", "y", "z", ""]
    df = df[["t",coord]]
    
    # Deixa os dados do comprimento e frequência de amostragem pré-calculados
    metadata = len(df["t"]), len(df["t"])/(df.iloc[-1]["t"]-df["t"][0])

    return metadata, df

def ham_filter(s: pd.Series, M: int, wc: float) -> np.ndarray:
    n = np.arange(M)
    hd = np.sinc((wc/np.pi)*(n - (M-1)/2))  # resposta ideal
    w = np.hamming(M)                       # janela
    h = hd * w                              # filtro final
    h /= np.sum(h)
    
    y = np.convolve(s, h, mode='same')
    
    return y

def fft(md: tuple[int, np.float64], s: pd.Series|np.ndarray):
    hl = md[0]//2
    return np.linspace(0, md[1]/2, hl), abs(np.fft.fft(s)[:hl])

file = "yf"
coord = "y"

md, sig = load(file, coord)

filtered = ham_filter(sig[coord], 51, 0.1*np.pi)

plt.figure(figsize=(14,6), layout="tight")
plt.plot(sig["t"], sig[coord], linewidth=3, linestyle="--")
plt.plot(sig["t"], filtered)
plt.title(f"Sinal: original vs filtrado (forçado em {coord})", fontsize=14)
plt.xlabel("Tempo (s)", fontsize=14)
plt.ylabel("Aceleração (m/s²)", fontsize=14)
plt.legend(["Original", "Filtrado"])
plt.savefig(f"Figures/filtered_signal/{file}.png")
plt.show()

w, f1 = fft(md, sig[coord])
_, f2 = fft(md, filtered)

plt.figure(figsize=(14,6), layout="tight")
plt.loglog(w, f1, linestyle="--")
plt.loglog(w, f2)
plt.title(f"Espectros de frequência: original vs filtrado (forçado em {coord})", fontsize=14)
plt.xlabel("Frequência (Hz)", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.legend(["Original", "Filtrado"])
plt.savefig(f"Figures/filtered_fft/{file}.png")