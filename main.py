import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

sig_DIR = "Signals\\"

inputs = {
    "i": "impulso",
    "d": "degrau",
    "f": "forçado"
}

# Carrega os sinais crus num dicionário
def load():
    files = [f"{a1}{a2}{a3}" for a3 in ("g", "") for a2 in ("i","d","f") for a1 in ("x","y","z")]
    files += [f"t{a1}" for a1 in range (1,4)]
    signal_data = {filename:pd.read_csv(f"{sig_DIR}{filename}.csv") for filename in files}
    
    for k in signal_data.keys():
        signal_data[k].columns = ["t", "x", "y", "z", ""]
        if k[0] == "t":
            signal_data[k] = signal_data[k][["t","x","y","z"]]
            continue
        signal_data[k] = signal_data[k][["t",k[0]]]
    
    # Deixa os dados do comprimento e frequência de amostragem pré-calculados
    signal_metadata = {k:{"l":len(v["t"]), "sf":len(v["t"])/(v.iloc[-1]["t"]-v["t"][0])} for k,v in signal_data.items()}

    return signal_metadata, signal_data
    
def plot_sig(signal_data: dict[str,pd.DataFrame], sig_key: str, display: bool = False):
    data = signal_data[sig_key]
    g = sig_key.endswith("g")
    plt.clf()
    if sig_key.startswith("t"):
        plt.plot(data["t"], data["x"], color='red')
        plt.plot(data["t"], data["y"], color='green')
        plt.plot(data["t"], data["z"], color='blue')
        plt.legend(["x","y","z"])
        plt.title(f"Sinal do acelerômetro em todas direções, medição {sig_key[1]}", fontsize=14)
    else:
        plt.plot(data["t"], data[sig_key[0]])
        plt.title(f"Sinal do {'giroscópio' if g else 'acelerômetro'} na direção {sig_key[0]}, {inputs[sig_key[1]]}", fontsize=14)
        
    plt.xlabel("Tempo (s)", fontsize=14)
    plt.ylabel(f"Aceleração {"angular (rad" if g else "(m"}/s²)", fontsize=14)
    plt.savefig(f"Figures/signals/{sig_key}.png")

    if not display: return
    plt.show()

def fft(signal_metadata: dict[str,dict[str,int|np.float64]] , signal_data: dict[str,pd.DataFrame], sig_key: str, display: bool = False):
    data = signal_data[sig_key]
    sf = signal_metadata[sig_key]["sf"]
    hl: int = signal_metadata[sig_key]["l"]//2 # type: ignore
    
    g = sig_key.endswith("g")
    total = sig_key.startswith("t")
    
    if total:   directions = ["x","y","z"]
    else:       directions = [sig_key[0]]
    
    ffts = [abs(np.fft.fft(data[d])[:hl]) for d in directions]
    w = np.linspace(0, sf/2, hl)

    w_n = []
    for d,f in zip(directions,ffts):
        plt.clf()
        plt.loglog(w, f)
        plt.title(f"Espectro de frequências do {'giroscópio' if g else 'acelerômetro'} na direção {d}, {'medição total' if total else inputs[sig_key[1]]}", fontsize=14)
        plt.xlabel("Frequência (Hz)", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
        plt.savefig(f"Figures/ffts/{f'total{d}{sig_key[1]}' if total else sig_key}.png")
        if display: plt.show()
        
        pk, _ = find_peaks(f)
        w_n.append([w[pk[n]] for n in np.argsort(f[pk])[-2:][::-1]])
    
    return w_n

sig_meta, sig = load()

w_n: dict[str,list[list[np.float64]]] = {}
plt.figure(figsize=(14,6), layout="tight")
for k in sig.keys():
    plot_sig(sig, k)
    w_n.update({k:fft(sig_meta, sig, k)})

avg_w = {s:0. for s in ["x","y","z","xg","yg","zg","xt1","xt2","yt1","yt2","zt1","zt2"]}
for k,v in w_n.items():
    if not k.startswith("t"):
        if k.endswith("g"):
            avg_w[f"{k[0]}g"] += v[0][0]
        else:
            avg_w[k[0]] += v[0][0]
    else:
        for i,n in enumerate(["x","y","z"]):
            avg_w[f"{n}t1"] += v[i][0]
            avg_w[f"{n}t2"] += v[i][1]

for k,v in avg_w.items():
    print(f"{k}: {v/3:.2f}Hz")