import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# --- símbolos gerais ---
t = sp.symbols('t', real=True)
T0 = sp.symbols('T0', positive=True)
k = sp.symbols('k', integer=True)
omega0 = 2*sp.pi/T0
j = sp.I

# --- função para calcular X_k simbolicamente ---
def fourier_coeff_complex_symbolic(x_t, T0_sym=T0, t_sym=t, k_sym=k):
    """
    Retorna X_k (expressão simbólica) = (1/T0) * integral over one period of x(t) * exp(-j k omega0 t) dt
    Assumimos integração de t0 a t0+T0 — o usuário pode escolher t0 passando uma função lambda externa se quiser.
    Aqui integramos de -T0/2 a T0/2 (centro).
    """
    omega0_sym = 2*sp.pi/T0_sym
    # integramos sobre um período centrado em zero
    t1 = -T0_sym/2
    t2 =  T0_sym/2
    integrand = x_t * sp.exp(-j * k_sym * omega0_sym * t_sym)
    Xk = (1/T0_sym) * sp.simplify(sp.integrate(integrand, (t_sym, t1, t2)))
    # simplifica trigonométricas/exponenciais
    Xk = sp.simplify(sp.trigsimp(sp.simplify(sp.factor(Xk))))
    return sp.simplify(Xk)

# --- geração da série parcial (reconstrução simbólica com N harmônicos) ---
def fourier_series_partial(Xk_expr, N, T0_sym=T0, t_sym=t):
    """
    Recebe Xk_expr(k) (funcão simbólica de k) e retorna x_N(t) = sum_{k=-N}^{N} X_k e^{j k omega0 t}
    """
    omega0_sym = 2*sp.pi/T0_sym
    k_var = sp.symbols('kk', integer=True)
    # construir soma
    expr_sum = 0
    for kk in range(-N, N+1):
        Xk_sub = sp.simplify(Xk_expr.subs({k: kk}))
        expr_sum += Xk_sub * sp.exp(j * kk * omega0_sym * t_sym)
    expr_sum = sp.simplify(sp.expand(expr_sum))
    return sp.simplify(sp.re(expr_sum))  # se x(t) real, tomar parte real

# --- helpers para as perguntas c,d,e simbolicamente ---
def add_constant_to_spectrum(Xk_expr, a_sym):
    # adicionar constante a => só altera DC (k=0)
    Xk_new = sp.simplify(sp.Piecewise((Xk_expr + a_sym, sp.Eq(k,0)), (Xk_expr, True)))
    # retorne expressão condicional; para manipular simbolicamente normalmente substituímos k
    return Xk_new

def shift_in_time_spectrum(Xk_expr, shift_time):
    # deslocamento x(t - t0) => X_k * exp(-j k omega0 t0)
    omega0_sym = 2*sp.pi/T0
    return sp.simplify(Xk_expr * sp.exp(-j * k * omega0_sym * shift_time))

def scale_spectrum(Xk_expr, b_sym):
    # multiplicação por b escala todos os coeficientes
    return sp.simplify(b_sym * Xk_expr)

# --- função para plotar amplitude e fase (numérico) ---
def plot_spectrum_numeric(Xk_expr, kmin=1, kmax=10, subs_dict=None, title_prefix=''):
    """
    Xk_expr: expressão simbólica em k
    subs_dict: dicionário para substituir símbolos (ex: {T0:1.0, a:1.0})
    plota |X_k| e arg(X_k) para k in [kmin,kmax]
    """
    if subs_dict is None:
        subs_dict = {}
    ks = np.arange(kmin, kmax+1)
    Xk_vals = []
    for kk in ks:
        val = complex(sp.N(Xk_expr.subs({k: kk}).subs(subs_dict)))
        Xk_vals.append(val)
    Xk_vals = np.array(Xk_vals)
    amps = np.abs(Xk_vals)
    phs  = np.angle(Xk_vals)

    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].stem(ks, amps, basefmt=" ")
    axs[0].set_title(title_prefix + "Espectro de Amplitude |X_k|")
    axs[0].set_xlabel("Harmônico (k)")
    axs[0].set_xticks(ks)
    axs[0].grid(True)

    axs[1].stem(ks, phs, basefmt=" ")
    axs[1].set_title(title_prefix + "Espectro de Fase ∠X_k (rad)")
    axs[1].set_xlabel("Harmônico (k)")
    axs[1].set_xticks(ks)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"Figures/{title_prefix.split()[0]}.png")

if __name__ == "__main__":
    # --- exemplo: triângulo central em um período (como discutimos)
    a = sp.symbols('a', real=True)   # amplitude do pico
    # definimos tau (largura do triângulo) como T0/2 (ajuste se quiser)
    tau = T0/2
    # triângulo simétrico centrado em zero: f(t) = a*(1 - 2|t|/tau) para |t|<=tau/2
    x_piece = sp.Piecewise(
        (a*(1 - 2*sp.Abs(t)/tau), sp.Abs(t) <= tau/2),
        (0, True)
    )

    # calcula X_k simbolicamente
    Xk_sym = fourier_coeff_complex_symbolic(x_piece, T0_sym=T0, t_sym=t, k_sym=k)
    sp.pprint(sp.simplify(Xk_sym))
    # Espere ver uma expressão que simplifique para algo ~ (4a/(pi^2 k^2)) sin^2(pi k /4)

    # Plot numérico (substitui T0 e a por valores numéricos)
    subs = {T0: 1.0, a: 1.0}
    plot_spectrum_numeric(Xk_sym, subs_dict=subs, title_prefix='Onda Quadrada: ')

    # -- (c) adicionar constante a_const:
    a_const = sp.symbols('a_const', real=True)
    Xk_added_const = add_constant_to_spectrum(Xk_sym, a_const)
    # Para plotar, substitua a_const e outros símbolos:
    # Observação: add_constant_to_spectrum retornou Piecewise. Para avaliar numericamente:
    Xk_added_callable = lambda kk: complex(sp.N(Xk_sym.subs({k:kk}).subs(subs)) + (a_const.subs({}) if kk==0 else 0))
    # Mais simples: re-calcule X0+ a_const e mantenha outros coeficientes.
    plot_spectrum_numeric(Xk_added_const, subs_dict=subs, title_prefix='+ Constante a=1: ')

    # -- (d) deslocamento em T0/4:
    shift_time = T0/4
    Xk_shifted = shift_in_time_spectrum(Xk_sym, shift_time)
    # amplitude igual, fase aumenta por -k*pi/2 (como esperado)
    plot_spectrum_numeric(Xk_shifted, subs_dict=subs, title_prefix='Deslocado T0/4: ')

    # -- (e) escala b
    b = sp.symbols('b', real=True)
    Xk_scaled = scale_spectrum(Xk_sym, b)