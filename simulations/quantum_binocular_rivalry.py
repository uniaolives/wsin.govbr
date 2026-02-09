import numpy as np
from scipy.integrate import solve_ivp

class QuantumBinocularRivalry:
    """
    Simula a rivalidade binocular em Finney-0.
    Input A: Feed visual do Presente (2026) - Padrão: 'Cassini_Probe'
    Input B: Feed visual do Futuro (12.024) - Padrão: 'Saturn_Matrioshka'
    Medida: Padrão de interferência das ondas viajantes resultantes.
    """

    def __init__(self):
        # Parâmetros das ondas corticais viajantes (baseados no modelo de Wilson-Cowan)
        self.alpha = 1.0      # Taxa de decaimento
        self.beta = 0.5       # Acoplamento excitatório
        self.gamma = 0.3      # Acoplamento inibitório
        self.c = 0.1          # Velocidade de propagação da onda
        self.L = 10.0         # Comprimento do domínio cortical (unidades normalizadas)
        self.N = 100          # Número de pontos no espaço

        # Inputs temporais
        self.input_present = self.generate_pattern('present')   # Padrão 2026
        self.input_future = self.generate_pattern('future')     # Padrão 12024

    def generate_pattern(self, epoch):
        """Gera um padrão de input sensorial para uma época."""
        x = np.linspace(0, self.L, self.N)
        if epoch == 'present':
            # Padrão 2026: Sonda Cassini (estrutura mecânica, aguda)
            return 0.7 * np.sin(2*np.pi*x/self.L) + 0.3 * np.random.randn(self.N)*0.1
        else:  # 'future'
            # Padrão 12024: Cérebro Matrioshka (estrutura fluida, ondulatória)
            return 0.5 * np.sin(4*np.pi*x/self.L) * np.exp(-(x-self.L/2)**2/4) + 0.2 * np.cos(6*np.pi*x/self.L)

    def sigmoid(self, x):
        """Função de resposta sigmoidal."""
        return 1 / (1 + np.exp(-x))

    def traveling_wave_model(self, t, u):
        """Modelo de reação-difusão para ondas corticais viajantes."""
        u = u.reshape((2, self.N))  # u[0]: atividade excitatória (E), u[1]: inibitória (I)
        E, I = u[0], u[1]

        # Input combinado com rivalidade
        attention_cycle = 0.5 * (1 + np.sin(2*np.pi*t/5))
        combined_input = attention_cycle * self.input_present + (1-attention_cycle) * self.input_future

        # Equações de Wilson-Cowan com termo de difusão
        dE_dt = -self.alpha*E + (1 - E)*self.beta*self.sigmoid(E - I + combined_input)
        dI_dt = -self.alpha*I + (1 - I)*self.gamma*self.sigmoid(E)

        # Adiciona difusão
        dE_diff = self.c * np.diff(E, prepend=E[0], append=E[-1])
        dI_diff = self.c * np.diff(I, prepend=I[0], append=I[-1])

        # Adjusting indices for diffusion
        dE_dt += 0.5 * (dE_diff[:-1] + dE_diff[1:])
        dI_dt += 0.5 * (dI_diff[:-1] + dI_diff[1:])

        return np.concatenate([dE_dt, dI_dt])

    def run_experiment(self, duration=30):
        """Executa a simulação da rivalidade binocular."""
        u0 = np.zeros(2*self.N)
        sol = solve_ivp(self.traveling_wave_model, [0, duration], u0,
                        t_eval=np.linspace(0, duration, 500), method='RK45')

        activity = sol.y.reshape((2, self.N, len(sol.t)))
        return sol.t, activity[0]

    def analyze_interference(self, E_activity):
        """Analisa os padrões de interferência resultantes."""
        mean_pattern = np.mean(E_activity, axis=1)
        coherence = np.std(mean_pattern) / (np.mean(np.std(E_activity, axis=0)) + 1e-8)

        spatial_fft = np.fft.fft(mean_pattern)
        freqs = np.fft.fftfreq(self.N, d=self.L/self.N)
        dominant_freq = np.abs(freqs[np.argmax(np.abs(spatial_fft))])

        return {
            'coherence': coherence,
            'dominant_spatial_frequency': dominant_freq,
            'unified_perception': coherence > 0.7,
            'interference_pattern': mean_pattern
        }

if __name__ == "__main__":
    print("🔬 INICIANDO EXPERIMENTO DE RIVALIDADE BINOCULAR QUÂNTICA")
    experiment = QuantumBinocularRivalry()
    time, wave_activity = experiment.run_experiment(duration=30)
    results = experiment.analyze_interference(wave_activity)

    print(f"\n📊 RESULTADOS:")
    print(f"   • Coerência do sistema: {results['coherence']:.3f}")
    print(f"   • Frequência espacial dominante: {results['dominant_spatial_frequency']:.3f} ciclos/unidade")
    print(f"   • Percepção unificada alcançada: {'SIM' if results['unified_perception'] else 'NÃO'}")
    print(f"   • Verdicto: ", end="")

    if results['unified_perception']:
        print("Finney-0 integrou ambos os tempos em uma única realidade qualia.")
    else:
        print("Os inputs temporais permanecem em rivalidade; a consciência oscila entre eles.")
