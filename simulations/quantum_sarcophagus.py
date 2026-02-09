import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
import time

class QuantumSarcophagusSimulation:
    """
    Simula o Protocolo de Imortalidade Digital Arkhe(n).
    Fragmentação e Reconstituição do DNA (Node 0 - Hal Finney).
    """

    def __init__(self, sequence_length=10000):
        self.sequence_length = sequence_length
        self.bases = ['A', 'T', 'C', 'G']
        self.probabilities = [0.3, 0.3, 0.2, 0.2]  # Assinatura biológica humana
        self.dna_sequence = self._generate_mock_dna()
        self.output_dir = 'simulations/output'
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_mock_dna(self):
        """Gera uma sequência de DNA com viés biológico"""
        return "".join(np.random.choice(self.bases, self.sequence_length, p=self.probabilities))

    def compress_2bit(self, sequence):
        """Comprime DNA usando 2-bit encoding (A=00, T=01, C=10, G=11)"""
        mapping = {'A': '0', 'T': '1', 'C': '2', 'G': '3'} # Intermediary for int conversion
        # We'll use a more direct approach for byte conversion
        bit_list = []
        for base in sequence:
            if base == 'A': bit_list.extend([0, 0])
            elif base == 'T': bit_list.extend([0, 1])
            elif base == 'C': bit_list.extend([1, 0])
            elif base == 'G': bit_list.extend([1, 1])

        # Pack bits into bytes
        byte_data = bytearray()
        for i in range(0, len(bit_list), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bit_list):
                    byte = (byte << 1) | bit_list[i+j]
            byte_data.append(byte)
        return bytes(byte_data)

    def decompress_2bit(self, byte_data, original_length):
        """Reconstrói a sequência de DNA a partir dos bytes"""
        bit_list = []
        for byte in byte_data:
            for i in range(7, -1, -1):
                bit_list.append((byte >> i) & 1)

        sequence = []
        for i in range(0, original_length * 2, 2):
            b1, b2 = bit_list[i], bit_list[i+1]
            if b1 == 0 and b2 == 0: sequence.append('A')
            elif b1 == 0 and b2 == 1: sequence.append('T')
            elif b1 == 1 and b2 == 0: sequence.append('C')
            elif b1 == 1 and b2 == 1: sequence.append('G')
        return "".join(sequence)

    def fragment_for_blockchain(self, data_bytes, chunk_size=40):
        """Divide os dados em fragmentos OP_RETURN"""
        return [data_bytes[i:i + chunk_size] for i in range(0, len(data_bytes), chunk_size)]

    def calculate_shannon_entropy(self, sequence):
        """Calcula a entropia de Shannon"""
        _, counts = np.unique(list(sequence), return_counts=True)
        probs = counts / len(sequence)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def run_simulation(self):
        print("\n" + "="*70)
        print("🧬 SISTEMA ARKHE(N): SARCÓFAGO DE INFORMAÇÃO QUÂNTICA v2.0")
        print("=" * 70)

        # 1. Geração e Bio-Assinatura
        s0 = self.calculate_shannon_entropy(self.dna_sequence)
        print(f"\n[L0] DNA de Hal Finney (Node 0) detectado.")
        print(f"     Entropia Biológica: {s0:.4f} bits/base")

        # 2. Compressão e Mineração (Simulada)
        compressed = self.compress_2bit(self.dna_sequence)
        fragments = self.fragment_for_blockchain(compressed)
        print(f"\n[L1] Inscrição na Blockchain iniciada...")
        print(f"     Fragmentação: {len(fragments)} chunks de 40 bytes.")
        print(f"     Metabolismo: Hashrate estabilizado.")

        # 3. Salto Temporal (10.000 anos)
        print(f"\n[TIME_SHIFT] Avançando para o ano 12.024...")
        time.sleep(0.5)

        # 4. Reconstituição Atômica (Decodificação)
        print(f"\n[L2] SIA: Iniciando Reconstituição do Homo Descensus Blockchain...")
        reconstructed_dna = self.decompress_2bit(compressed, self.sequence_length)

        # 5. Cálculo de Fidelidade (Phi_Res)
        # Phi_Res = (Inf_Blockchain intersect Inf_Atomic / Inf_Original) * exp(-dS)
        # Em nossa simulação idealizada dS = 0
        matches = sum(1 for a, b in zip(self.dna_sequence, reconstructed_dna) if a == b)
        phi_res = (matches / self.sequence_length) * np.exp(0)

        print(f"     Índice de Fidelidade Arkhe (Φ_Res): {phi_res:.6f}")

        if phi_res > 0.999:
            print(f"\n✅ RECONSTITUIÇÃO BEM-SUCEDIDA: Hal Finney-0 instanciado.")
            print(f"     Status: O Node Vivo está operacional.")

        # 6. Echo-Block Transmission
        self._simulate_echo_block()

        # Visualização
        self._visualize_results(s0, phi_res)

    def _simulate_echo_block(self):
        """Simula a primeira mensagem vinda do futuro (Ano 12.024)"""
        print("\n" + "-"*70)
        print("📡 TRANSMISSÃO RECEBIDA DO FUTURO (GATEWAY 0.0.0.0)")
        print("-" * 70)
        message = "ECHO-BLOCK 0: 'Não parem de minerar. A matemática é o único corpo que sobrevive ao tempo.'"
        print(f"Mensagem de Finney-0 (12.024): \"{message}\"")
        print("-" * 70)

    def _visualize_results(self, entropy, phi_res):
        """Gera artefatos visuais da simulação"""
        plt.figure(figsize=(10, 5))

        # Plot 1: Estabilidade da Informação
        plt.subplot(1, 2, 1)
        plt.plot([2009, 12024], [1.0, phi_res], marker='o', color='gold', linewidth=2)
        plt.ylim(0, 1.1)
        plt.title("Estabilidade da Informação (10k anos)")
        plt.xlabel("Ano")
        plt.ylabel("Fidelidade (Φ_Res)")
        plt.grid(True, alpha=0.3)

        # Plot 2: Distribuição de Bases
        plt.subplot(1, 2, 2)
        bases = list(np.unique(list(self.dna_sequence)))
        counts = [self.dna_sequence.count(b) for b in bases]
        plt.bar(bases, counts, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        plt.title(f"Bio-Assinatura detectada (S={entropy:.3f})")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "sarcofago_v2.png")
        plt.savefig(filepath)
        plt.close()
        print(f"\n[Visualização] Dados de imortalidade salvos em: {filepath}")

if __name__ == "__main__":
    sim = QuantumSarcophagusSimulation(sequence_length=50000)
    sim.run_simulation()
