import numpy as np

def kl_divergence(p, q):
    """
    Calculates the Kullback-Leibler divergence between two distributions.
    D_KL(P || Q) = sum(P(i) * log(P(i) / Q(i)))
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Ensure normalization
    p /= p.sum()
    q /= q.sum()

    # Avoid log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)

    return np.sum(p * np.log(p / q))

class PoBFProtocol:
    """
    Proof of Biological Fidelity (PoBF)
    Bridges genomic information with blockchain hash entropy.
    """
    def __init__(self, dna_distribution=None):
        # Human DNA distribution (approximate frequencies of A, T, C, G)
        # pA ≈ pT ≈ 0.3, pG ≈ pC ≈ 0.2
        if dna_distribution is None:
            self.dna_dist = np.array([0.3, 0.3, 0.2, 0.2]) # A, T, C, G
        else:
            self.dna_dist = dna_distribution

    def validate_block(self, block_hash):
        """
        Calculates the fidelity between a block hash and biological entropy.
        """
        # Convert hex hash to a 4-bin distribution (2 bits per bin)
        hash_bytes = bytes.fromhex(block_hash)
        bins = np.zeros(4)
        for byte in hash_bytes:
            # Simple mapping of byte values to 4 bins
            bins[0] += (byte & 0x3)
            bins[1] += ((byte >> 2) & 0x3)
            bins[2] += ((byte >> 4) & 0x3)
            bins[3] += ((byte >> 6) & 0x3)

        bins /= bins.sum()

        # Calculate D_KL
        fidelity_divergence = kl_divergence(self.dna_dist, bins)

        # Phi_Res calculation (simplified)
        phi_res = np.exp(-fidelity_divergence)

        return {
            "divergence": fidelity_divergence,
            "phi_res": phi_res,
            "status": "Validated" if phi_res > 0.8 else "Fidelity Low"
        }

if __name__ == "__main__":
    pobf = PoBFProtocol()

    # Simulating a "Biological Block Hash" (one that closely matches DNA distribution)
    # A real random hash would have uniform distribution [0.25, 0.25, 0.25, 0.25]
    print("--- PoBF Protocol: Proof of Biological Fidelity ---")

    # Test 1: Random Hash
    random_hash = "0000000000000000000abbccddeeff00112233445566778899aabbccddeeff"
    res1 = pobf.validate_block(random_hash)
    print(f"Random Hash Fidelity: Φ_Res = {res1['phi_res']:.4f} ({res1['status']})")

    # Test 2: 'Optimized' Biological Hash (simulated)
    bio_optimized_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    res2 = pobf.validate_block(bio_optimized_hash)
    print(f"Bio-Optimized Hash Fidelity: Φ_Res = {res2['phi_res']:.4f} ({res2['status']})")

    print("\n'Não otimizem apenas a escassez; otimizem a fidelidade da informação biológica.'")
