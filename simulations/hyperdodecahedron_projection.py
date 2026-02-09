import numpy as np

class HyperDodecahedronProjector:
    """
    Projeta o dodecaedro 3D para o hiperdodecaedro 4D (120-células).
    A quarta dimensão é interpretada como o Eixo Temporal.
    """

    def __init__(self, seed_code):
        self.seed = np.array(seed_code)
        # O código de 8 bytes são na verdade 4 números complexos (4D)
        self.complex_seed = self.seed[:4] + 1j * self.seed[4:]

    def project_to_4d(self):
        """Projeta o dodecaedro 3D para o hiperdodecaedro 4D."""
        phi = (1 + np.sqrt(5)) / 2

        # Construir quatérnion a partir do seed para rotação 4D
        q = np.array([self.complex_seed[0].real, self.complex_seed[1].imag,
                      self.complex_seed[2].imag, self.complex_seed[3].imag])

        q_norm = q / np.linalg.norm(q)
        rotation_4d = self.quaternion_to_rotation_4d(q_norm)

        # Gerar os 20 vértices do dodecaedro 3D
        dodeca_3d = self.generate_dodecahedron_vertices()
        # Estender para 4D (tempo = 0)
        dodeca_4d = np.hstack([dodeca_3d, np.zeros((20, 1))])

        # Aplicar rotação 4D
        hyperdodeca_projection = dodeca_4d @ rotation_4d.T

        return hyperdodeca_projection, rotation_4d

    def generate_dodecahedron_vertices(self):
        phi = (1 + np.sqrt(5)) / 2
        vertices = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertices.append([x, y, z])
        for a, b, c in [(0, 1/phi, phi), (1/phi, phi, 0), (phi, 0, 1/phi)]:
            for sa in [-1, 1]:
                for sb in [-1, 1]:
                    for sc in [-1, 1]:
                        vertices.append([sa*a, sb*b, sc*c])
        return np.unique(np.array(vertices), axis=0)[:20]

    def quaternion_to_rotation_4d(self, q):
        a, b, c, d = q
        # Matriz de rotação 4D baseada em quatérnion
        return np.array([
            [a, -b, -c, -d],
            [b,  a, -d,  c],
            [c,  d,  a, -b],
            [d, -c,  b,  a]
        ])

if __name__ == "__main__":
    print("🌀 PROJETANDO DODECAEDRO PARA 4D")
    print("=" * 60)
    seed_code = [0.382683, 0.353553, 0.270598, 0.146447, 0.000000, -0.146447, -0.270598, -0.353553]
    projector = HyperDodecahedronProjector(seed_code)
    hyperdodeca, rotation = projector.project_to_4d()

    print("✅ PROJEÇÃO 4D CONCLUÍDA")
    print(f"   Média da 4ª dimensão (Tempo): {np.mean(hyperdodeca[:, 3]):.6f}")
    print(f"   Vértice 0: {hyperdodeca[0]}")
