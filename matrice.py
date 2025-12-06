import numpy as np
import math


def matrice_td4_propre():
    print("\n=== MATRICE DE RIGIDITÉ (TD4 Ex 1) - COEFFICIENTS PURS ===")
    print("Hypothèse : On pose E=1, S=1, L=1 pour voir les coefficients géométriques.")

    # 1. Paramètres normalisés
    E = 1.0
    S = 1.0
    L_ref = 1.0  # Hauteur de la barre verticale

    # 2. Coordonnées (Repère TD4 : X vertical descendant, Y droite)
    # A=(0,0), B=(-1,-1), C=(-1,0), D=(-1,1)
    nodes = {
        1: np.array([0.0, 0.0]),  # A
        2: np.array([-1.0, -1.0]),  # B
        3: np.array([-1.0, 0.0]),  # C (Verticale)
        4: np.array([-1.0, 1.0])  # D
    }

    # 3. Connectivité
    elements = [
        (1, 2),  # A-B
        (1, 3),  # A-C
        (1, 4)  # A-D
    ]

    # 4. Assemblage
    NDOF = 8  # 4 noeuds * 2 ddl
    K = np.zeros((NDOF, NDOF))

    for n1, n2 in elements:
        p1 = nodes[n1]
        p2 = nodes[n2]
        vec = p2 - p1
        L = np.linalg.norm(vec)

        # Cosinus directeurs
        c = vec[0] / L
        s = vec[1] / L

        # Raideur k = ES/L
        k_elt = (E * S) / L

        # Matrice élémentaire 4x4
        m = np.array([
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s]
        ])

        ke = k_elt * m

        # Indices globaux (u1,v1, u2,v2...)
        idx = [2 * (n1 - 1), 2 * (n1 - 1) + 1, 2 * (n2 - 1), 2 * (n2 - 1) + 1]

        for i in range(4):
            for j in range(4):
                K[idx[i], idx[j]] += ke[i, j]

    # 5. Nettoyage radical des zéros (Bruit numérique)
    K[np.abs(K) < 1e-9] = 0.0

    # 6. Affichage Spécifique pour validation
    np.set_printoptions(precision=3, linewidth=150, suppress=True)

    print("\n--- MATRICE GLOBALE (A, B, C, D) ---")
    print(K)

    print("\n--- VÉRIFICATION DES TERMES CLÉS (Noeud A) ---")
    # K11 = Rigidité verticale en A
    val_k11 = K[0, 0]
    # Calcul théorique : 1 (barre vert.) + 2 * c^3 (barres diag)
    # avec theta=45, c = 1/sqrt(2) approx 0.707
    # c^3 = 0.3535... => 2*c^3 = 0.707... => Total = 1.707
    print(f"K(uA, uA) calculé par code : {val_k11:.4f}")
    print(f"Théorie (1 + 2c^3)         : {1.0 + 2.0 * (1.0 / math.sqrt(2)) ** 3:.4f}")

    # K12 = Couplage uA, vA
    val_k12 = K[0, 1]
    print(f"K(uA, vA) calculé par code : {val_k12:.4f}")
    print(f"Théorie (0 - Symétrie)     : 0.0000")

    # Terme croisé barre verticale (A vers C)
    # Ligne 0 (uA), Colonne 4 (uC) -> Doit être -1
    val_k_ac = K[0, 4]
    print(f"K(uA, uC) calculé par code : {val_k_ac:.4f}")
    print(f"Théorie (-1)               : -1.0000")


if __name__ == "__main__":
    matrice_td4_propre()