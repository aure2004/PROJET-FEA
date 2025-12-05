import numpy as np
import matplotlib.pyplot as plt
import math


# --- 1. PRÉ-TRAITEMENT (DATA) ---

def get_input_data():
    # Paramètres du TD4 Ex 1
    P_val = 1000.0  # Force en Newton
    L = 1.0  # Longueur de référence en mètres
    E_val = 210e9  # Module de Young (Pa)
    S_val = 6e-4  # Section (m^2)

    # Angles (TD4: alpha=0, beta=45 deg)
    beta = math.radians(45)
    alpha = math.radians(0)

    # 1. Coordonnées des Noeuds (Nn x 2)
    # On place A en (0,0). B, C, D sont au-dessus.
    # A=Node 0, B=Node 1, C=Node 2, D=Node 3 (Attention: Python commence à 0)
    # Note: Dans le PDF TD4, A est le noeud 1. Ici on adapte les indices.
    dx = L * math.tan(beta)
    Coord = np.array([
        [0.0, 0.0],  # Node 0 (A)
        [-dx, L],  # Node 1 (B)
        [0.0, L],  # Node 2 (C)
        [dx, L]  # Node 3 (D)
    ])

    # 2. Connectivité (Nr x 2) - Indices des noeuds
    # Barre 1 (A-B), Barre 2 (A-C), Barre 3 (A-D)
    Connec = np.array([
        [0, 1],
        [0, 2],
        [0, 3]
    ], dtype=int)

    # 3. Propriétés des matériaux (Nr x 1)
    # Attention: TD4 Ex 2 dit S pour 1 et 2, S*sqrt(2) pour 3.
    # Mais TD4 Ex 1 (Part B) dit "identical section S". On suit Ex 1 Part B.
    N_rods = Connec.shape[0]
    Module = np.full(N_rods, E_val)
    Section = np.full(N_rods, S_val)

    # 4. Conditions Limites (CL)
    # Format suggéré par le PDF Mini-Project
    Nn = Coord.shape[0]

    # Tableaux pour indiquer si un DDL est bloqué (1) ou libre (0)
    BC_Ux = np.zeros(Nn, dtype=int)
    BC_Uy = np.zeros(Nn, dtype=int)

    # On bloque B(1), C(2), D(3) en X et Y
    BC_Ux[1] = 1;
    BC_Uy[1] = 1
    BC_Ux[2] = 1;
    BC_Uy[2] = 1
    BC_Ux[3] = 1;
    BC_Uy[3] = 1

    # Valeurs imposées (ici 0 partout car supports fixes)
    VAL_Ux = np.zeros(Nn)
    VAL_Uy = np.zeros(Nn)

    # 5. Forces Nodales
    # Force P appliquée en A(0) avec angle alpha par rapport à la verticale
    # Fx = P * sin(alpha), Fy = -P * cos(alpha) (selon repère Y vers le haut)
    VAL_Fx = np.zeros(Nn)
    VAL_Fy = np.zeros(Nn)

    VAL_Fx[0] = P_val * math.sin(alpha)
    VAL_Fy[0] = -P_val * math.cos(alpha)  # Négatif car vers le bas

    return Coord, Connec, Module, Section, BC_Ux, BC_Uy, VAL_Ux, VAL_Uy, VAL_Fx, VAL_Fy


# --- 2. RÉSOLUTION (SOLVER) ---

def solve_truss(Coord, Connec, Module, Section, BC_Ux, BC_Uy, VAL_Ux, VAL_Uy, VAL_Fx, VAL_Fy):
    Nn = len(Coord)
    Nr = len(Connec)
    DoF = 2 * Nn  # Degrés de liberté total

    # Initialisation Matrice de Rigidité Globale et Vecteur Forces
    K_global = np.zeros((DoF, DoF))
    F_global = np.zeros(DoF)

    # Remplissage du vecteur Force global
    for i in range(Nn):
        F_global[2 * i] = VAL_Fx[i]
        F_global[2 * i + 1] = VAL_Fy[i]

    # --- Assemblage ---
    print("--- Calcul des Matrices Élémentaires et Assemblage ---")
    for i in range(Nr):
        n1 = Connec[i, 0]
        n2 = Connec[i, 1]

        # Coordonnées
        x1, y1 = Coord[n1]
        x2, y2 = Coord[n2]

        # Longueur et Orientation
        L_e = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L_e  # cosinus
        s = (y2 - y1) / L_e  # sinus

        # Matrice élémentaire (Formule du Chapitre 5 / Mini-Project PDF)
        k_val = (Module[i] * Section[i]) / L_e

        # Matrice 4x4 locale (u1, v1, u2, v2)
        Ke = k_val * np.array([
            [c ** 2, c * s, -c ** 2, -c * s],
            [c * s, s ** 2, -c * s, -s ** 2],
            [-c ** 2, -c * s, c ** 2, c * s],
            [-c * s, -s ** 2, c * s, s ** 2]
        ])

        # Indices dans la matrice globale
        # Noeuds n1 et n2. DDLs: 2*n1, 2*n1+1, 2*n2, 2*n2+1
        indices = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]

        # Assemblage
        for row in range(4):
            for col in range(4):
                K_global[indices[row], indices[col]] += Ke[row, col]

    # --- Application des Conditions Limites (Méthode de Réduction) ---
    # On identifie les lignes/colonnes à garder (celles où le déplacement est LIBRE)
    free_dofs = []
    for i in range(Nn):
        if BC_Ux[i] == 0: free_dofs.append(2 * i)
        if BC_Uy[i] == 0: free_dofs.append(2 * i + 1)

    # Extraction de la matrice réduite et du vecteur force réduit
    # Numpy permet de faire ça très facilement avec le slicing
    K_red = K_global[np.ix_(free_dofs, free_dofs)]
    F_red = F_global[free_dofs]

    # Résolution
    U_red = np.linalg.solve(K_red, F_red)

    # Reconstitution du vecteur déplacement complet
    U_final = np.zeros(DoF)
    # On remplit avec les déplacements calculés
    U_final[free_dofs] = U_red
    # (Les déplacements imposés non-nuls devraient être gérés ici si VAL_Ux != 0, 
    # mais pour ce projet les appuis sont souvent fixes = 0)

    return U_final, K_global


# --- 3. POST-TRAITEMENT (RESULTS) ---

def post_process(U_final, K_global, Coord, Connec, Module, Section):
    Nn = len(Coord)
    Nr = len(Connec)

    # 1. Calcul des Réactions
    # F = K * U. Les forces aux noeuds bloqués sont les réactions.
    F_calc = K_global @ U_final

    print("\n--- Déplacements Nodaux (m) ---")
    for i in range(Nn):
        print(f"Noeud {i + 1} (A={1}): Ux = {U_final[2 * i]:.4e}, Uy = {U_final[2 * i + 1]:.4e}")

    print("\n--- Forces de Réaction (N) ---")
    # On affiche les réactions là où BC = 1 (bloqué)
    # Ici on affiche tout pour vérifier l'équilibre
    for i in range(Nn):
        print(f"Noeud {i + 1}: Rx = {F_calc[2 * i]:.2f}, Ry = {F_calc[2 * i + 1]:.2f}")

    print("\n--- Efforts Normaux dans les barres (N) ---")
    # N = (ES/L) * [-c -s c s] * {u_elem}
    for i in range(Nr):
        n1, n2 = Connec[i]
        x1, y1 = Coord[n1]
        x2, y2 = Coord[n2]
        L_e = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L_e
        s = (y2 - y1) / L_e

        u_elem = np.array([U_final[2 * n1], U_final[2 * n1 + 1], U_final[2 * n2], U_final[2 * n2 + 1]])

        # Vecteur de transition pour l'effort normal
        B_stress = (Module[i] * Section[i] / L_e) * np.array([-c, -s, c, s])

        Normal_Force = np.dot(B_stress, u_elem)
        print(f"Barre {i + 1} ({n1 + 1}-{n2 + 1}): Effort = {Normal_Force:.2f} N")

    # 2. Visualisation (Déformée vs Initiale)
    scale_factor = 1000  # Pour mieux voir la déformation (c'est très petit sinon)

    # Structure Initiale
    for i in range(Nr):
        n1, n2 = Connec[i]
        plt.plot([Coord[n1, 0], Coord[n2, 0]], [Coord[n1, 1], Coord[n2, 1]], 'b-o', label='Initial' if i == 0 else "")

    # Structure Déformée
    New_Coord = np.zeros_like(Coord)
    for i in range(Nn):
        New_Coord[i, 0] = Coord[i, 0] + U_final[2 * i] * scale_factor
        New_Coord[i, 1] = Coord[i, 1] + U_final[2 * i + 1] * scale_factor

    for i in range(Nr):
        n1, n2 = Connec[i]
        plt.plot([New_Coord[n1, 0], New_Coord[n2, 0]], [New_Coord[n1, 1], New_Coord[n2, 1]], 'r--x',
                 label=f'Déformée (x{scale_factor})' if i == 0 else "")

    plt.title("Structure: Initiale vs Déformée")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


# --- MAIN ---
if __name__ == "__main__":
    # 1. Obtenir les données
    Coord, Connec, Module, Section, BC_Ux, BC_Uy, VAL_Ux, VAL_Uy, VAL_Fx, VAL_Fy = get_input_data()

    # 2. Résoudre
    U_final, K_global = solve_truss(Coord, Connec, Module, Section, BC_Ux, BC_Uy, VAL_Ux, VAL_Uy, VAL_Fx, VAL_Fy)

    # 3. Résultats
    post_process(U_final, K_global, Coord, Connec, Module, Section)