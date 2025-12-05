import numpy as np
import matplotlib.pyplot as plt
import math


# --- FONCTIONS UTILITAIRES POUR LA SAISIE ROBUSTE ---

def lire_oui_non(message):
    """Demande une réponse y/n et boucle tant que la réponse n'est pas valide."""
    while True:
        reponse = input(message).strip().lower()
        if reponse == 'y':
            return True
        elif reponse == 'n':
            return False
        else:
            print("  [Erreur] Veuillez répondre uniquement par 'y' (yes) ou 'n' (no).")


def lire_int(message, min_val=None, max_val=None):
    """Demande un entier et vérifie qu'il est dans les bornes."""
    while True:
        try:
            valeur = int(input(message))
            if min_val is not None and valeur < min_val:
                print(f"  [Erreur] La valeur doit être supérieure ou égale à {min_val}.")
                continue
            if max_val is not None and valeur > max_val:
                print(f"  [Erreur] La valeur doit être inférieure ou égale à {max_val}.")
                continue
            return valeur
        except ValueError:
            print("  [Erreur] Veuillez entrer un nombre entier valide.")


def lire_float(message, positive_only=False):
    """Demande un nombre décimal."""
    while True:
        try:
            valeur = float(input(message))
            if positive_only and valeur <= 0:
                print("  [Erreur] La valeur doit être strictement positive.")
                continue
            return valeur
        except ValueError:
            print("  [Erreur] Veuillez entrer un nombre décimal valide.")


# --- 1. PRÉ-TRAITEMENT (SAISIE UTILISATEUR) ---

def get_user_input():
    print("=== DÉBUT DE LA SAISIE DES DONNÉES ===")

    # 1. Nombre de noeuds et de barres
    Nn = lire_int("Entrez le nombre de nœuds (Nn) : ", min_val=2)
    Nr = lire_int("Entrez le nombre de barres (Nr) : ", min_val=1)

    # 2. Coordonnées (Matrice Nn x 2)
    print(f"\n--- Saisie des Coordonnées (Matrice {Nn}x2) ---")
    Coord = np.zeros((Nn, 2))
    for i in range(Nn):
        print(f"Noeud {i + 1} :")
        x = lire_float(f"  Coordonnée X : ")
        y = lire_float(f"  Coordonnée Y : ")
        Coord[i] = [x, y]

    # 3. Connectivité, Module et Section (MODIFIÉ)
    print(f"\n--- Saisie des Barres (Matrice {Nr}x2) et Propriétés ---")
    Connec = np.zeros((Nr, 2), dtype=int)
    Module = np.zeros(Nr)
    Section = np.zeros(Nr)

    # On demande si c'est identique pour tout le monde
    props_identiques = lire_oui_non("Les propriétés (E, S) sont-elles identiques pour toutes les barres ? (y/n) : ")

    # Si oui, on demande les valeurs une seule fois maintenant
    if props_identiques:
        print("  > Saisie des propriétés globales :")
        E_global = lire_float(f"    Module de Young E (Pa) : ", positive_only=True)
        S_global = lire_float(f"    Section S (m^2) : ", positive_only=True)
        # On remplit tout le tableau d'un coup
        Module.fill(E_global)
        Section.fill(S_global)

    for i in range(Nr):
        print(f"Barre {i + 1} :")
        # On demande toujours la connectivité
        n1 = lire_int(f"  Numéro du noeud de départ (1 à {Nn}) : ", min_val=1, max_val=Nn) - 1
        n2 = lire_int(f"  Numéro du noeud d'arrivée (1 à {Nn}) : ", min_val=1, max_val=Nn) - 1

        while n1 == n2:
            print("  [Erreur] Le noeud de départ et d'arrivée doivent être différents.")
            n2 = lire_int(f"  Numéro du noeud d'arrivée (1 à {Nn}) : ", min_val=1, max_val=Nn) - 1

        Connec[i] = [n1, n2]

        # Si NON identique, on demande E et S à chaque tour
        if not props_identiques:
            E = lire_float(f"  Module de Young E (Pa) : ", positive_only=True)
            S = lire_float(f"  Section S (m^2) : ", positive_only=True)
            Module[i] = E
            Section[i] = S

    # 4. Conditions Limites (CL)
    print("\n--- Conditions Limites (Appuis) ---")
    BC_Ux = np.zeros(Nn, dtype=int)
    BC_Uy = np.zeros(Nn, dtype=int)

    nb_appuis = lire_int("Combien de noeuds ont des appuis (bloqués) ? ", min_val=1, max_val=Nn)

    for _ in range(nb_appuis):
        node_idx = lire_int(f"  Numéro du noeud bloqué (1 à {Nn}) : ", min_val=1, max_val=Nn) - 1
        print(f"  Pour le noeud {node_idx + 1} :")
        bloque_x = lire_oui_non("    Bloqué en X ? (y/n) : ")
        bloque_y = lire_oui_non("    Bloqué en Y ? (y/n) : ")

        if bloque_x: BC_Ux[node_idx] = 1
        if bloque_y: BC_Uy[node_idx] = 1

    VAL_Ux = np.zeros(Nn)
    VAL_Uy = np.zeros(Nn)

    # 5. Forces Nodales
    print("\n--- Chargement (Forces) ---")
    VAL_Fx = np.zeros(Nn)
    VAL_Fy = np.zeros(Nn)

    nb_forces = lire_int("Combien de noeuds subissent une force ? ", min_val=1, max_val=Nn)

    for _ in range(nb_forces):
        node_idx = lire_int(f"  Numéro du noeud chargé (1 à {Nn}) : ", min_val=1, max_val=Nn) - 1
        fx = lire_float(f"    Force en X (N) : ")
        fy = lire_float(f"    Force en Y (N) : ")
        VAL_Fx[node_idx] += fx
        VAL_Fy[node_idx] += fy

    return Coord, Connec, Module, Section, BC_Ux, BC_Uy, VAL_Ux, VAL_Uy, VAL_Fx, VAL_Fy


# --- 2. RÉSOLUTION (SOLVER) ---

def solve_truss(Coord, Connec, Module, Section, BC_Ux, BC_Uy, VAL_Ux, VAL_Uy, VAL_Fx, VAL_Fy):
    Nn = len(Coord)
    Nr = len(Connec)
    DoF = 2 * Nn

    K_global = np.zeros((DoF, DoF))
    F_global = np.zeros(DoF)

    for i in range(Nn):
        F_global[2 * i] = VAL_Fx[i]
        F_global[2 * i + 1] = VAL_Fy[i]

    print("\n[INFO] Assemblage de la matrice de rigidité...")
    for i in range(Nr):
        n1 = Connec[i, 0]
        n2 = Connec[i, 1]

        x1, y1 = Coord[n1]
        x2, y2 = Coord[n2]

        L_e = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L_e
        s = (y2 - y1) / L_e

        k_val = (Module[i] * Section[i]) / L_e

        Ke = k_val * np.array([
            [c ** 2, c * s, -c ** 2, -c * s],
            [c * s, s ** 2, -c * s, -s ** 2],
            [-c ** 2, -c * s, c ** 2, c * s],
            [-c * s, -s ** 2, c * s, s ** 2]
        ])

        indices = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]

        for row in range(4):
            for col in range(4):
                K_global[indices[row], indices[col]] += Ke[row, col]

    free_dofs = []
    for i in range(Nn):
        if BC_Ux[i] == 0: free_dofs.append(2 * i)
        if BC_Uy[i] == 0: free_dofs.append(2 * i + 1)

    K_red = K_global[np.ix_(free_dofs, free_dofs)]
    F_red = F_global[free_dofs]

    try:
        U_red = np.linalg.solve(K_red, F_red)
    except np.linalg.LinAlgError:
        print("\n[ERREUR] La matrice est singulière ! Vérifiez vos conditions limites.")
        return None, None

    U_final = np.zeros(DoF)
    U_final[free_dofs] = U_red

    return U_final, K_global


# --- 3. POST-TRAITEMENT (RÉSULTATS) ---

def post_process(U_final, K_global, Coord, Connec, Module, Section):
    if U_final is None: return

    Nn = len(Coord)
    Nr = len(Connec)

    F_calc = K_global @ U_final

    print("\n" + "=" * 30)
    print("       RÉSULTATS FINAUX       ")
    print("=" * 30)

    print("\n--- 0. Matrice de Rigidité Globale (K) ---")
    np.set_printoptions(precision=1, linewidth=200, suppress=True)
    print(K_global)
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False,
                        threshold=1000, formatter=None)

    print("\n--- 1. Déplacements Nodaux (m) ---")
    print(f"{'Noeud':<6} {'Ux (m)':<15} {'Uy (m)':<15}")
    for i in range(Nn):
        print(f"{i + 1:<6} {U_final[2 * i]:.4e}     {U_final[2 * i + 1]:.4e}")

    print("\n--- 2. Forces de Réaction (N) ---")
    print(f"{'Noeud':<6} {'Rx (N)':<15} {'Ry (N)':<15}")
    for i in range(Nn):
        rx = F_calc[2 * i] if abs(F_calc[2 * i]) > 1e-5 else 0.0
        ry = F_calc[2 * i + 1] if abs(F_calc[2 * i + 1]) > 1e-5 else 0.0
        print(f"{i + 1:<6} {rx:.2f}          {ry:.2f}")

    print("\n--- 3. Efforts Normaux dans les barres (N) ---")
    print(f"{'Barre':<6} {'Noeuds':<10} {'Effort (N)':<15} {'État'}")
    for i in range(Nr):
        n1, n2 = Connec[i]
        x1, y1 = Coord[n1]
        x2, y2 = Coord[n2]
        L_e = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L_e
        s = (y2 - y1) / L_e

        u_elem = np.array([U_final[2 * n1], U_final[2 * n1 + 1], U_final[2 * n2], U_final[2 * n2 + 1]])
        B_stress = (Module[i] * Section[i] / L_e) * np.array([-c, -s, c, s])

        Normal_Force = np.dot(B_stress, u_elem)
        state = "Traction" if Normal_Force > 0 else "Compression"
        print(f"{i + 1:<6} {n1 + 1}-{n2 + 1:<7} {Normal_Force:.2f}          {state}")

    # Visualisation
    scale_factor = 1.0
    max_disp = np.max(np.abs(U_final))
    if max_disp > 0:
        L_mean = np.mean(
            [math.sqrt((Coord[c[1]][0] - Coord[c[0]][0]) ** 2 + (Coord[c[1]][1] - Coord[c[0]][1]) ** 2) for c in
             Connec])
        scale_factor = 0.15 * L_mean / max_disp

    print(f"\n[INFO] Facteur d'échelle graphique : x{scale_factor:.1f}")

    plt.figure(figsize=(8, 6))
    for i in range(Nr):
        n1, n2 = Connec[i]
        plt.plot([Coord[n1, 0], Coord[n2, 0]], [Coord[n1, 1], Coord[n2, 1]], 'b--o', alpha=0.5,
                 label='Initial' if i == 0 else "")
        if i == 0:
            for j in range(Nn):
                plt.text(Coord[j, 0], Coord[j, 1], f" {j + 1}", color='blue', fontsize=12)

    New_Coord = np.zeros_like(Coord)
    for i in range(Nn):
        New_Coord[i, 0] = Coord[i, 0] + U_final[2 * i] * scale_factor
        New_Coord[i, 1] = Coord[i, 1] + U_final[2 * i + 1] * scale_factor

    for i in range(Nr):
        n1, n2 = Connec[i]
        plt.plot([New_Coord[n1, 0], New_Coord[n2, 0]], [New_Coord[n1, 1], New_Coord[n2, 1]], 'r-x', linewidth=2,
                 label='Déformée' if i == 0 else "")

    plt.title(f"Structure (Échelle x{scale_factor:.1f})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


# --- MAIN ---
if __name__ == "__main__":
    data = get_user_input()

    if data is not None:
        Coord, Connec, Module, Section, BC_Ux, BC_Uy, VAL_Ux, VAL_Uy, VAL_Fx, VAL_Fy = data
        U_final, K_global = solve_truss(Coord, Connec, Module, Section, BC_Ux, BC_Uy, VAL_Ux, VAL_Uy, VAL_Fx, VAL_Fy)
        post_process(U_final, K_global, Coord, Connec, Module, Section)