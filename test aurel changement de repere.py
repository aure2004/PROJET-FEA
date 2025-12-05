import numpy as np
import matplotlib.pyplot as plt
import math


# ==============================================================================
# 1. OUTILS DE SAISIE SÉCURISÉE ET ROBUSTE
# ==============================================================================

def nettoyer_entree(texte):
    """Retire les espaces avant/après."""
    if texte is None: return ""
    return texte.strip()


def lire_choix_menu(message, options_valides):
    """Force un choix parmi une liste (ex: ['1', '2'])."""
    while True:
        raw_input = input(message)
        choix = nettoyer_entree(raw_input)
        if choix in options_valides:
            return choix
        print(f"  [Erreur] Choix invalide ({', '.join(options_valides)}).")


def lire_oui_non(message):
    """Retourne True pour 'y'/'o', False pour 'n'."""
    while True:
        raw_input = input(message)
        reponse = nettoyer_entree(raw_input).lower()
        if reponse in ['y', 'o', 'yes', 'oui']: return True
        if reponse in ['n', 'no', 'non']: return False
        print("  [Erreur] Répondre par 'y' (oui) ou 'n' (non).")


def lire_int(message, min_val=None, max_val=None):
    """Demande un entier validé."""
    while True:
        try:
            valeur = int(nettoyer_entree(input(message)))
            if min_val is not None and valeur < min_val:
                print(f"  [Erreur] Valeur >= {min_val} requise.")
                continue
            if max_val is not None and valeur > max_val:
                print(f"  [Erreur] Valeur <= {max_val} requise.")
                continue
            return valeur
        except ValueError:
            print("  [Erreur] Entier invalide.")


def lire_float(message, positive_only=False):
    """Demande un float validé."""
    while True:
        try:
            cleaned = nettoyer_entree(input(message)).replace(',', '.')
            valeur = float(cleaned)
            if positive_only and valeur <= 0:
                print("  [Erreur] Valeur positive requise.")
                continue
            return valeur
        except ValueError:
            print("  [Erreur] Nombre invalide.")


def lire_liste_ints(message, min_val=1, max_val=None):
    """Lit une liste d'entiers séparés par des espaces (ex: '2 3')."""
    while True:
        try:
            raw = input(message)
            parts = raw.split()
            if not parts: return []  # Liste vide

            valeurs = []
            for p in parts:
                v = int(p)
                if min_val is not None and v < min_val:
                    raise ValueError(f"Le noeud {v} est < {min_val}.")
                if max_val is not None and v > max_val:
                    raise ValueError(f"Le noeud {v} n'existe pas (Max {max_val}).")
                valeurs.append(v)
            return valeurs
        except ValueError as e:
            print(f"  [Erreur] Saisie invalide ({e}). Entrez les numéros séparés par un espace.")


# ==============================================================================
# 2. SAISIE DES DONNÉES (LOGIQUE AMÉLIORÉE)
# ==============================================================================

def definition_repere_utilisateur():
    print("\n=== CONFIGURATION DU REPÈRE ===")
    print("Orientation de l'axe X dans votre schéma ?")
    print("  1. Droite (Standard 0°)")
    print("  2. Bas    (Type TD4 -90°)")
    print("  3. Haut   (+90°)")
    print("  4. Gauche (180°)")

    choix = lire_choix_menu("  Votre choix (1-4) : ", ['1', '2', '3', '4'])

    angle = 0.0
    if choix == '2':
        angle = -90.0
    elif choix == '3':
        angle = 90.0
    elif choix == '4':
        angle = 180.0

    theta = math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    R_frame = np.array([[c, -s], [s, c]])
    return R_frame, angle


def to_python_vec(vec_user, R_frame): return R_frame @ vec_user


def to_user_vec(vec_python, R_frame): return R_frame.T @ vec_python


def get_user_input():
    R_frame, angle_deg = definition_repere_utilisateur()

    print("\n=== SAISIE GÉOMÉTRIE ===")
    Nn = lire_int("Nombre de nœuds : ", min_val=2)
    Nr = lire_int("Nombre de barres : ", min_val=1)

    print("\n--- Coordonnées (VOTRE repère) ---")
    Coord_Py = np.zeros((Nn, 2))
    for i in range(Nn):
        print(f"Noeud {i + 1} :")
        xu = lire_float("  X : ")
        yu = lire_float("  Y : ")
        Coord_Py[i] = to_python_vec(np.array([xu, yu]), R_frame)

    print("\n--- Barres ---")
    Connec = np.zeros((Nr, 2), dtype=int)
    Module = np.zeros(Nr)
    Section = np.zeros(Nr)

    if lire_oui_non("Propriétés (E, S) identiques pour toutes les barres ? (y/n) : "):
        E_g = lire_float("  E (Pa) : ", True)
        S_g = lire_float("  S (m^2) : ", True)
        Module.fill(E_g)
        Section.fill(S_g)
        identique = True
    else:
        identique = False

    for i in range(Nr):
        print(f"Barre {i + 1} :")
        n1 = lire_int(f"  Départ (1-{Nn}) : ", 1, Nn) - 1
        n2 = lire_int(f"  Arrivée (1-{Nn}) : ", 1, Nn) - 1
        while n1 == n2:
            print("  [Erreur] Départ = Arrivée impossible.")
            n2 = lire_int(f"  Arrivée (1-{Nn}) : ", 1, Nn) - 1
        Connec[i] = [n1, n2]
        if not identique:
            Module[i] = lire_float("  E (Pa) : ", True)
            Section[i] = lire_float("  S (m^2) : ", True)

    # --- Saisie des Appuis (Nouvelle Logique) ---
    print("\n--- Appuis et Conditions Limites ---")
    BC_Type = np.zeros((Nn, 2), dtype=int)
    BC_Incline = {}

    # 1. Appuis Inclinés
    if lire_oui_non("Y a-t-il des appuis INCLINÉS (sur pente) ? (y/n) : "):
        print(f"  Quels sont les numéros des nœuds inclinés ? (ex: 3 4)")
        liste_inclines = lire_liste_ints(f"  Numéros (1-{Nn}) : ", 1, Nn)

        for num_noeud in liste_inclines:
            idx = num_noeud - 1
            print(f"  -> Pour le noeud {num_noeud} :")
            angle_pente = lire_float("     Angle de la pente par rapport à VOTRE axe X (degrés) : ")
            BC_Incline[idx] = angle_pente + 90.0  # On stocke la normale
            print(f"     [OK] Appui glissant configuré (Normale à {angle_pente + 90}°).")

    # 2. Appuis Standards
    nb_std = lire_int(f"Combien d'AUTRES nœuds sont bloqués ? : ", 0, Nn)

    for _ in range(nb_std):
        n = lire_int(f"  Numéro du noeud standard (1-{Nn}) : ", 1, Nn) - 1

        # Vérification pour ne pas écraser un appui incliné
        if n in BC_Incline:
            print(f"  [Attention] Le noeud {n + 1} est déjà défini comme incliné. Ignoré.")
            continue

        print(f"  Pour le noeud {n + 1} :")
        if lire_oui_non("    Bloqué en X ? (y/n) : "): BC_Type[n, 0] = 1
        if lire_oui_non("    Bloqué en Y ? (y/n) : "): BC_Type[n, 1] = 1

    print("\n--- Forces ---")
    Fx_Py = np.zeros(Nn)
    Fy_Py = np.zeros(Nn)
    nb_f = lire_int(f"Nombre de nœuds chargés (0-{Nn}) : ", 0, Nn)
    for _ in range(nb_f):
        n = lire_int(f"  Numéro du noeud (1-{Nn}) : ", 1, Nn) - 1
        fx = lire_float("    Force X : ")
        fy = lire_float("    Force Y : ")
        f_py = to_python_vec(np.array([fx, fy]), R_frame)
        Fx_Py[n] += f_py[0]
        Fy_Py[n] += f_py[1]

    return Coord_Py, Connec, Module, Section, BC_Type, BC_Incline, Fx_Py, Fy_Py, R_frame, angle_deg


# ==============================================================================
# 3. SOLVEUR (Inchangé - Calculs Matriciels)
# ==============================================================================

def solve_truss(Coord, Connec, Module, Section, BC_Type, BC_Incline, Fx, Fy, R_frame, angle_user_deg):
    Nn = len(Coord)
    DoF = 2 * Nn
    K_global = np.zeros((DoF, DoF))
    F_global = np.zeros(DoF)

    for i in range(Nn):
        F_global[2 * i] = Fx[i]
        F_global[2 * i + 1] = Fy[i]

    for i in range(len(Connec)):
        n1, n2 = Connec[i]
        x1, y1 = Coord[n1]
        x2, y2 = Coord[n2]
        L = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c, s = (x2 - x1) / L, (y2 - y1) / L
        k = Module[i] * Section[i] / L

        Ke = k * np.array([[c * c, c * s, -c * c, -c * s],
                           [c * s, s * s, -c * s, -s * s],
                           [-c * c, -c * s, c * c, c * s],
                           [-c * s, -s * s, c * s, s * s]])

        idx = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        for r in range(4):
            for col in range(4):
                K_global[idx[r], idx[col]] += Ke[r, col]

    T_struct = np.eye(DoF)
    for node_idx, angle_normale in BC_Incline.items():
        rad = math.radians(angle_normale + angle_user_deg)
        c_n, s_n = math.cos(rad), math.sin(rad)
        T_struct[2 * node_idx:2 * node_idx + 2, 2 * node_idx:2 * node_idx + 2] = np.array([[c_n, -s_n], [s_n, c_n]])

    K_rot = T_struct.T @ K_global @ T_struct
    F_rot = T_struct.T @ F_global

    dofs_remove = []
    for i in range(Nn):
        if i in BC_Incline:
            dofs_remove.append(2 * i)
        else:
            if BC_Type[i, 0]: dofs_remove.append(2 * i)
            if BC_Type[i, 1]: dofs_remove.append(2 * i + 1)

    free = np.setdiff1d(np.arange(DoF), dofs_remove)
    K_red = K_rot[np.ix_(free, free)]
    F_red = F_rot[free]

    try:
        U_red = np.linalg.solve(K_red, F_red)
    except:
        return None, None, None

    U_full_rot = np.zeros(DoF)
    U_full_rot[free] = U_red
    U_final = T_struct @ U_full_rot

    return U_final, K_global, K_red


# ==============================================================================
# 4. POST-TRAITEMENT
# ==============================================================================

def post_process(U, K, Kred, Coord, Connec, Module, Section, R_frame):
    if U is None:
        print("\n[ERREUR CRITIQUE] Matrice singulière (Structure instable).")
        return

    Nn = len(Coord)
    F_reac = K @ U

    print("\n" + "=" * 60)
    print("              RÉSULTATS (VOTRE REPÈRE)       ")
    print("=" * 60)

    print("\n--- 0. Matrices ---")
    np.set_printoptions(precision=1, linewidth=200, suppress=True)

    DoF = 2 * Nn
    T_user = np.zeros((DoF, DoF))
    for i in range(Nn):
        T_user[2 * i:2 * i + 2, 2 * i:2 * i + 2] = R_frame
    K_user_view = T_user.T @ K @ T_user

    print("Matrice Globale Assemblée (Dans votre repère) :")
    print(K_user_view)
    print("\nMatrice Réduite (Calcul) :")
    print(Kred)
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False,
                        threshold=1000, formatter=None)

    print("\n--- 1. Déplacements & Réactions ---")
    print(f"{'Nd':<3} | {'Ux (m)':<12} {'Uy (m)':<12} | {'Rx (N)':<12} {'Ry (N)':<12}")
    print("-" * 65)

    for i in range(Nn):
        u_u = to_user_vec(U[2 * i:2 * i + 2], R_frame)
        f_u = to_user_vec(F_reac[2 * i:2 * i + 2], R_frame)

        ux = u_u[0] if abs(u_u[0]) > 1e-12 else 0.0
        uy = u_u[1] if abs(u_u[1]) > 1e-12 else 0.0
        rx = f_u[0] if abs(f_u[0]) > 1e-5 else 0.0
        ry = f_u[1] if abs(f_u[1]) > 1e-5 else 0.0
        print(f"{i + 1:<3} | {ux:12.4e} {uy:12.4e} | {rx:12.1f} {ry:12.1f}")

    print("\n--- 2. Efforts Barres ---")
    print(f"{'Barre':<6} {'Effort (N)':<12} {'État'}")
    for i in range(len(Connec)):
        n1, n2 = Connec[i]
        x1, y1 = Coord[n1]
        x2, y2 = Coord[n2]
        L = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c, s = (x2 - x1) / L, (y2 - y1) / L
        u_el = np.array([U[2 * n1], U[2 * n1 + 1], U[2 * n2], U[2 * n2 + 1]])
        N = (Module[i] * Section[i] / L) * np.dot([-c, -s, c, s], u_el)
        etat = "Traction" if N > 0 else "Compress."
        val_N = N if abs(N) > 1e-5 else 0.0
        print(f"{i + 1:<6} {val_N:<12.1f} {etat}")

    print("\n[INFO] Graphique (Repère écran standard)")
    plt.figure()
    max_d = np.max(np.abs(U))
    scale = 1.0
    if max_d > 0:
        L_moy = np.mean([math.sqrt((Coord[c[1]][0] - Coord[c[0]][0]) ** 2) for c in Connec])
        scale = 0.15 * L_moy / max_d

    for i in range(len(Connec)):
        n1, n2 = Connec[i]
        plt.plot([Coord[n1, 0], Coord[n2, 0]], [Coord[n1, 1], Coord[n2, 1]], 'b--o', alpha=0.4)
        if i == 0:
            for j in range(len(Coord)): plt.text(Coord[j, 0], Coord[j, 1], f"{j + 1}", color='b')
        x1d, y1d = Coord[n1] + U[2 * n1:2 * n1 + 2] * scale
        x2d, y2d = Coord[n2] + U[2 * n2:2 * n2 + 2] * scale
        plt.plot([x1d, x2d], [y1d, y2d], 'r-')

    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        data = get_user_input()
        U, K, Kred = solve_truss(*data)
        post_process(U, K, Kred, *data[:4], data[-2])
    except KeyboardInterrupt:
        pass