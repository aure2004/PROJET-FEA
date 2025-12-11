import numpy as np
import math


def nettoyer_entree(texte):
    if texte is None: return ""
    return texte.strip()


def lire_choix_menu(message, options_valides):
    while True:
        raw_input = input(message)
        choix = nettoyer_entree(raw_input)
        if choix in options_valides: return choix
        print(f"  [Erreur] Choix invalide ({', '.join(options_valides)}).")


def lire_oui_non(message):
    while True:
        reponse = nettoyer_entree(input(message)).lower()
        if reponse in ['y', 'o', 'yes', 'oui']: return True
        if reponse in ['n', 'no', 'non']: return False
        print("  [Erreur] Répondre par 'y' (oui) ou 'n' (non).")


def lire_int(message, min_val=None, max_val=None):
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


def clean_zeros(matrice, tolerance=1e-9):
    matrice_propre = np.copy(matrice)
    matrice_propre[np.abs(matrice_propre) < tolerance] = 0.0
    return matrice_propre


def definition_repere_utilisateur():
    print("\n=== CONFIGURATION DU REPÈRE ===")
    print("Orientation de l'axe X dans votre schéma ?")
    print("  1. Droite (Standard 0°)")
    print("  2. Bas    (-90°)")
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
    R_frame, angle_repere_deg = definition_repere_utilisateur()

    print("\nSAISIE GÉOMÉTRIE")
    Nn = lire_int("Nombre de nœuds : ", min_val=2)
    Nr = lire_int("Nombre de barres : ", min_val=1)

    print("\nCoordonnées (VOTRE repère)")
    Coord_Py = np.zeros((Nn, 2))
    for i in range(Nn):
        print(f"Noeud {i + 1} :")
        xu = lire_float("  X : ")
        yu = lire_float("  Y : ")
        Coord_Py[i] = to_python_vec(np.array([xu, yu]), R_frame)

    print("\nBarres")
    Connec = np.zeros((Nr, 2), dtype=int)
    Module = np.zeros(Nr)
    Section = np.zeros(Nr)

    if lire_oui_non("Propriétés (E, S) identiques ? (y/n) : "):
        E_g = lire_float("  E (Pa) : ", True)
        S_g = lire_float("  S (m^2) : ", True)
        Module.fill(E_g)
        Section.fill(S_g)
    else:
        for i in range(Nr):
            print(f"Barre {i + 1} :")
            Module[i] = lire_float(f"  E : ", True)
            Section[i] = lire_float(f"  S : ", True)

    for i in range(Nr):
        print(f"Barre {i + 1} Connectivité :")
        n1 = lire_int(f"  Départ (1-{Nn}) : ", 1, Nn) - 1
        n2 = lire_int(f"  Arrivée (1-{Nn}) : ", 1, Nn) - 1
        Connec[i] = [n1, n2]

    print("\nAppuis et Conditions Limites")
    BC_Type = np.zeros((Nn, 2), dtype=int)
    noeuds_traites = set()

    if lire_oui_non("Y a-t-il des nœuds bloqués ? (y/n) : "):
        print("Saisie des appuis (Tapez '0' comme numéro pour arrêter) :")
        count = 1
        while True:
            if len(noeuds_traites) == Nn:
                print("  [Info] Tous les nœuds sont configurés.")
                break
            entree = lire_int(f"  Appui {count} - Numéro du noeud (0 pour finir) : ", 0, Nn)
            if entree == 0: break
            idx = entree - 1
            if idx in noeuds_traites:
                print(f"  [ERREUR] Le noeud {entree} est déjà configuré.")
                continue

            print(f"    Pour le noeud {entree} :")
            if lire_oui_non("      Bloqué X ? (y/n) : "): BC_Type[idx, 0] = 1
            if lire_oui_non("      Bloqué Y ? (y/n) : "): BC_Type[idx, 1] = 1
            noeuds_traites.add(idx)
            count += 1

    print("\nForces")
    Fx_Py = np.zeros(Nn)
    Fy_Py = np.zeros(Nn)
    if lire_oui_non("Saisir des forces ? (y/n) : "):
        nb_f = lire_int("Nombre de nœuds chargés : ", 0, Nn)
        for i in range(nb_f):
            print(f"  Charge n°{i + 1} :")
            n = lire_int("    Numéro du noeud : ", 1, Nn) - 1

            norme = lire_float("    Norme de la force (N) : ")
            angle_f = lire_float("    Angle par rapport à votre axe X (deg) : ")

            rad = math.radians(angle_f)
            fx_user = norme * math.cos(rad)
            fy_user = norme * math.sin(rad)

            f_py = to_python_vec(np.array([fx_user, fy_user]), R_frame)
            Fx_Py[n] += f_py[0]
            Fy_Py[n] += f_py[1]

    return Coord_Py, Connec, Module, Section, BC_Type, Fx_Py, Fy_Py, R_frame

def solve_and_display(Coord, Connec, Module, Section, BC_Type, Fx, Fy, R_frame):
    # 1. INITIALISATION
    Nn = len(Coord)
    DDL = 2 * Nn
    K_global = np.zeros((DDL, DDL))
    F_global = np.zeros(DDL)

    for i in range(Nn):
        F_global[2 * i] = Fx[i]
        F_global[2 * i + 1] = Fy[i]

    # 2. ASSEMBLAGE
    for i in range(len(Connec)):
        n1, n2 = Connec[i]
        x1, y1 = Coord[n1]
        x2, y2 = Coord[n2]
        L = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if L < 1e-12: continue

        c = (x2 - x1) / L
        s = (y2 - y1) / L
        k_val = Module[i] * Section[i] / L

        Ke = k_val * np.array([[c * c, c * s, -c * c, -c * s],
                               [c * s, s * s, -c * s, -s * s],
                               [-c * c, -c * s, c * c, c * s],
                               [-c * s, -s * s, c * s, s * s]])

        idx = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        for r in range(4):
            for col in range(4):
                K_global[idx[r], idx[col]] += Ke[r, col]

    # 3. RÉDUCTION
    dofs_to_remove = []
    for i in range(Nn):
        if BC_Type[i, 0] == 1: dofs_to_remove.append(2 * i)  # Bloqué X (BC_Ux=1)
        if BC_Type[i, 1] == 1: dofs_to_remove.append(2 * i + 1)  # Bloqué Y (BC_Uy=1)

    dofs_to_remove = sorted(list(set(dofs_to_remove)))
    free_dofs = np.setdiff1d(np.arange(DDL), dofs_to_remove)

    K_red = K_global[np.ix_(free_dofs, free_dofs)]
    F_red = F_global[free_dofs]

    # 4. RÉSOLUTION
    try:
        U_red = np.linalg.solve(K_red, F_red)
    except np.linalg.LinAlgError:
        print("\n[ERREUR CRITIQUE] Matrice singulière (det=0). Structure instable.")
        return

    # 5. RECONSTRUCTION
    U_global = np.zeros(DDL)
    U_global[free_dofs] = U_red

    # 6. AFFICHAGE
    print("\n" + "=" * 30)
    print("           RÉSULTATS       ")
    print("=" * 30)

    # Affichage Matrice (Projetée pour lisibilité)
    T_disp = np.zeros((DDL, DDL))
    for i in range(Nn):
        T_disp[2 * i:2 * i + 2, 2 * i:2 * i + 2] = R_frame
    K_user_view = T_disp.T @ K_global @ T_disp

    np.set_printoptions(linewidth=300, precision=3, suppress=True)
    print("Matrice de Rigidité Globale :")
    print(clean_zeros(K_user_view))

    print("\nDéplacements aux Nœuds :")
    print(f"{'Nd':<3} | {'Ux (m)':<12} {'Uy (m)':<12}")
    print("-" * 35)

    for i in range(Nn):
        u_u = to_user_vec(U_global[2 * i:2 * i + 2], R_frame)
        print(f"{i + 1:<3} | {u_u[0]:12.4e} {u_u[1]:12.4e}")


if __name__ == "__main__":
    try:
        data = get_user_input()
        solve_and_display(*data)
    except KeyboardInterrupt:
        pass