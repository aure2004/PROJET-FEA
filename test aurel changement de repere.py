import numpy as np
import matplotlib.pyplot as plt
import math


# ==============================================================================
# 1. OUTILS ET FONCTIONS UTILITAIRES
# ==============================================================================

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
    """Met à 0.0 tous les termes inférieurs à la tolérance (bruit numérique)."""
    matrice_propre = np.copy(matrice)
    matrice_propre[np.abs(matrice_propre) < tolerance] = 0.0
    return matrice_propre


# ==============================================================================
# 2. SAISIE DES DONNÉES (LOGIQUE INCLINAISON CORRIGÉE)
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

    # Nettoyage du bruit
    if abs(c) < 1e-10: c = 0.0
    if abs(s) < 1e-10: s = 0.0

    R_frame = np.array([[c, -s], [s, c]])
    return R_frame, angle


def to_python_vec(vec_user, R_frame): return R_frame @ vec_user


def get_user_input():
    R_frame, angle_repere_deg = definition_repere_utilisateur()

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

    if lire_oui_non("Propriétés (E, S) identiques ? (y/n) : "):
        E_g = lire_float("  E (Pa) : ", True)
        S_g = lire_float("  S (m^2) : ", True)
        Module.fill(E_g)
        Section.fill(S_g)
    else:
        for i in range(Nr):
            Module[i] = lire_float(f"  Barre {i + 1} E : ", True)
            Section[i] = lire_float(f"  Barre {i + 1} S : ", True)

    for i in range(Nr):
        print(f"Barre {i + 1} :")
        n1 = lire_int(f"  Départ (1-{Nn}) : ", 1, Nn) - 1
        n2 = lire_int(f"  Arrivée (1-{Nn}) : ", 1, Nn) - 1
        Connec[i] = [n1, n2]

    # --- GESTION DES APPUIS (Nouvelle Logique) ---
    print("\n--- Appuis et Conditions Limites ---")
    BC_Type = np.zeros((Nn, 2), dtype=int)
    BC_Incline = {}  # Stocke {indice_noeud: angle_deg}
    noeuds_traites = []  # Pour ne pas redemander les noeuds inclinés

    # 1. Noeuds Inclinés
    if lire_oui_non("Y a-t-il des appuis INCLINÉS (sur pente) ? (y/n) : "):
        nb_inc = lire_int(f"Combien de nœuds inclinés ? (Max {Nn}) : ", 1, Nn)

        for k in range(nb_inc):
            print(f"  >> Appui Incliné n°{k + 1} :")
            # Boucle pour s'assurer qu'on ne saisit pas deux fois le même
            while True:
                num = lire_int(f"     Numéro du noeud (1-{Nn}) : ", 1, Nn)
                idx = num - 1
                if idx in noeuds_traites:
                    print(f"     [Info] Le noeud {num} est déjà traité. Choisissez-en un autre.")
                else:
                    noeuds_traites.append(idx)
                    break

            angle = lire_float("     Angle de l'inclinaison (degrés par rapport à X) : ")
            BC_Incline[idx] = angle
            print(f"     [OK] Noeud {num} configuré avec inclinaison {angle}°.")

    # 2. Noeuds Standards (Blocage X/Y)
    nb_restants = Nn - len(noeuds_traites)
    if nb_restants > 0:
        if lire_oui_non(f"Y a-t-il d'autres nœuds bloqués (Standards X/Y) ? (y/n) : "):
            nb_std = lire_int(f"Combien de nœuds standards bloqués ? (Max {nb_restants}) : ", 0, nb_restants)

            for k in range(nb_std):
                print(f"  >> Appui Standard n°{k + 1} :")
                while True:
                    num = lire_int(f"     Numéro du noeud (1-{Nn}) : ", 1, Nn)
                    idx = num - 1
                    if idx in noeuds_traites:
                        print(f"     [Erreur] Le noeud {num} est déjà défini comme incliné (ou déjà traité).")
                    else:
                        noeuds_traites.append(idx)  # On le marque comme traité
                        break

                if lire_oui_non("     Bloqué en X ? (y/n) : "): BC_Type[idx, 0] = 1
                if lire_oui_non("     Bloqué en Y ? (y/n) : "): BC_Type[idx, 1] = 1

    print("\n--- Forces ---")
    Fx_Py = np.zeros(Nn)
    Fy_Py = np.zeros(Nn)
    if lire_oui_non("Saisir des forces ? (y/n) : "):
        nb_f = lire_int(f"Nombre de nœuds chargés : ", 0, Nn)
        for _ in range(nb_f):
            n = lire_int(f"  Numéro du noeud : ", 1, Nn) - 1
            fx = lire_float("    Force X : ")
            fy = lire_float("    Force Y : ")
            f_py = to_python_vec(np.array([fx, fy]), R_frame)
            Fx_Py[n] += f_py[0]
            Fy_Py[n] += f_py[1]

    return Coord_Py, Connec, Module, Section, BC_Type, BC_Incline, Fx_Py, Fy_Py, R_frame


# ==============================================================================
# 3. SOLVEUR ET POST-TRAITEMENT
# ==============================================================================

def solve_and_display(Coord, Connec, Module, Section, BC_Type, BC_Incline, Fx, Fy, R_frame):
    Nn = len(Coord)
    DoF = 2 * Nn
    K_global = np.zeros((DoF, DoF))

    # Assemblage (inchangé)
    for i in range(len(Connec)):
        n1, n2 = Connec[i]
        x1, y1 = Coord[n1]
        x2, y2 = Coord[n2]
        L = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if L < 1e-12: continue

        c = (x2 - x1) / L
        s = (y2 - y1) / L
        if abs(c) < 1e-10: c = 0.0
        if abs(s) < 1e-10: s = 0.0

        k = Module[i] * Section[i] / L

        Ke = k * np.array([[c * c, c * s, -c * c, -c * s],
                           [c * s, s * s, -c * s, -s * s],
                           [-c * c, -c * s, c * c, c * s],
                           [-c * s, -s * s, c * s, s * s]])

        idx = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        for r in range(4):
            for col in range(4):
                K_global[idx[r], idx[col]] += Ke[r, col]

    # --- MATRICE DE PASSAGE GLOBALE ---
    # T_struct va contenir :
    # - La rotation du repère Python -> Repère Utilisateur (R_frame) pour les noeuds standards
    # - La rotation du repère Python -> Repère Incliné pour les noeuds inclinés

    T_struct = np.zeros((DoF, DoF))

    for i in range(Nn):
        # Par défaut, on applique R_frame pour revenir dans le repère utilisateur X,Y
        mat_rot_noeud = R_frame

        # Si le noeud est incliné, on ajoute la rotation spécifique de l'inclinaison
        if i in BC_Incline:
            angle_inc = BC_Incline[i]
            # Rotation de l'inclinaison par rapport au repère utilisateur
            rad = math.radians(angle_inc)
            c_inc, s_inc = math.cos(rad), math.sin(rad)
            R_inc = np.array([[c_inc, -s_inc], [s_inc, c_inc]])

            # La transformation totale est : Repère Python -> Repère Utilisateur -> Repère Incliné
            # On veut afficher la matrice dans le repère de l'appui (incliné)
            # U_py = R_frame @ U_user
            # U_user = R_inc @ U_inc
            # Donc U_py = (R_frame @ R_inc) @ U_inc

            # NOTE : R_frame est la matrice de passage User -> Python
            mat_rot_noeud = R_frame @ R_inc

        T_struct[2 * i:2 * i + 2, 2 * i:2 * i + 2] = mat_rot_noeud

    # Calcul de la matrice à afficher (K dans les repères locaux/utilisateurs)
    # K_display = T^T * K_global * T
    K_display = T_struct.T @ K_global @ T_struct

    K_final = clean_zeros(K_display)

    print("\n" + "=" * 60)
    print("           MATRICE DE RIGIDITÉ (VOTRE REPÈRE)      ")
    print("=" * 60)
    print("NOTE : Pour les nœuds inclinés, la matrice est exprimée dans leur repère local (x', y').")

    np.set_printoptions(precision=2, linewidth=200, suppress=True, floatmode='fixed')
    print("\nMatrice K :")
    print(K_final)

    print("\nLégende des lignes/colonnes :")
    for i in range(Nn):
        if i in BC_Incline:
            angle = BC_Incline[i]
            print(f"  {2 * i}: u{i + 1}' (Incliné {angle}°) |  {2 * i + 1}: v{i + 1}' (Perpendiculaire)")
        else:
            print(f"  {2 * i}: u{i + 1}  (X)            |  {2 * i + 1}: v{i + 1}  (Y)")


if __name__ == "__main__":
    try:
        data = get_user_input()
        solve_and_display(*data)
    except KeyboardInterrupt:
        pass