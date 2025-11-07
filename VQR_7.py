# https://medium.com/quantum-computing-and-ai-ml/variational-quantum-circuits-for-machine-learning-aa982a03605f

"""
Variational Quantum Circuits
Variational Quantum Regressor
"""

from qiskit.circuit.library import TwoLocal, ZFeatureMap
from qiskit_machine_learning.algorithms import VQR
from qiskit_algorithms.optimizers import ADAM

from sklearn import model_selection
import numpy as np
import pandas as pd
import random

from qiskit_machine_learning.utils import algorithm_globals

from sklearn.preprocessing import MinMaxScaler


# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED


# Učitavanje poslednjih 100 redova iz CSV fajla
df = pd.read_csv("/Users/milan/Desktop/GHQ/data/loto7_4506_k87.csv", header=None)


N = 7  # broj qubita = broj feature-a

feature_map = ZFeatureMap(feature_dimension=N, reps=1)
ansatz = TwoLocal(num_qubits=N, rotation_blocks='ry', entanglement='cz', reps=2)
optimizer = ADAM(maxiter=150, lr=0.1)


# =======================================
# PREDIKCIJA SLEDEĆE LOTO KOMBINACIJE (Korišćeno zadnjih 1000 kombinacija)
# =======================================

print("\n=== Predikcija sledeće loto kombinacije 7/39 (na osnovu zadnjih 1000 kombinacija) ===")

df_last1000 = df.tail(1000).reset_index(drop=True)

# Učitavanje celog seta
full_data = df_last1000.values

# X - sve osim poslednje
X_full = full_data[:-1]
# Y - pomereno za 1 unapred
Y_full = full_data[1:]


#######################


# Skaliranje X
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X_full)

# Poslednja kombinacija za predikciju
last_scaled = scaler_X.transform([X_full[-1]]).astype(np.float64)

predicted_combination = []

for i in range(7):
    print(f"=== Trening finalnog modela za POZICIJU {i + 1} ===")

    # Skaliranje Y za tu poziciju
    scaler_y_pos = MinMaxScaler(feature_range=(0, 1))
    y_scaled_pos = scaler_y_pos.fit_transform(Y_full[:, i].reshape(-1, 1))

    # Kreiranje i treniranje VQR modela
    vqr = VQR(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )
    vqr.fit(X_scaled, y_scaled_pos.ravel())

    # Predikcija sledećeg broja
    pred_scaled = vqr.predict(last_scaled)
    pred = scaler_y_pos.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    pred = max(1, min(39, int(round(pred))))  # ograničenje 1–39

    predicted_combination.append(pred)
    print(f"Predikcija za broj {i + 1}: {pred}")

print("\n=== Predviđena sledeća loto kombinacija 7/39 ===")
print(" ".join(str(num) for num in predicted_combination))
print()


"""
=== Predikcija sledeće loto kombinacije 7/39 (na osnovu svih 4506) ===
=== Trening finalnog modela za POZICIJU 1 ===
Predikcija za broj 1: 1
=== Trening finalnog modela za POZICIJU 2 ===
Predikcija za broj 2: 2
=== Trening finalnog modela za POZICIJU 3 ===
Predikcija za broj 3: 3
=== Trening finalnog modela za POZICIJU 4 ===
Predikcija za broj 4: 5
=== Trening finalnog modela za POZICIJU 5 ===
Predikcija za broj 5: 8
=== Trening finalnog modela za POZICIJU 6 ===
Predikcija za broj 6: 11
=== Trening finalnog modela za POZICIJU 7 ===
Predikcija za broj 7: 15

=== Predviđena sledeća loto kombinacija 7/39 ===
1 2 3 5 8 11 15
"""
