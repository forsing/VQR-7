"""
VQR
Variational Quantum Regressor
"""


"""
Loto Skraceni Sistemi
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


"""
svih 4586 izvlacenja
30.07.1985.- 24.03.2026.
"""



from qiskit.circuit.library import TwoLocal, ZFeatureMap
from qiskit_machine_learning.algorithms import VQR

from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.optimizers import ADAM

from sklearn import model_selection
import numpy as np
import pandas as pd
import random

from qiskit_machine_learning.utils import algorithm_globals

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED


# Učitavanje poslednjih 100 redova iz CSV fajla
df = pd.read_csv("/Users/4c/Desktop/GHQ/data/loto7_4586_k24.csv", header=None)
 

N = 7  # broj qubita = broj feature-a

# v2: malo jači feature map i ansatz; finiji COBYLA
feature_map = ZFeatureMap(feature_dimension=N, reps=2)
ansatz = TwoLocal(num_qubits=N, rotation_blocks='ry', entanglement='cz', reps=3)

optimizer = COBYLA(maxiter=500, tol=1e-7)
# optimizer = SPSA(maxiter=300)
# optimizer = ADAM(maxiter=150, lr=0.1)


# =======================================
# PREDIKCIJA SLEDEĆE LOTO KOMBINACIJE (Korišćeno zadnjih 100 kombinacija)
# =======================================

print("\n=== Predikcija sledeće loto kombinacije 7/39 (na osnovu zadnjih 100 kombinacija) ===")

df_last100 = df.tail(100).reset_index(drop=True)

# Učitavanje celog seta
full_data = df_last100.values

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

    # v2: R² na trening skupu (isto X/Y kao fit)
    y_hat_scaled = vqr.predict(X_scaled).reshape(-1, 1)
    y_hat = scaler_y_pos.inverse_transform(y_hat_scaled).ravel()
    y_true = Y_full[:, i]
    r2 = r2_score(y_true, y_hat)
    print(f"R² train pozicija {i + 1}: {r2:.4f}")

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
feature_map = ZFeatureMap(feature_dimension=N, reps=2)

ansatz = TwoLocal(num_qubits=N, rotation_blocks='ry', entanglement='cz', reps=3)



=== Predikcija sledeće loto kombinacije 7/39 (na osnovu zadnjih 100 kombinacija) ===
=== Trening finalnog modela za POZICIJU 1 ===
No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
R² train pozicija 1: -0.2779
Predikcija za broj 1: x
=== Trening finalnog modela za POZICIJU 2 ===
No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
R² train pozicija 2: -0.6455
Predikcija za broj 2: y
=== Trening finalnog modela za POZICIJU 3 ===
No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
R² train pozicija 3: -0.9975
Predikcija za broj 3: 4
=== Trening finalnog modela za POZICIJU 4 ===
No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
R² train pozicija 4: -1.3017
Predikcija za broj 4: 7
=== Trening finalnog modela za POZICIJU 5 ===
No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
R² train pozicija 5: -2.0616
Predikcija za broj 5: 11
=== Trening finalnog modela za POZICIJU 6 ===
No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
R² train pozicija 6: -2.8277
Predikcija za broj 6: 16
=== Trening finalnog modela za POZICIJU 7 ===
No gradient function provided, creating a gradient function. If your Estimator requires transpilation, please provide a pass manager.
R² train pozicija 7: -7.7157
Predikcija za broj 7: z

=== Predviđena sledeća loto kombinacija 7/39 ===
x y 4 7 11 16 z
"""





"""
Poboljšanja u VQR_7_v2.py su konkretno:

jači kvantni izraz modela: ZFeatureMap(reps=2) umesto 1
dublji ansatz: TwoLocal(..., reps=3) umesto 2
finiji optimizer: COBYLA(maxiter=500, tol=1e-7) umesto 300 i 1e-6
dodat merni signal kvaliteta: ispis R² train za svaku poziciju (1..7), da vidiš odmah gde model puca
zadržan isti tok i isti izlaz (predikcija 7 brojeva), bez menjanja originalnog VQR_7.py
"""
