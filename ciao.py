from qiskit.quantum_info import Operator
from inspect import trace
import numpy as np
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
import qiskit as qc
from qiskit_aer import Aer
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp

n = 4  # Numero di qubit

def pauli_matrices_divided():
    sqrt2 = np.sqrt(2)
    return {
        'I': np.array([[1, 0], [0, 1]], dtype=complex) / sqrt2,
        'X': np.array([[0, 1], [1, 0]], dtype=complex) / sqrt2,
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex) / sqrt2,
        'Z': np.array([[1, 0], [0, -1]], dtype=complex) / sqrt2
    }

# Creazione delle matrici di Pauli divise
pauli_matrices = pauli_matrices_divided()

# Definizione delle etichette delle matrici di Pauli
matrix_labels = list(pauli_matrices.keys())  # ['I', 'X', 'Y', 'Z']

# 2. Genera tutti i possibili prodotti tensore e definisce la "binary tuple"
tensor_products = []
for combination in itertools.product(matrix_labels, repeat=n):
    tensor = pauli_matrices[combination[0]]
    for label in combination[1:]:
        tensor = np.kron(tensor, pauli_matrices[label])
    binary_tuple = tuple(1 if label != 'I' else 0 for label in combination)
    tensor_products.append((binary_tuple, tensor))

# Ordinamento in base al valore della tuple (facoltativo)
tensor_products.sort(key=lambda item: sum(bit * (2**i) for i, bit in enumerate(item[0])))

# Raggruppa i tensori in base alla binary tuple
groups = defaultdict(list)
for binary_tuple, tensor in tensor_products:
    groups[binary_tuple].append(tensor)

# Lista ordinata delle tuple uniche e definizione del numero J di gruppi
unique_tuples = sorted(groups.keys(), key=lambda tup: sum(bit * (2**i) for i, bit in enumerate(tup)))
J = len(unique_tuples)



############_-------------------------------------------------
# 3. Definizione della matrice H (in questo caso 4 gate pauli X)

# Applica rotazioni casuali
parameters = 2 * np.pi * np.random.rand(2 * n)
circuit = QuantumCircuit(n)

circuit.cx(0,1)
for i in range (n-2):
  circuit.cx(i+1, i+2)
  circuit.cx(i, i+1)
circuit.cx(2,3)

H=Operator(circuit)
H=np.array(H.data)
#-----------------------------------

# 4 Costruzione della matrice T e di l:
# Per ciascun gruppo j (per tensori B) e per ciascun gruppo k (per tensori P)
# si somma (con normalizzazione) il contributo: (trace(H† B H P))^2.



T = np.zeros((J, J), dtype=complex)




for j_idx, tup_j in enumerate(unique_tuples):
    LT_vec = np.zeros(J, dtype=complex)
    # Calcola il fattore di normalizzazione per il gruppo B: per ogni '1' nella tuple si ha un fattore 3
    norm_factor_B = 3 ** (sum(tup_j))

    # Per ogni tensore B nel gruppo j
    for B in groups[tup_j]:
        # Ciclo sui gruppi per P (indice k)
        for k_idx, tup_k in enumerate(unique_tuples):
            trace_sum = 0
            # Fattore di normalizzazione per il gruppo P
            norm_factor_P = 3 ** (sum(tup_k))
            for P in groups[tup_k]:
                trace_val = np.trace(H.conj().T @ B @ H @ P)
                # Normalizzazione: il contributo viene diviso per il prodotto dei fattori per B e per P
                trace_sum += (trace_val**2) / (norm_factor_B )
            LT_vec[k_idx] += trace_sum
    T[j_idx, :] = LT_vec



# 5. Diagonalizzazione a blocchi fortemente connessi tramite permutazioni
# Costruiamo una "matrice di connettività": consideriamo un arco tra i nodi i e j se |T[i,j]| > tol
tol = 1e-16
connectivity = (np.abs(T) > tol).astype(int)

# Trova i componenti connessi (trattando la matrice come grafo non diretto)
n_components, comp_labels = connected_components(connectivity, directed=False, connection='weak')

# La permutazione che raggruppa per componente è data da:
permutation = np.argsort(comp_labels)
# Riordiniamo T per ottenere una forma bloc-diagonale
T_perm = T[permutation, :][:, permutation]

# Salviamo la mappatura della permutazione: (nuovo indice -> indice originale)
perm_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(permutation)}

# 6. Plot della matrice T permutata come heatmap (si plottano i valori reali)
plt.figure(figsize=(8, 6))
plt.imshow(np.real(T_perm), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xlabel("Entrata località")
plt.ylabel("Uscita località")
plt.title("Heatmap della matrice T slow-entangle")
plt.show()

# Salva (o stampa) la mappatura della permutazione
print("Mappatura della permutazione (nuovo indice -> indice originale):")
print(perm_mapping)
