import numpy as np

# Constantes
k = 8.988e9  # Constante de Coulomb (N·m²/C²)
mu0 = 4 * np.pi * 1e-7  # Permeabilidad magnética del vacío (T·m/A)

# Parámetros de la espira y cargas
q = 12e-9  # Carga de cada punto (12 nC)
I = 4  # Corriente en la espira (4 A)
N_cargas_por_lado = 10  # Número de cargas por lado
N_segmentos_por_lado = 10  # Número de segmentos por lado
lado = 0.4  # Tamaño de la espira cuadrada (metros)
half_lado = lado / 2

# Generamos las posiciones de las cargas en la espira cuadrada en el plano YZ
cargas_posiciones = []

# Lado inferior (de (-0.2, -0.2) a (0.2, -0.2)) en el plano YZ
y_inf = np.linspace(-half_lado, half_lado, N_cargas_por_lado, endpoint=False)
z_inf = np.full(N_cargas_por_lado, -half_lado)
for y, z in zip(y_inf, z_inf):
    cargas_posiciones.append([0, y, z])

# Lado derecho (de (0.2, -0.2) a (0.2, 0.2)) en el plano YZ
y_der = np.full(N_cargas_por_lado, half_lado)
z_der = np.linspace(-half_lado, half_lado, N_cargas_por_lado, endpoint=False)
for y, z in zip(y_der, z_der):
    cargas_posiciones.append([0, y, z])

# Lado superior (de (0.2, 0.2) a (-0.2, 0.2)) en el plano YZ
y_sup = np.linspace(half_lado, -half_lado, N_cargas_por_lado, endpoint=False)
z_sup = np.full(N_cargas_por_lado, half_lado)
for y, z in zip(y_sup, z_sup):
    cargas_posiciones.append([0, y, z])

# Lado izquierdo (de (-0.2, 0.2) a (-0.2, -0.2)) en el plano YZ
y_izq = np.full(N_cargas_por_lado, -half_lado)
z_izq = np.linspace(half_lado, -half_lado, N_cargas_por_lado, endpoint=False)
for y, z in zip(y_izq, z_izq):
    cargas_posiciones.append([0, y, z])

# Convertimos la lista a un array numpy
cargas_posiciones = np.array(cargas_posiciones)

# Generamos las posiciones y vectores dl de los segmentos para el cálculo del campo magnético en el plano YZ
segmentos_posiciones = []

# Lado inferior
y_inf_seg = np.linspace(-half_lado, half_lado, N_segmentos_por_lado + 1)
z_inf_seg = np.full(N_segmentos_por_lado + 1, -half_lado)
for i in range(N_segmentos_por_lado):
    inicio = np.array([0, y_inf_seg[i], z_inf_seg[i]])
    fin = np.array([0, y_inf_seg[i+1], z_inf_seg[i+1]])
    segmentos_posiciones.append((inicio, fin))

# Lado derecho
y_der_seg = np.full(N_segmentos_por_lado + 1, half_lado)
z_der_seg = np.linspace(-half_lado, half_lado, N_segmentos_por_lado + 1)
for i in range(N_segmentos_por_lado):
    inicio = np.array([0, y_der_seg[i], z_der_seg[i]])
    fin = np.array([0, y_der_seg[i+1], z_der_seg[i+1]])
    segmentos_posiciones.append((inicio, fin))

# Lado superior
y_sup_seg = np.linspace(half_lado, -half_lado, N_segmentos_por_lado + 1)
z_sup_seg = np.full(N_segmentos_por_lado + 1, half_lado)
for i in range(N_segmentos_por_lado):
    inicio = np.array([0, y_sup_seg[i], z_sup_seg[i]])
    fin = np.array([0, y_sup_seg[i+1], z_sup_seg[i+1]])
    segmentos_posiciones.append((inicio, fin))

# Lado izquierdo
y_izq_seg = np.full(N_segmentos_por_lado + 1, -half_lado)
z_izq_seg = np.linspace(half_lado, -half_lado, N_segmentos_por_lado + 1)
for i in range(N_segmentos_por_lado):
    inicio = np.array([0, y_izq_seg[i], z_izq_seg[i]])
    fin = np.array([0, y_izq_seg[i+1], z_izq_seg[i+1]])
    segmentos_posiciones.append((inicio, fin))

# Generamos la cuadrícula de puntos en el plano XZ
grid_size = 11
x_grid = np.linspace(-1, 1, grid_size)
z_grid = np.linspace(-1, 1, grid_size)
X, Z = np.meshgrid(x_grid, z_grid)
Y = np.zeros_like(X)  # El plano XZ está en Y=0

# Preparamos el archivo para guardar los resultados del campo eléctrico
with open('campo_electrico.txt', 'w') as file_e:
    # Iteramos sobre cada punto de la cuadrícula
    for i in range(grid_size):
        for j in range(grid_size):
            punto = np.array([X[i, j], Y[i, j], Z[i, j]])
            campo_total = np.zeros(3)
            # Calculamos el campo eléctrico debido a cada carga
            for carga_pos in cargas_posiciones:
                r_vector = punto - carga_pos
                r_mag = np.linalg.norm(r_vector)
                if r_mag == 0:
                    continue  # Evitamos división por cero
                r_hat = r_vector / r_mag
                campo = (k * q / r_mag**2) * r_hat
                campo_total += campo
            # Guardamos el resultado en el archivo
            file_e.write(f"Punto ({X[i,j]:.2f}, {Y[i,j]:.2f}, {Z[i,j]:.2f}): Campo Eléctrico = ({campo_total[0]:.2e}, {campo_total[1]:.2e}, {campo_total[2]:.2e})\n")

print("El cálculo del campo eléctrico ha finalizado y los resultados se han guardado en 'campo_electrico.txt'.")

# Preparamos el archivo para guardar los resultados del campo magnético
with open('campo_magnetico.txt', 'w') as file_b:
    # Iteramos sobre cada punto de la cuadrícula
    for i in range(grid_size):
        for j in range(grid_size):
            punto = np.array([X[i, j], Y[i, j], Z[i, j]])
            campo_total = np.zeros(3)
            # Calculamos el campo magnético debido a cada segmento de corriente
            for segmento in segmentos_posiciones:
                r0 = segmento[0]
                r1 = segmento[1]
                dl = r1 - r0  # Vector diferencial de longitud
                r = punto - r0  # Vector desde el inicio del segmento al punto
                r_mag = np.linalg.norm(r)
                if r_mag == 0:
                    continue  # Evitamos división por cero
                dB = (mu0 * I / (4 * np.pi)) * np.cross(dl, r) / r_mag**3
                campo_total += dB
            # Guardamos el resultado en el archivo
            file_b.write(f"Punto ({X[i,j]:.2f}, {Y[i,j]:.2f}, {Z[i,j]:.2f}): Campo Magnético = ({campo_total[0]:.2e}, {campo_total[1]:.2e}, {campo_total[2]:.2e})\n")

print("El cálculo del campo magnético ha finalizado y los resultados se han guardado en 'campo_magnetico.txt'.")
