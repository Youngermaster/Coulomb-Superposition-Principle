import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Constantes
k = 8.988e9  # Constante de Coulomb (N·m²/C²)
mu0 = 4 * np.pi * 1e-7  # Permeabilidad magnética del vacío (T·m/A)

# Parámetros de la espira y cargas
# q = 12e-9  # Carga de cada punto (12 nC)
q = 3e-10  # Carga de cada punto (0,3 nC)
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

# Visualización
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Intercambiamos Y y Z en las gráficas para que Y sea vertical
# Graficamos las cargas en la espira cuadrada
cargas_x = cargas_posiciones[:, 0]
cargas_y = cargas_posiciones[:, 1]
cargas_z = cargas_posiciones[:, 2]
ax.scatter(cargas_x, cargas_z, cargas_y, color='red', label='Cargas en la espira')  # Intercambiamos Y y Z

# Graficamos los segmentos de la espira
for segmento in segmentos_posiciones:
    inicio = segmento[0]
    fin = segmento[1]
    xs = [inicio[0], fin[0]]
    zs = [inicio[2], fin[2]]  # Z
    ys = [inicio[1], fin[1]]  # Y
    ax.plot(xs, zs, ys, color='blue')  # Intercambiamos Y y Z

# Graficamos la cuadrícula en el plano XZ
grid_scatter = ax.scatter(X, Z, Y, color='green', s=20, alpha=0.5, label='Puntos de la cuadrícula')  # Intercambiamos Y y Z

# Configuramos las etiquetas y límites de los ejes
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')  # Ahora Z es horizontal
ax.set_zlabel('Y (m)')  # Ahora Y es vertical
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Ajustamos la vista para que el eje Z sea horizontal y Y vertical
ax.view_init(elev=0, azim=-90)

# Preparamos arrays para almacenar los campos eléctricos en los puntos de la cuadrícula
E_total = np.zeros((grid_size, grid_size, 3))

# Calculamos el campo eléctrico en cada punto de la cuadrícula
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
        E_total[i, j] = campo_total

# Visualización del campo eléctrico
E_magnitude = np.linalg.norm(E_total, axis=2)
E_magnitude[E_magnitude == 0] = 1e-20  # Evitar división por cero
E_normalized = E_total / E_magnitude[:, :, np.newaxis]

# Preparamos arrays para almacenar los campos magnéticos en los puntos de la cuadrícula
B_total = np.zeros((grid_size, grid_size, 3))

# Calculamos el campo magnético en cada punto de la cuadrícula
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
        B_total[i, j] = campo_total

# Visualización del campo magnético
B_magnitude = np.linalg.norm(B_total, axis=2)
B_magnitude[B_magnitude == 0] = 1e-20  # Evitar división por cero
B_normalized = B_total / B_magnitude[:, :, np.newaxis]

# Podemos reducir el número de vectores para evitar una gráfica sobrecargada
skip = (slice(None, None, 2), slice(None, None, 2))

# Graficamos los vectores del campo eléctrico en los puntos de la cuadrícula
E_quiver = ax.quiver(X[skip], Z[skip], Y[skip], # Intercambiamos Y y Z
                     E_normalized[skip][:, :, 0], E_normalized[skip][:, :, 2], E_normalized[skip][:, :, 1],
                     length=0.1, color='black', normalize=True)

# Graficamos los vectores del campo magnético en los puntos de la cuadrícula
B_quiver = ax.quiver(X[skip], Z[skip], Y[skip], # Intercambiamos Y y Z
                     B_normalized[skip][:, :, 0], B_normalized[skip][:, :, 2], B_normalized[skip][:, :, 1],
                     length=0.1, color='blue', normalize=True)

# Añadimos los "chart legends"
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Cargas en la espira',
           markerfacecolor='red', markersize=5),
    Line2D([0], [0], color='blue', lw=2, label='Espira cuadrada'),
    Line2D([0], [0], marker='o', color='w', label='Puntos de la cuadrícula',
           markerfacecolor='green', markersize=5),
    Line2D([0], [0], color='black', lw=0, marker='^', markersize=10, markerfacecolor='black', label='Campo Eléctrico'),
    Line2D([0], [0], color='blue', lw=0, marker='^', markersize=10, markerfacecolor='blue', label='Campo Magnético')
]
ax.legend(handles=legend_elements)

ax.set_title('Espira cuadrada en YZ y plano XZ con cuadrícula')

plt.show()
