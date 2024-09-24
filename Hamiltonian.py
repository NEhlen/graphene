import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pywavefront import Wavefront

tCC = 2.7
aCC = 1.42

delta1 = 0.5*aCC*np.array([1, np.sqrt(3)])
delta2 = 0.5*aCC*np.array([1, -np.sqrt(3)])
delta3 = -aCC * np.array([1, 0])
delta123 = np.vstack([delta1, delta2, delta3])

def Hamiltonian(k: np.array):
    # make k at least 2d
    k = np.atleast_2d(k)
    # calculate off-diagonal
    Dk = np.sum(np.exp(1j*delta123@k.T), axis=0)

    # generate Hamiltonian, dim 0 is for multiple
    # k-vectors, the Hamiltonian for k-vec i
    # is given by the slice H[i, :, :]
    H = np.zeros((k.shape[0], 2, 2), dtype=complex)

    H[:, 0, 1] = Dk
    H[:, 1, 0] = Dk.conj()

    return tCC*H

def Hamiltonian_meshgrid(kx, ky):
    # make k at least 2d
    k = np.vstack((kx.flatten(), ky.flatten())).T  # Shape (N, 2)
    # calculate off-diagonal
    Dk = np.sum(np.exp(1j*delta123@k.T), axis=0)

    # generate Hamiltonian, dim 0 is for multiple
    # k-vectors, the Hamiltonian for k-vec i
    # is given by the slice H[i, :, :]
    H = np.zeros((k.shape[0], 2, 2), dtype=complex)

    H[:, 0, 1] = Dk
    H[:, 1, 0] = Dk.conj()

    return np.linalg.eigvalsh(tCC*H)



kstart = np.array([0, 0])
kstop = np.array([0, 2.5])
num_points = 1000
kx = np.linspace(kstart[0], kstop[0], num_points)
ky = np.linspace(kstart[1], kstop[1], num_points)
kvecs = np.vstack([kx, ky]).T

H = Hamiltonian(kvecs)
energies = np.linalg.eigvalsh(H)

plt.plot(np.linalg.norm(kvecs, axis=1), energies)

# Define the range of values for x and y
edge=3.2
x = np.linspace(-edge, edge, 500)
y = np.linspace(-edge, edge, 500)

# Create the coordinate matrices
X, Y = np.meshgrid(x, y)

Z = Hamiltonian_meshgrid(X, Y)

Z1 = Z[:, 0].reshape(X.shape)
Z2 = Z[:, 1].reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the first set of eigenvalues
surf1 = ax.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.7, rstride=20, cstride=20)

# Plot the second set of eigenvalues
surf2 = ax.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.7, rstride=20, cstride=20)

# Add a color bar for reference
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)
fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=5)

# Add labels
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('Eigenvalues')

plt.title('3D Surface Plot of Graphene Eigenvalues')
plt.show(block=True)


# Prepare vertices and faces for OBJ export
vertices = []
faces = []

for i in range(X.shape[0] - 1):
    for j in range(Y.shape[1] - 1):
        # Vertices of the first triangle
        v1 = (X[i, j], Y[i, j], Z1[i, j])
        v2 = (X[i + 1, j], Y[i + 1, j], Z1[i + 1, j])
        v3 = (X[i, j + 1], Y[i, j + 1], Z1[i, j + 1])
        
        # Vertices of the second triangle
        v4 = (X[i + 1, j + 1], Y[i + 1, j + 1], Z1[i + 1, j + 1])
        
        # Add vertices
        vertices.append(v1)
        vertices.append(v2)
        vertices.append(v3)
        vertices.append(v4)
        
        # Create faces
        faces.append((len(vertices) - 4, len(vertices) - 3, len(vertices) - 2))
        faces.append((len(vertices) - 3, len(vertices) - 1, len(vertices) - 2))

# Save to OBJ file
obj_file = 'surface.obj'
with open(obj_file, 'w') as f:
    for v in vertices:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in faces:
        f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

print(f"OBJ file saved as {obj_file}")