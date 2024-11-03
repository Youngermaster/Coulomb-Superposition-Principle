import numpy as np


def calculateElectricCamp(q, x, x1, y, y1, z, z1):
    k = 9E9  # Constante de Coulomb en N m²/C²
    r = np.linalg.norm(np.array([x - x1, y - y1, z - z1]))
    if r == 0:
        raise ValueError("Los puntos de observación y de la carga son iguales; no se puede calcular el campo eléctrico.")
    unitaryVector = calculateUnitaryVector(x, x1, y, y1, z, z1)
    electricCamp = ((k * q) / r**2) * unitaryVector
    return electricCamp
        
def calculateUnitaryVector(x, x1, y, y1, z, z1):
    # Calcula el vector de componentes
    r = np.array([x - x1, y - y1, z - z1])
    # Calcula la magnitud del vector
    rMagnitude = np.linalg.norm(r)
    # Calcula el vector unitario
    unitaryVector = r / rMagnitude
    return unitaryVector

if __name__=="__main__":
    xCoordinate = []
    yCoordinate = []
    zCoordinate = []
    qCharge = len(xCoordinate)/12E-9
    pPoint = [1,2,3]
    
    totalElectricCamp = np.array([0.0, 0.0, 0.0])
    
    for i in range(len(xCoordinate)):
        x1=xCoordinate[i]
        y1=yCoordinate[i]
        z1=zCoordinate[i]
        electricCamp = calculateElectricCamp(qCharge,pPoint[0],x1,pPoint[1],y1,pPoint[2],z1)
        totalElectricCamp += electricCamp  # Sumar el campo eléctrico de cada carga
    
    print("Campo eléctrico total en el punto", pPoint, ":", totalElectricCamp)
