# Proyecto final de Robótica Inteligente
Por Irene Alavés, Alejandro Conesa, Claudia Fernández e Irene Robles.

Nuestro trabajo final es un estudio sobre los cambios en velocidad, consumo de recursos y precisión al sustituir por completo la planificación y guía de Nav2 en un robot por una red neuronal. Las trayectorias se graban en un entorno pequeño con obstáculos que el robot puede rodear sin complicaciones, ya que al no tener memoria local un trayecto más complejo como un laberinto es una tarea difícil.

Usamos Torch, SciPy y NumPy como librerías externas de Python para facilitar scripts de generación de mapas, la propia red neuronal y la automatización del dataset. La simulación se hace en un entorno de Ros2, usando Nav2, Gazebo y SLAM.