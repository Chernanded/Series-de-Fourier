import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

# Definir la función
def f(t, T):
  f = np.zeros_like(t)
  """ Función triangular"""
  f_1 = np.logical_and(-T/2 < t, t < -T/4)# -T/2 <= t < -T/4
  f_2 = np.logical_and(-T/4 <= t, t < T/4)# -T/4 <= t < T/4
  f_3 = np.logical_and(T/4 <= t, t < T/2)# T/4 <= t < T/2
  # Asignar valores a la función
  f[f_1] = -1
  f[f_2] = 1
  f[f_3] = -1
  return f

# Definir los parámetros
T = 2*np.pi # Período
N = 8 # Número de coeficientes de la serie de Fourier

# Calcular los coeficientes de la serie de Fourier
# Puntos para calcular la integral (los mismos que se usarán para graficar) 
t = np.linspace(-T/2, T/2, 10000)# 1000 puntos entre -T/2 y T/2
# Frecuencias para cada coeficiente (de -N a N con paso de 2*pi/T)  
w = (2*np.pi/T )* np.arange(-N, N+1)

# Vector vacío para guardar los coeficientes 
c = np.zeros_like(w, dtype=np.complex_) # dtype=np.complex_ para que sea un número complejo
# Calcular los coeficientes con la integral   
for i, n in enumerate(range(-N, N+1)):
  c[i] = 2/T * np.trapz(f(t, T) * np.exp(-1j*w[i]*t), t) # Integral de f(t) * exp(-j*w*t) dt


# Calcular la serie de Fourier
f_series = np.zeros_like(t, dtype=np.complex_)
for n in range(-N, N+1):
  f_series += c[N+n]*np.exp(2j*np.pi*n/T*t) # Suma de los coeficientes * exp(j*2*pi*n*t/T)




# Gráfica  estática
fig1, axs1 = plt.subplots(2, 1, figsize=(6, 16),tight_layout=True)
#graficamos con limite de x de -3 a 3 y de y de -1.5 a 1.5
axs1[0].set_xlim(-3.1, 3.1)
axs1[0].set_ylim(-1.5, 1.5)
axs1[0].plot(t, f(t, T), linewidth=2)
axs1[0].set_title('Función original')
axs1[0].set_xlabel('Tiempo')
axs1[0].set_ylabel('Amplitud')
axs1[1].stem(w, c, use_line_collection=True)
axs1[1].set_title('Serie de Fourier')
axs1[1].set_xlabel('Frecuencia')
axs1[1].set_ylabel('Amplitud')
# plt.show()



# Crear objeto de la animación
fig2, axs2 = plt.subplots(2, 1, figsize=(6, 16),tight_layout=True)
camera = Camera(fig2)

# Animar la gráfica
for n in range(-N, N+1):
  f_series = np.zeros_like(t, dtype=np.complex_)
  for i in range(N-n, N+n+1):
    f_series += c[i]*np.exp(2j*np.pi*(i-N)*t/T)
  axs2[0].plot(t, f_series.real, linewidth=2)
  axs2[0].set_title('Serie de Fourier - Tiempo: {:.2f}'.format(n*0.1))
  axs2[0].set_xlabel('Tiempo')
  axs2[0].set_ylabel('Amplitud')
  axs2[1].stem(w, np.abs(c), use_line_collection=True)
  axs2[1].set_title('Espectro de frecuencias')
  axs2[1].set_xlabel('Frecuencia')
  axs2[1].set_ylabel('Amplitud')
  camera.snap()

# Guardar y mostrar la animación
animation = camera.animate(interval=50)
animation.save('fourier.gif', writer='imagemagick')
plt.show()

