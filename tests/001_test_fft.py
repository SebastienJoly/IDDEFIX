import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import iddefix
from scipy.constants import c as c_light

# Importing wake potential data
data_wake_potential = np.loadtxt('../examples/data/004_SPS_model_transitions_q26.txt', comments='#', delimiter='\t')

# Extracting data
data_wake_time = data_wake_potential[:,0]*1e-9 # [s]
data_wake_dipolar = data_wake_potential[:,2]
sigma = 1e-10

# Decimate wake data
x = np.linspace(data_wake_time[0],data_wake_time[-1],1000)
y = np.interp(x, data_wake_time, data_wake_dipolar)

# Read old results for DE model
DE_model = iddefix.EvolutionaryAlgorithm(data_wake_time, 
                                         data_wake_dipolar*c_light, # remove normalization 
                                         N_resonators=10, 
                                         parameterBounds=None,
                                         plane='transverse',
                                         fitFunction='wake potential',
                                         sigma=sigma)
from io import StringIO
data_str = StringIO('''
    1     |        2.22e+00        |      76.87       |    1.005e+09     
    2     |        7.62e+00        |      138.95      |    1.176e+09     
    3     |        1.15e+00        |      15.49       |    1.268e+09     
    4     |        1.19e+00        |      39.99       |    1.657e+09     
    5     |        1.54e+00        |      169.72      |    2.075e+09     
    6     |        1.79e+00        |      177.73      |    2.199e+09     
    7     |        1.67e+00        |      53.54       |    2.251e+09     
    8     |        1.87e+00        |      39.01       |    2.431e+09     
    9     |        1.84e+00        |       5.01       |    2.675e+09     
    10    |        1.99e+00        |      178.88      |    2.908e+09     
    11    |        1.99e+00        |      38.55       |    3.184e+09     
''')
DE_model.minimizationParameters = np.loadtxt(
        data_str, skiprows=3, usecols=(1, 2, 3),
        delimiter='|', dtype=float,).flatten()

# Getting the wake function with neffint
import neffint

time = np.linspace(1e-11, 50e-9, 1000)
f_fd = np.linspace(0, 5e9, 10000)
Z_fd = DE_model.get_impedance(frequency_data=f_fd, wakelength=None) 

W = neffint.fourier_integral_fixed_sampling(
    times=time,
    frequencies=f_fd,
    func_values=Z_fd,
    pos_inf_correction_term=True,
    neg_inf_correction_term=False,
    interpolation="linear" # Feel free to change to "linear"
)

# Compare wake functions
W_de = DE_model.get_wake(time)
fig, ax0 = plt.subplots(1, 1, figsize=(12,5))

ax0.plot(time, W.imag/np.pi, #for transverse is imag!
         lw = 2, c='tab:blue', linestyle='-', label='neffint wake function', alpha=0.8)

ax0.plot(time, W_de,
         lw = 2, c='tab:red', linestyle='-', label='DE fitting wake function', alpha=0.8)

ax0.set_xlabel('f [Hz]')
ax0.set_ylabel('$[ABS](Z_{transverse})$ [$\Omega$]')
ax0.legend(loc='best', fontsize=14)
ax0.grid()
plt.show()
fig.savefig('001_compare_wakes.png')

# Compare impedances

f_de = np.linspace(0, 5e9, 10000)
Z_de = DE_model.get_impedance(frequency_data=f_de, wakelength=None) 

f_fft, Z_fft = iddefix.compute_fft(time, W_de/c_light, fmax=5e9) 
f_nft, Z_nft = iddefix.compute_fft(time, W.imag/c_light/np.pi, fmax=5e9) 

Z_fft *= 1j #transverse
Z_nft *= 1j #transverse

fig = plt.figure(figsize=(8, 5))
plt.plot(f_de, np.real(Z_de), color='tab:red', label='Fully decayed real impedance')
plt.plot(f_de, np.imag(Z_de), color='tab:blue', label='Fully decayed imag. impedance')
plt.plot(f_de, np.abs(Z_de), color='tab:green', label='Fully decayed Abs. impedance')

plt.plot(f_fft, np.real(Z_fft), color='tab:red', ls='--', label='fft real impedance')
plt.plot(f_fft, np.imag(Z_fft), color='tab:blue', ls='--', label='fft imag. impedance')
plt.plot(f_fft, np.abs(Z_fft), color='tab:green', ls='--', label='fft Abs. impedance')

plt.plot(f_nft, np.real(Z_nft), color='tab:red', ls=':', label='nft real impedance')
plt.plot(f_nft, np.imag(Z_nft), color='tab:blue', ls=':', label='nft imag. impedance')
plt.plot(f_nft, np.abs(Z_nft), color='tab:green', ls=':', label='nft Abs. impedance')

plt.legend()
plt.xlabel('f [Hz]')
plt.ylabel('$Z_{Transverse}$ [$\Omega$]')
plt.show()
fig.savefig('001_compare_imp.png')

