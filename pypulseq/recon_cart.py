import matplotlib.pyplot as plt
import numpy as np
from rawdatarinator import twixread

data = twixread('./Data/110121/slr_90.dat', A=True)
im_data = np.zeros(data.shape)

for ch in range(data.shape[3]):  #
    temp = np.squeeze(data[:, :, 0, ch])
    temp2 = temp
    im_temp = np.fft.fftshift(np.fft.fft2(temp2))
    im_data[:, :, 0, ch] = np.abs(im_temp)

im_data_sos = np.sum(((np.squeeze(im_data))), axis=-1)
print(im_data_sos.shape)
im_data_sos = np.rot90(im_data_sos, k=1)
plt.imshow(im_data_sos, cmap='gray')
plt.show()
