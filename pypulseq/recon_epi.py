import matplotlib.pyplot as plt
import numpy as np
from rawdatarinator import twixread

data = twixread('./Data/sms_epi_3.dat', A=True)
im_data = np.zeros(data.shape)

for ch in range(data.shape[3]):  #
    temp = np.squeeze(data[:, :, 0, ch])
    temp2 = temp
    temp2[:, ::2] = np.flipud(np.squeeze(temp[:, ::2]))
    im_temp = np.fft.fftshift(np.fft.fft2(temp2))
    im_data[:, :, 0, ch] = np.abs(im_temp)

im_data_sos = np.sum(((np.squeeze(im_data))), axis=-1)
print(im_data_sos.shape)
im_data_sos = np.rot90(im_data_sos, k=1)
plt.imshow(im_data_sos, cmap='gray')
plt.show()
