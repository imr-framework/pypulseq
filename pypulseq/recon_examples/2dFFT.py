import numpy as np
from matplotlib import pyplot as plt

from dat2py import dat2py_main

path = r"C:\Users\sravan953\Downloads\FINAL_meas_MID00169_FID00800_pulseq_3D_mprage.dat"
kspace, img = dat2py_main.main(dat_file_path=path)
img = np.abs(np.sqrt(np.sum(np.square(img), -1)))

plt.imshow(img)
plt.show()


def main():
    path = r"C:\Users\sravan953\Desktop\20210424_7datas\gre_meas_MID00176_FID00172_pulseq.dat"
    # kspace, img = dat2py_main.main(dat_file_path=path)


if __name__ == '__main__':
    main()
