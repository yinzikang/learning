import cv2
import matplotlib.pyplot as plt

img = plt.imread('lenna.jpeg')
res = cv2.resize(img,(32,32))
plt.ion()
plt.imshow(res)
# plt.show(img)



plt.pause(3)  #显示秒数
plt.close()