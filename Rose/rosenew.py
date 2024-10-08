import cv2
import numpy as np


image = cv2.imread("rose.jpg")

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

alt = np.array([0, 100, 70])
ust = np.array([10, 255, 255])
alt1 = np.array([170, 100, 70])
ust1 = np.array([180, 255, 255])

maske1 = cv2.inRange(hsv_image, alt, ust)
maske2 = cv2.inRange(hsv_image, alt1, ust1)
kirmizimaske = maske1 + maske2

kernel = np.ones((4, 4), np.uint8)

kirmizimaske = cv2.erode(kirmizimaske, kernel, iterations=1)
kirmizimaske = cv2.dilate(kirmizimaske, kernel, iterations=1)
kirmizimaske = cv2.morphologyEx(kirmizimaske, cv2.MORPH_OPEN, kernel)
kirmizimaske = cv2.GaussianBlur(kirmizimaske, (3, 3), 0)

mor_gul_hsv = hsv_image.copy()

mor_gul_hsv[:,:,0] = 140

mor_gul = cv2.cvtColor(mor_gul_hsv, cv2.COLOR_HSV2BGR)

ters_maske = cv2.bitwise_not(kirmizimaske)
siyah_arkaplan = np.zeros_like(image)


sonuc = cv2.bitwise_and(mor_gul, mor_gul, mask=kirmizimaske)
sonuc += cv2.bitwise_and(siyah_arkaplan, siyah_arkaplan, mask=ters_maske)


cv2.imshow("Orijinal Resim", image)
cv2.imshow("Mor Gül ve Siyah Arka Plan", sonuc)
cv2.imwrite('New Image.jpeg',sonuc)

cv2.waitKey(0)
cv2.destroyAllWindows()
