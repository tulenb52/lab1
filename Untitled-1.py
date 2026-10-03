# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %% [markdown]
# # Базовая работа с изображением

# %%
image = cv2.imread('vid.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# %%
plt.imshow(image)

# %%
image.shape # h,w,c

# %%
image[250,250] # b,g,r

# %%
# ROI
img_roi = image[100:200, 500:700]

# %%
plt.imshow(img_roi)

# %%
b,g,r = cv2.split(image)

# %%
plt.imshow(b, cmap = 'gray')

# %%
plt.imshow(g, cmap = 'gray')

# %%
# alternative approach
b = image[:,:,0]

# %%
import copy

image2 = copy.deepcopy(image)

# %%
image2[50:100,50:100] = [0,0,0]

# %%
plt.imshow(image2)

# %%
# empty image
image_template = np.zeros(image.shape,np.uint8)

# %%
plt.imshow(image_template)

# %% [markdown]
# # Конвертация цветовых моделей

# %%
image_template[0,0]

# %%
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# %%
image_gray[0,0]

# %%
image_gray.shape

# %%
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

# %%
image_hsv.shape

# %%
image_hsv[0,0]

# %%
image[0,0]

# %%
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# %%
image_lab[0,0]

# %% [markdown]
# # Пороговая фильтрация

# %%
_,thresh1 = cv2.threshold(image_gray,200,255,cv2.THRESH_BINARY)

# %%
plt.imshow(thresh1, cmap='gray')

# %%
thresh1[thresh1==100].sum()

# %% [markdown]
# # Построение гистограммы

# %%
histSize = 256
histRange = (0, 256)
accumulate = False

b_hist = cv2.calcHist([b], [0], None, [histSize], histRange, accumulate=accumulate)

# %%
plt.plot(b_hist)

# %%
b_hist_cum = b_hist.cumsum()

# %%
plt.plot(b_hist_cum)

# %%
b_hist_norm = b_hist /  (image.shape[0] * image.shape[1])

# %%
plt.plot(b_hist_norm)

# %% [markdown]
# # Сравнение двух изображений

# %%
from skimage.metrics import structural_similarity, mean_squared_error

image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
(ssim, diff) = structural_similarity(image_gray, image2_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(ssim))

# %%
plt.imshow(diff)

# %%
mse = mean_squared_error(image_gray, image_gray)
mse

# %% [markdown]
# # Статистические характеристики изображений

# %%
mean = image_gray.mean()

# %%
std = image_gray.std()

# %%
print(mean,std)

# %%
eq_gray = cv2.equalizeHist(image_gray)

# %%
plt.imshow(eq_gray, cmap="gray")


# %%
plt.imshow(image_gray, cmap="gray")

# %%
# Задание 1
sar_1_gray_dz = cv2.imread('vid_gray.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(sar_1_gray_dz, cmap='gray')

# %%
sar_1_gray_dz.shape

# %%
# Задание 2
histSize = 256
histRange = (0, 256)
accumulate = False

gray_hist_dz = cv2.calcHist([sar_1_gray_dz], [0], None, [histSize], histRange, accumulate=accumulate)

# %%
plt.plot(gray_hist_dz)

# %%
gray_hist_dz_cum = gray_hist_dz.cumsum()
plt.plot(gray_hist_dz_cum)

# %%
gray_hist_dz_norm = gray_hist_dz / (sar_1_gray_dz.shape[0] * sar_1_gray_dz.shape[1])
plt.plot(gray_hist_dz_norm)

# %%
# Задание 3
def gamma_correction(image, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)

# %%
# gamma < 1
image_gamma_low_dz = gamma_correction(sar_1_gray_dz, 0.5)
plt.imshow(image_gamma_low_dz, cmap='gray')

# %%
# gamma > 1
image_gamma_high_dz = gamma_correction(sar_1_gray_dz, 2.0)
plt.imshow(image_gamma_high_dz, cmap='gray')

# %%
# Задание 4
from skimage.metrics import structural_similarity, mean_squared_error

# %%
# gamma < 1
(ssim_low_dz, diff_low_dz) = structural_similarity(sar_1_gray_dz, image_gamma_low_dz, full=True, data_range=255)
diff_low_dz = np.clip((1 - diff_low_dz) * 255, 0, 255).astype("uint8")
print("SSIM (gamma<1): {}".format(ssim_low_dz))

# %%
mse_low_dz = mean_squared_error(sar_1_gray_dz, image_gamma_low_dz)
mse_low_dz

# %%
# gamma > 1
(ssim_high_dz, diff_high_dz) = structural_similarity(sar_1_gray_dz, image_gamma_high_dz, full=True, data_range=255)
diff_high_dz = np.clip((1 - diff_high_dz) * 255, 0, 255).astype("uint8")
print("SSIM (gamma>1): {}".format(ssim_high_dz))

# %%
mse_high_dz = mean_squared_error(sar_1_gray_dz, image_gamma_high_dz)
mse_high_dz

# %%
print(f"SSIM (gamma < 1): {ssim_low_dz:.4f}, MSE: {mse_low_dz:.2f}")
print(f"SSIM (gamma > 1): {ssim_high_dz:.4f}, MSE: {mse_high_dz:.2f}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(sar_1_gray_dz, cmap='gray')
axes[0, 0].set_title('Оригинал')

axes[0, 1].imshow(image_gamma_low_dz, cmap='gray')
axes[0, 1].set_title('gamma < 1 (0.5)')

axes[0, 2].imshow(image_gamma_high_dz, cmap='gray')
axes[0, 2].set_title('gamma > 1 (2.0)')

axes[1, 0].imshow(diff_low_dz, cmap='gray')
axes[1, 0].set_title('SSIM diff (gamma < 1)')

axes[1, 1].imshow(diff_high_dz, cmap='gray')
axes[1, 1].set_title('SSIM diff (gamma > 1)')

axes[1, 2].axis('off')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
# Задание 5
eq_gray.mean(), eq_gray.std()

# %%
sar_1_gray_dz.mean(), sar_1_gray_dz.std()

# %%
def match_statistics(src, ref):
    src = src.astype(np.float32)
    ref = ref.astype(np.float32)
    out = (src - src.mean()) * (ref.std() / (src.std() + 1e-8)) + ref.mean()
    return np.clip(out, 0, 255).astype(np.uint8)

# %%
stat_corrected_dz = match_statistics(sar_1_gray_dz, eq_gray)
stat_corrected_dz.mean(), stat_corrected_dz.std()

# %%
plt.imshow(stat_corrected_dz, cmap='gray')

# %%
# Задание 6 
_, thresh_50_dz = cv2.threshold(sar_1_gray_dz, 50, 255, cv2.THRESH_BINARY)
plt.imshow(thresh_50_dz, cmap='gray')

# %%
_, thresh_127_dz = cv2.threshold(sar_1_gray_dz, 127, 255, cv2.THRESH_BINARY)
plt.imshow(thresh_127_dz, cmap='gray')

# %%
_, thresh_tozero_inv_dz = cv2.threshold(sar_1_gray_dz, 127, 255, cv2.THRESH_TOZERO_INV)
plt.imshow(thresh_tozero_inv_dz, cmap='gray')

# %%
# 1. Загрузите изображение в оттенках серого sar_1_gray.jpg. +
# 2. постройте гистограмму +
# 3. реализуйте алгоритм гамма коррекции с параметром гамма <1, >1. +
# 4. Сравните исходное изображение, скорректированное при помощи гамма-фильтра. MSE, SSIM. +
# 5. реализуйте алгоритм статистической цветокоррекции на основе статистики eq_gray. +
# 6. Протестируйте работу алгоритмов пороговой фильтрации с различными параметрами. +
# Для каждого решения - напечатайте результат +