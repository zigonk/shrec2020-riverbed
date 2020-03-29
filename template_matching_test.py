import cv2
import numpy as np
import scipy.stats as st

def gkern(kernlen=101, nsig=2):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

img_path = 'Data/PointsPoisson/Train/Class5_1_origin.png'

img = cv2.imread(img_path, 0)
template = gkern()
template *= 255/template.max()

template = template.astype('uint8')
# print(template)

# template = img[100:200, 100:200]
w, h = template.shape[::-1]

res = cv2.matchTemplate(img, template, cv2.TM_CCORR)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
res -= res.min()
print(res.max())
res *= 255/res.max()

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(res,top_left, bottom_right, 255, 2)

cv2.imwrite('./Data/template_matching.png', res)

