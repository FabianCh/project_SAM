from skimage.feature import hog
import skimage.io as io
io.use_plugin('pil') # Use all capabilities provided by PIL

image = io.imread("data/msdi_img/msdi_img/img/0000000020_200.jpg")

fd, hog_image = hog(image, pixels_per_cell=(3, 3),
                    cells_per_block=(3, 3), visualize=True, multichannel=True)


import matplotlib.pyplot as plt
from skimage import exposure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()




