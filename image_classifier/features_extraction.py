import os
import pickle
import skimage.io as io
from skimage.feature import hog


data_label = pickle.load(open("data/msdi_img/msdi_img_labels.pickle", "rb"))
print(data_label.values())


def load_image(filename):
    image = io.imread("data/msdi_img/msdi_img/img/" + filename)
    return image


def image_feature_extraction(image):
    fd = hog(image, pixels_per_cell=(3, 3), cells_per_block=(3, 3), visualize=False, multichannel=True)
    return fd


def save_feature_extraction(fd, filename):
    pickle.dump(fd, open("data/msdi_img/msdi_fd/" + filename, "wb"))


def extract_feature_all_image():
    file_list = os.listdir("data/msdi_img/msdi_img/img/")
    for i in file_list:
        img = load_image(i)
        fd_img = image_feature_extraction(img)
        save_feature_extraction(fd_img, i[:-4])


extract_feature_all_image()

# # Exemple of feature extraction on 1 image
#
# file = "0000000020_200.jpg"
# img = load_image(file)
# fd_img = image_feature_extraction(img)
# save_feature_extraction(fd_img, file[:-4])


