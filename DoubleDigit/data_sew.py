from matplotlib import pyplot as plt
import numpy as np
import random


class ImageStitcher(object):
    """Stitches together two single digit images into a double digit image with
       possible overlap"""

    def __init__(self, img_width, images, labels, overlap_range=(-25, 0),
                 repeated_digits=True, tuple_ = False):
        if img_width >= images[0].shape[0] * 2 + overlap_range[1]:
            self.img_width = img_width
        else:
            self.img_width = images[0].shape[0] * 2 + overlap_range[1]
        self.overlap_range = overlap_range
        self.original_imgs = images
        self.original_labels = labels
        self.resized_imgs = [np.concatenate((np.zeros((28, 14)), img,np.zeros((28, 14))), axis=1) for img in self.original_imgs]
        self.stitched_imgs = []
        self.stitched_labels = []
        if repeated_digits:
            self.repeated_digits = True
        else:
            self.repeated_digits = False
        self.tuple_ = tuple_

    def view_image(self, image):
        plt.matshow(image, aspect='auto', cmap='gray')
        plt.show()

    # overlap_range should be a tuple of values i.e. (-25, 0)
    def set_overlap(self, overlap_range):
        self.overlap_range = overlap_range

    def get_overlap(self):
        return self.overlap_range

    def resize_all(self):
        pass

    def stitch(self,image1,image2,num_pixels):
        if num_pixels==0:
            new_image=np.concatenate((image1,image2),axis=1)
        else:
            overlap=image1[:,num_pixels:]+image2[:,:num_pixels*-1]
            #ensuring no values over 255
            overlap[overlap>255]=255
            new_image=np.concatenate((image1[:,:image1.shape[1]+num_pixels],
                                        overlap,
                                        image2[:,-1*num_pixels:]),axis=1)
        #resizes image
        new_image=np.concatenate(((np.zeros((28,((self.img_width-new_image.shape[1])//2)))),
                                new_image,
                                np.zeros((28,((self.img_width-new_image.shape[1])//2)))),
                                axis=1)
        if new_image.shape[1]==self.img_width-1:
            new_image=np.concatenate((np.zeros((28,1)),new_image),axis=1)
        return new_image

    def overlap_images(self, num_imgs, overlap_range=None):
        if overlap_range is None:
            overlap_range = self.overlap_range
        self.stitched_imgs = np.zeros((num_imgs,
                                      self.original_imgs[0].shape[0],
                                      self.img_width))
        self.stitched_labels = np.zeros((num_imgs), dtype='int64')
        sample_idxs = len(self.original_imgs) - 1
        for img in range(num_imgs):
            img1_idx = random.randint(0, sample_idxs)
            # to ensure a non-zero first digit is chosen
            while 0 == self.original_labels[img1_idx]:
                img1_idx = random.randint(0, sample_idxs)
            img2_idx = random.randint(0, sample_idxs)
            if not self.repeated_digits:
                while self.original_labels[img1_idx] == self.original_labels[img2_idx]:
                    img2_idx = random.randint(0, sample_idxs)
            img1 = self.original_imgs[img1_idx]
            img2 = self.original_imgs[img2_idx]
            num_pixels = random.randint(overlap_range[0], overlap_range[1])
            new_image = self.stitch(img1, img2, num_pixels)
            new_image = new_image.astype('float32') / 255
            self.stitched_imgs[img] = new_image
            # print(self.original_labels[img1_idx], self.original_labels[img2_idx])
            self.stitched_labels[img] = int(str(self.original_labels[img1_idx]) + str(self.original_labels[img2_idx]))
        if self.tuple_ is True:
            new_labels = []
            for i in self.stitched_labels:
                new_labels.append([int(x) for x in str(i)])
            self.stitched_labels = new_labels



    def __repr__(self):
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("                   R E P R                   ")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("self.img_width =", self.img_width)
        print("self.overlap_range =", self.overlap_range)
        print("self.original_imgs =", self.original_imgs)
        print("self.original_labels =", self.original_labels)
        print("self.stitched_imgs =", self.stitched_imgs)
        print("self.stitched_labels =", self.stitched_labels)
        print("+++++++++++++++++++++++++++++++++++++++++++++")


# For basic testing
def main():
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_stiches = ImageStitcher(40, train_images, train_labels,
                                  overlap_range=(-17, 0), repeated_digits=False, tuple_ = True)
    train_stiches.overlap_images(4000, overlap_range=(-25, 0))
    for i in train_stiches.stitched_labels:
        s_label = str(i)
        assert s_label[0] != s_label[1]
    print("++++++++++++++++++++++++++++++++")
    print("+ repeated_digits test: PASSED +")
    print("++++++++++++++++++++++++++++++++")
    for i in range(3):
        print(train_stiches.stitched_labels[i])
        train_stiches.view_image(train_stiches.stitched_imgs[i])
    # train_stiches.__repr__()
    train_stiches.resize_all()
    for i in range(3):
        print(train_stiches.original_labels[i])
        train_stiches.view_image(train_stiches.resized_imgs[i])


if __name__ == '__main__':
    from keras.datasets import mnist
    main()
