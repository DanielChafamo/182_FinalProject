import numpy as np
from PIL import Image


class Segment(object):

    def __init__(self, file, num_features= 28*28):
        self.num_features = num_features
        self.file = file
        self.chars = []
        self.image = Image.open(file)
        self.pixels = np.array(self.image, dtype='int64')

    # convert to black and white in grayscale
    def cleanup(self):
        width, height = self.image.size
        # this could be cleaned up with a function probs
        for row in self.pixels:
            for col in row:
                gray = (col[1] + col[2] + col[3]) / 3.
                # need to fine-tune this
                if gray > 200:
                    gray = 255
                else:
                    gray = 0
                col[0] = gray
        self.pixels = self.pixels[:, :, :1].reshape(height, width)

    def extract_char(self):
        swappedPixels = np.swapaxes(self.pixels, 0, 1)
        columns = []
        for col in swappedPixels:
            colored = 0
            for row in col:
                if row == 0:
                    colored += 1
            if colored == 0:
                colored = 'w'
            columns.append(colored)
        segments = []
        seg = []
        for i, c in enumerate(columns):
            if c != 'w':
                seg.append(c)
            elif seg != []:
                segments.append((i - len(seg), seg))
                seg = []
        for index, columns in segments:
            character = (swappedPixels[index:(index + len(columns))])
            character = np.swapaxes(character, 0, 1)
            self.chars.append(character)

    def resize(self):
        # print len(self.chars)
        for i, image in enumerate(self.chars):
            image = np.array(image, dtype='int8')
            image = Image.fromarray(image, 'L')
            h, w = image.size
            ratio = min(h / float(w), w / float(h))
            if h > w:
                image = image.resize((28, int(28 * ratio)))
            else:
                image = image.resize((int(28 * ratio), 28))
            pixels = np.array(image, dtype='int64')
            h, w = pixels.shape
            if h > w:
                dif = 28 - w
                add = np.full((28, dif / 2), 255)
                image = np.hstack((add, image))
                if dif % 2 == 1:
                    add = np.full((28, (dif + 1) / 2), 255)
                image = np.hstack((image, add))
            else:
                dif = 28 - h
                add = np.full((28, dif / 2), 255)
                image = np.vstack((add, image))
                if dif % 2 == 1:
                    add = np.full((28, (dif + 1) / 2), 255)
                image = np.hstack((image, add))
            print image.shape
            self.chars[i] = image

    def segment(self):
        self.cleanup()
        self.extract_char()
        self.resize()
        return self.chars
