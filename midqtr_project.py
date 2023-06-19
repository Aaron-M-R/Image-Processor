"""
DSC 20 Mid-Quarter Project
Name(s): Aaron Rasin and Aryan Shah
PID(s):  A16269679 and A17639598
"""

import numpy as np
from PIL import Image

NUM_CHANNELS = 3


# Part 1: RGB Image #
class RGBImage:

    def __init__(self, pixels):
        """

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        if not isinstance(pixels, list) or len(pixels)<=0:
            raise TypeError
        for row in pixels:
            if len(row)<=0 or len(row)!=len(pixels[0])\
             or not isinstance(row,list):
                raise TypeError
        for row in pixels:
            for pixel in row:
                if len(pixel)!=3 or not isinstance(pixel, list):
                    raise TypeError
            
        for row in pixels:
            for pixel in row:
                for intensity in pixel:
                    if not 0<=intensity<=255 or type(intensity)!=int:
                        raise ValueError

        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[color for color in pixel] \
        for pixel in row] for row in self.pixels]

    def copy(self):
        """

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError
        if not 0<=row<self.num_rows or not 0<=col<self.num_cols:
            raise ValueError
        else:
            return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the resulting pixel list
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if not isinstance(row,int) or not isinstance(col,int):
            raise TypeError()
        if not 0 <= row <= self.num_rows - 1:
            raise ValueError()
        if not 0 <= col <= self.num_cols - 1:
            raise ValueError()
        if not isinstance(new_color,tuple) or not len(new_color) == 3:
            raise TypeError()
        if not (isinstance(i,int) for i in new_color):
            raise TypeError()
        for i in new_color:
            if not i <= 255:
                raise ValueError()

        for i in range(len(new_color)):
            if new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i]
            else:
                self.pixels[row][col][i] = self.pixels[row][col][i]


# Part 2: Image Processing Template Methods #

class ImageProcessingTemplate:

    def __init__(self):
        """

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img_input)                         # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """
        pixels = image.get_pixels()
        return RGBImage([[[255 - i for i in cood]for cood in row]for row in pixels])

    def grayscale(self, image):
        """

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        pixels = image.get_pixels()
        return RGBImage([[[sum(cood)//3 for i in cood]for cood in row]for row in pixels])

    def rotate_180(self, image):
        """

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        pixels = image.get_pixels()
        return RGBImage([row[::-1] for row in pixels][::-1])



# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):

    def __init__(self):
        """
        initializes the child class of ImageProcessingTemplate and 
        sets cost to be paid, coupons to be redeemed and rotations made to 0

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0
        self.coupons = 0
        self.rotations = 0

    def negate(self, image):
        """
        gives the negative of an image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.coupons == 0:
            self.cost+=5
        else:
            self.coupons-=1
        return ImageProcessingTemplate.negate(self, image)

    def grayscale(self, image):
        """
        makes an image black and white

        """
        if self.coupons == 0:
            self.cost+=6
        else:
            self.coupons-=1
        return ImageProcessingTemplate.grayscale(self, image)

    def rotate_180(self, image):
        """
        rotates an image by 180 degrees

        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if self.coupons >0:
            self.coupons-=1
        elif self.rotations%2==0:
            self.cost+=10
        else:
            self.cost-=10
        self.rotations+=1
        return ImageProcessingTemplate.rotate_180(self, image)

    def redeem_coupon(self, amount):
        """
        method for using a coupon

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        0
        """
        if amount<=0:
            raise ValueError
        if not isinstance(amount, int):
            raise TypeError
        else:
            self.coupons+=amount

        


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):

    def __init__(self):
        """
        initializes class with instances from ImageProcessingTemplate
        and sets cost to 50

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        super().__init__()
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Replaces all pixels of a certain color with corresponding pixels
        from a background image

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
        """
        if not isinstance(chroma_image,RGBImage) or not \
           isinstance(background_image, RGBImage):
            raise TypeError

        if len(chroma_image.get_pixels()) != \
           len(background_image.get_pixels()) or \
           len(chroma_image.get_pixels()[0]) != \
           len(background_image.get_pixels()[0]):
            raise ValueError()

        chroma_image_1 = chroma_image.get_pixels()
        background = background_image.get_pixels()

        for row in range(len(chroma_image_1)):
            for col in range(len(chroma_image_1[row])):
                if list(color) == chroma_image_1[row][col]:
                    chroma_image_1[row][col] = background[row][col]

        return RGBImage(chroma_image_1)


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Puts sticker on an image at a given point

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        if not isinstance(sticker_image, RGBImage) \
        or not isinstance(background_image, RGBImage):
            raise TypeError()

        if sticker_image.num_rows >= background_image.num_rows \
        or sticker_image.num_cols >= background_image.num_cols:
            raise ValueError()

        if not isinstance(x_pos,int) or not isinstance(y_pos,int):
            raise TypeError()

        sticker = sticker_image.get_pixels()
        background = background_image.get_pixels()

        if len(sticker) + x_pos > len(background) or \
           len(sticker[0]) + y_pos > len(background[0]):
            raise ValueError()

        for row in range(len(sticker)):
            for col in range(len(sticker[row])):
                background[row + x_pos][col + y_pos] = sticker[row][col]

        return RGBImage(background)


# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
        # make random training data (type: List[Tuple[RGBImage, str]])
        >>> train = []

        # create training images with low intensity values
        >>> train.extend(
        ...     (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
        ...     for _ in range(20)
        ... )

        # create training images with high intensity values
        >>> train.extend(
        ...     (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
        ...     for _ in range(20)
        ... )

        # initialize and fit the classifier
        >>> knn = ImageKNNClassifier(5)
        >>> knn.fit(train)

        # should be "low"
        >>> print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
        low

        # can be either "low" or "high" randomly
        >>> print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
        This will randomly be either low or high

        # should be "high"
        >>> print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))
        high
    """

    def __init__(self, n_neighbors):
        """
        initializes class with number of neighbors and absence of data
        """
        self.n_neighbors = n_neighbors
        self.data = None

    def fit(self, data):
        """
        stores given data in self
        """
        if len(data) <= self.n_neighbors:
            raise ValueError
        if self.data is not None:
            raise ValueError
        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        Checks that both images are RGBImage instances of the same size
        """
        if not isinstance(image1, RGBImage) \
        or not isinstance(image2, RGBImage):
            raise ValueError

        if image1.num_rows != image2.num_rows \
        or image1.num_cols != image2.num_cols:
            raise ValueError

        square = 2
        return [[sum([(a-b)**2 for a,b in zip(color1,color2) ]) \
        for color1,color2 in zip(row1,row2) ]\
        for row1,row2 in zip(image1.pixels,image2.pixels) ][0][0]**.5

    @staticmethod
    def vote(candidates):
        """
        method for finding the most frequent item in a list
        """
        return max(set(candidates), key=candidates.count)
        
    def predict(self, image):
        """
        Predicts label of an image based on similar images 
        from data based on color
        """
        if self.data is None:
            raise ValueError

        # empty list to store distances
        dist_list = []

        #finds distances between each image in self.data and the given image#
        # tuple has distance first, label second #
        for img in self.data:
            dist_list.append((ImageKNNClassifier.distance(image, img[0]), img[1]))

        # sorts list of tuples of distances and labels #
        dist_list = sorted(dist_list)

        # makes cutoff at index n_neighbors #
        if len(dist_list)>self.n_neighbors:
            dist_list = dist_list[:self.n_neighbors]

        # returns most popular image label from shortened dictionary # 
        return ImageKNNClassifier.vote(dist_list)[1]

# --------------------------------------------------------------------------- #
def create_random_pixels(low, high, nrows, ncols):
    """
    Create a random pixels matrix with dimensions of
    3 (channels) x `nrows` x `ncols`, and fill in integer
    values between `low` and `high` (both exclusive).
    """
    return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)
