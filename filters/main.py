import cv2

image = cv2.imread('cat.png')

def convert_to_hsb(img):
    hsb_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsb_image

def convert_to_rgb(img):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_image

def convert_to_greyscale(img):
    greyscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return greyscale_image

def convert_to_binary(img, value):
    _, binary_image = cv2.threshold(img, value, 255, cv2.THRESH_BINARY)
    return binary_image

def resize_image(img, width, height):
    resized_image = cv2.resize(img, (width, height))
    return resized_image

def rotate_image(img):
    rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image

def change_pixel_value(img, x, y, value):
    img[x, y] = value
    return img

hsb_img       = convert_to_hsb(image)
rgb_img       = convert_to_rgb(image)
greyscale_img = convert_to_greyscale(image)
binary_img    = convert_to_binary(greyscale_img, 100)
resized_img   = resize_image(image, 200, 200)
rotated_img   = rotate_image(image)
modified_img  = change_pixel_value(image.copy(), 50, 50, [0, 0, 255])

cv2.imshow('HSB Image'           , hsb_img)
cv2.imshow('RGB Image'           , rgb_img)
cv2.imshow('Greyscale Image'     , greyscale_img)
cv2.imshow('Binary Image'        , binary_img)
cv2.imshow('Resized Image'       , resized_img)
cv2.imshow('Rotated Image'       , rotated_img)
cv2.imshow('Modified Pixel Image', modified_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
