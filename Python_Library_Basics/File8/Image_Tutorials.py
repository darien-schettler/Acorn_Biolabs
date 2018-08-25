from PIL import Image

img1 = Image.open('img_1.jpg')
img2 = Image.open('img_2.png')
img3 = Image.open('img_3.jpg')
img4 = Image.open('img_4.jpg')

# print(img1.size)
# print(img1.format)

''' --------------- Cropping Images --------------- '''

area = (270, 250, 920, 500)
cropped_img = img4.crop(area)

''' Combining Images '''

img1.paste(cropped_img, area)
img3 = img3.crop((0, 0, 1365, 1365))
print(img3.size)
# img1.show()

''' --------------- How to break Images Up Into Channels --------------- '''

print(img3.mode) # RGBA - Red Green Blue Alpha - All possibilities (Alpha is opaque/transparency)

r, g, b = img3.split()
r1, g1, b1 = img1.split()

# Showing Individual Channels
# r.show()
# g.show()
# b.show()

# new_img2 = Image.merge("RGB", (b, r, g))
# new_img2 = Image.merge("RGB", (r, b, g))
# new_img2 = Image.merge("RGB", (g, b, r))
# new_img2 = Image.merge("RGB", (r, r, g))
# ETC.
# new_img2.show()

new_img = Image.merge("RGB", (r, b1, g))

new_img.show()