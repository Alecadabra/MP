import cv2

# Every image file path
relPath = 'Practical Exercises/P1/ex01/'
imgPaths = [f'{relPath}prac01ex01img{num:02d}.png' for num in range(1,5)]

for imgPath in imgPaths:
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    #cv2.imshow(imgPath, img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Print out the image file name and its dimensions (width and height)

    width, height, _ = img.shape
    print(f'{imgPath}: Width = {width}, height = {height}')
