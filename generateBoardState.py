import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from numpy import asarray
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from shapely.geometry import Polygon


IMAGE_DIR = 'C:/images'
OUTPUT_DIR = 'C:/results'
IMAGE = 'image.jpg'

TL = "A8"


def setOrientation():
    global TL
    valid_values = ["A8", "A1", "H1", "H8"]
    while True:
        value = input(f"Enter the actual top-left corner of your image ({', '.join(valid_values)}): ").strip()
        if value in valid_values:
            TL = value
            break
        else:
            print("Invalid value for corner. Must be one of ['A8', 'A1', 'H1', 'H8'].")


def orderPoints(pts):    
    global TL

    sorted_by_y = pts[np.argsort(pts[:, 1])]

    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]

    top_left, top_right = top_points[np.argsort(top_points[:, 0])]

    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

    sorted_corners = np.array([top_left, top_right, bottom_right, bottom_left])

    if (TL == "A8"):
        rect = sorted_corners

    elif (TL == "H8"):
        rect = np.array([bottom_left, top_left, top_right, bottom_right])

    elif (TL == "H1"):
        rect = np.array([bottom_right, bottom_left, top_left, top_right])

    elif (TL == "A1"):
        rect = np.array([top_right, bottom_right, bottom_left, top_left])

    return rect


def detectCorners(image):
    model_corners = YOLO("cornersDetection.pt")
    results = model_corners.predict(source=image, line_width=1, conf=0.5, save_txt=True, save=True)

    boxes = results[0].boxes
    arr = boxes.xywh.numpy()
    points = arr[:, 0:2]

    corners = orderPoints(points)
    return corners


def cutChessboard(image_path, corners):
    img = Image.open(image_path)
    image = asarray(img)
    (tl, tr, br, bl) = corners

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    img = Image.fromarray(warped, "RGB")
    return img


def transformImage(image_path):
    if image_path.endswith('.jpg') or image_path.endswith('.png'):        

        setOrientation()
        corners = detectCorners(image_path)
        transformed_image = cutChessboard(image_path, corners)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        new_file_name = os.path.join(OUTPUT_DIR, base_name + "_transformed.jpg")
        
        transformed_image.save(new_file_name)
        print(f"Processed and saved: {image_path}")
        return transformed_image
    else:
        print("Incorrect image file!")


def detectChessPieces(image):
    
    model_trained = YOLO("piecesRecognition.pt")
    results = model_trained.predict(source=image, line_width=1, conf=0.5, augment=False, save_txt=True, save=True)
    
    boxes = results[0].boxes
    detections = boxes.xyxy.numpy()
    
    return detections, boxes


def makeGrid(image):
    
    corners = np.array([[0,0], 
        [image.size[0], 0], 
        [0, image.size[1]], 
        [image.size[0], image.size[1]]])

    figure(figsize=(10, 10), dpi=80)

    implot = plt.imshow(image)

    TL = corners[0]
    BL = corners[2]
    TR = corners[1]
    BR = corners[3]

    def interpolate(xy0, xy1):
        x0,y0 = xy0
        x1,y1 = xy1
        dx = (x1-x0) / 8
        dy = (y1-y0) / 8
        pts = [(x0+i*dx,y0+i*dy) for i in range(9)]
        return pts

    ptsT = interpolate( TL, TR )
    ptsL = interpolate( TL, BL )
    ptsR = interpolate( TR, BR )
    ptsB = interpolate( BL, BR )
        
    for a,b in zip(ptsL, ptsR):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
    for a,b in zip(ptsT, ptsB):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
        
    plt.axis('off')

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    new_file_name = os.path.join(OUTPUT_DIR, base_name + "_grid.jpg")

    plt.savefig(new_file_name)
    return ptsT, ptsL


def calculateIoU(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    iou = poly1.intersection(poly2).area / poly1.union(poly2).area
    return iou


def connectDetectionToSquare(detections, square):
    
    pieceLetter = {0: 'P', 1: 'B', 2: 'N',
      3: 'R', 4: 'Q', 5: 'K', 
      6: 'p', 7: 'b', 8: 'n',
      9: 'r', 10: 'q', 11: 'k'}

    list_of_iou=[]
    
    for i in detections:

        box_x1 = i[0]
        box_y1 = i[1]

        box_x2 = i[2]
        box_y2 = i[1]

        box_x3 = i[2]
        box_y3 = i[3]

        box_x4 = i[0]
        box_y4 = i[3]
              
        if box_y4 - box_y1 > 60:
            box_complete = np.array([[box_x1,box_y1+40], [box_x2, box_y2+40], [box_x3, box_y3], [box_x4, box_y4]])
        else:
            box_complete = np.array([[box_x1,box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])

        list_of_iou.append(calculateIoU(box_complete, square))

    num = list_of_iou.index(max(list_of_iou))

    piece = boxes.cls[num].tolist()
    
    if max(list_of_iou) > 0.15:
        piece = boxes.cls[num].tolist()
        return pieceLetter[piece]
    
    else:
        piece = "1"
        return piece
    
    
def saveFENUrlToFile(fen_url):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    text_file_name = os.path.normpath(os.path.join(OUTPUT_DIR, base_name + "_fen.txt"))
    with open(text_file_name, 'w') as file:
        file.write(fen_url)
    print(f"Saved FEN URL to: {text_file_name}")


image_path = os.path.join(IMAGE_DIR, IMAGE)

transformed_image = transformImage(image_path)
detections, boxes = detectChessPieces(transformed_image)
ptsT, ptsL = makeGrid(transformed_image)

xA = ptsT[0][0]
xB = ptsT[1][0]
xC = ptsT[2][0]
xD = ptsT[3][0]
xE = ptsT[4][0]
xF = ptsT[5][0]
xG = ptsT[6][0]
xH = ptsT[7][0]
xI = ptsT[8][0]

y9 = ptsL[0][1]
y8 = ptsL[1][1] 
y7 = ptsL[2][1] 
y6 = ptsL[3][1]  
y5 = ptsL[4][1]  
y4 = ptsL[5][1] 
y3 = ptsL[6][1]  
y2 = ptsL[7][1] 
y1 = ptsL[8][1] 

a8 = np.array([[xA,y9], [xB, y9], [xB, y8], [xA, y8]])
a7 = np.array([[xA,y8], [xB, y8], [xB, y7], [xA, y7]])
a6 = np.array([[xA,y7], [xB, y7], [xB, y6], [xA, y6]])
a5 = np.array([[xA,y6], [xB, y6], [xB, y5], [xA, y5]])
a4 = np.array([[xA,y5], [xB, y5], [xB, y4], [xA, y4]])
a3 = np.array([[xA,y4], [xB, y4], [xB, y3], [xA, y3]])
a2 = np.array([[xA,y3], [xB, y3], [xB, y2], [xA, y2]])
a1 = np.array([[xA,y2], [xB, y2], [xB, y1], [xA, y1]])

b8 = np.array([[xB,y9], [xC, y9], [xC, y8], [xB, y8]])
b7 = np.array([[xB,y8], [xC, y8], [xC, y7], [xB, y7]])
b6 = np.array([[xB,y7], [xC, y7], [xC, y6], [xB, y6]])
b5 = np.array([[xB,y6], [xC, y6], [xC, y5], [xB, y5]])
b4 = np.array([[xB,y5], [xC, y5], [xC, y4], [xB, y4]])
b3 = np.array([[xB,y4], [xC, y4], [xC, y3], [xB, y3]])
b2 = np.array([[xB,y3], [xC, y3], [xC, y2], [xB, y2]])
b1 = np.array([[xB,y2], [xC, y2], [xC, y1], [xB, y1]])

c8 = np.array([[xC,y9], [xD, y9], [xD, y8], [xC, y8]])
c7 = np.array([[xC,y8], [xD, y8], [xD, y7], [xC, y7]])
c6 = np.array([[xC,y7], [xD, y7], [xD, y6], [xC, y6]])
c5 = np.array([[xC,y6], [xD, y6], [xD, y5], [xC, y5]])
c4 = np.array([[xC,y5], [xD, y5], [xD, y4], [xC, y4]])
c3 = np.array([[xC,y4], [xD, y4], [xD, y3], [xC, y3]])
c2 = np.array([[xC,y3], [xD, y3], [xD, y2], [xC, y2]])
c1 = np.array([[xC,y2], [xD, y2], [xD, y1], [xC, y1]])

d8 = np.array([[xD,y9], [xE, y9], [xE, y8], [xD, y8]])
d7 = np.array([[xD,y8], [xE, y8], [xE, y7], [xD, y7]])
d6 = np.array([[xD,y7], [xE, y7], [xE, y6], [xD, y6]])
d5 = np.array([[xD,y6], [xE, y6], [xE, y5], [xD, y5]])
d4 = np.array([[xD,y5], [xE, y5], [xE, y4], [xD, y4]])
d3 = np.array([[xD,y4], [xE, y4], [xE, y3], [xD, y3]])
d2 = np.array([[xD,y3], [xE, y3], [xE, y2], [xD, y2]])
d1 = np.array([[xD,y2], [xE, y2], [xE, y1], [xD, y1]])

e8 = np.array([[xE,y9], [xF, y9], [xF, y8], [xE, y8]])
e7 = np.array([[xE,y8], [xF, y8], [xF, y7], [xE, y7]])
e6 = np.array([[xE,y7], [xF, y7], [xF, y6], [xE, y6]])
e5 = np.array([[xE,y6], [xF, y6], [xF, y5], [xE, y5]])
e4 = np.array([[xE,y5], [xF, y5], [xF, y4], [xE, y4]])
e3 = np.array([[xE,y4], [xF, y4], [xF, y3], [xE, y3]])
e2 = np.array([[xE,y3], [xF, y3], [xF, y2], [xE, y2]])
e1 = np.array([[xE,y2], [xF, y2], [xF, y1], [xE, y1]])

f8 = np.array([[xF,y9], [xG, y9], [xG, y8], [xF, y8]])
f7 = np.array([[xF,y8], [xG, y8], [xG, y7], [xF, y7]])
f6 = np.array([[xF,y7], [xG, y7], [xG, y6], [xF, y6]])
f5 = np.array([[xF,y6], [xG, y6], [xG, y5], [xF, y5]])
f4 = np.array([[xF,y5], [xG, y5], [xG, y4], [xF, y4]])
f3 = np.array([[xF,y4], [xG, y4], [xG, y3], [xF, y3]])
f2 = np.array([[xF,y3], [xG, y3], [xG, y2], [xF, y2]])
f1 = np.array([[xF,y2], [xG, y2], [xG, y1], [xF, y1]])

g8 = np.array([[xG,y9], [xH, y9], [xH, y8], [xG, y8]])
g7 = np.array([[xG,y8], [xH, y8], [xH, y7], [xG, y7]])
g6 = np.array([[xG,y7], [xH, y7], [xH, y6], [xG, y6]])
g5 = np.array([[xG,y6], [xH, y6], [xH, y5], [xG, y5]])
g4 = np.array([[xG,y5], [xH, y5], [xH, y4], [xG, y4]])
g3 = np.array([[xG,y4], [xH, y4], [xH, y3], [xG, y3]])
g2 = np.array([[xG,y3], [xH, y3], [xH, y2], [xG, y2]])
g1 = np.array([[xG,y2], [xH, y2], [xH, y1], [xG, y1]])

h8 = np.array([[xH,y9], [xI, y9], [xI, y8], [xH, y8]])
h7 = np.array([[xH,y8], [xI, y8], [xI, y7], [xH, y7]])
h6 = np.array([[xH,y7], [xI, y7], [xI, y6], [xH, y6]])
h5 = np.array([[xH,y6], [xI, y6], [xI, y5], [xH, y5]])
h4 = np.array([[xH,y5], [xI, y5], [xI, y4], [xH, y4]])
h3 = np.array([[xH,y4], [xI, y4], [xI, y3], [xH, y3]])
h2 = np.array([[xH,y3], [xI, y3], [xI, y2], [xH, y2]])
h1 = np.array([[xH,y2], [xI, y2], [xI, y1], [xH, y1]])


FEN_annotation = [[a8, b8, c8, d8, e8, f8, g8, h8],
                [a7, b7, c7, d7, e7, f7, g7, h7],
                [a6, b6, c6, d6, e6, f6, g6, h6],
                [a5, b5, c5, d5, e5, f5, g5, h5],
                [a4, b4, c4, d4, e4, f4, g4, h4],
                [a3, b3, c3, d3, e3, f3, g3, h3],
                [a2, b2, c2, d2, e2, f2, g2, h2],
                [a1, b1, c1, d1, e1, f1, g1, h1]]

board_FEN = []
complete_board_FEN = []

for line in FEN_annotation:
    line_to_FEN = []
    for square in line:
        piece_on_square = connectDetectionToSquare(detections, square)    
        line_to_FEN.append(piece_on_square)
    print(line_to_FEN)
    board_FEN.append(line_to_FEN)

complete_board_FEN = [''.join(line) for line in board_FEN] 

to_FEN = '/'.join(complete_board_FEN)

fen_url = "https://lichess.org/analysis/" + to_FEN
print(fen_url)
saveFENUrlToFile(fen_url)