# The project aims to detect the corners of the chessboard and identify the pieces (type and color) on given squares.

![Slide3](https://github.com/user-attachments/assets/607662d6-1c72-4334-afc3-79781c5511fe)

![Slide2](https://github.com/user-attachments/assets/060f5839-92c0-4e91-840a-24024657b371)

# Dataset
The dataset for training and testing the model consists of synthetic images with chessboards.
https://www.kaggle.com/datasets/thefamousrat/synthetic-chess-board-images/data

![Slide4](https://github.com/user-attachments/assets/66f6762e-8e34-45dc-9404-840babaa56d3)

# How it works
![Slide8](https://github.com/user-attachments/assets/e3cf3443-755d-4325-b6c5-38589a7963af)

# How to use it
Before running the script, you must update global variables with the 
paths to the images and output folders and specify the image file to 
process. 
You can run it from console.

Steps to run the script: 
1. Navigate to the directory where the script is located. 
2. Execute this command (if you have installed Python and added to 
environment variables):
```bash
python  .\generateBoardState.py
``` 
4. Enter orientation of the chessboard on the original image. 
5. You will receive a visualization of the chessboard in the console 
and URL to Lichess.org analysis site with the generated FEN notation. 
Also, there is a folder specified in global variables (OUTPUT_DIR) 
with saved results.

# Examples
![Slide14](https://github.com/user-attachments/assets/6a9d5757-b5d0-4ba6-86b9-2f6dd2c8934d)

![Slide15](https://github.com/user-attachments/assets/c84bd25c-6ee9-4d78-bccb-087a5ec3e95f)

![Slide21](https://github.com/user-attachments/assets/578ae645-9687-4bcb-8935-e931c511bbbf)

![Slide22](https://github.com/user-attachments/assets/d5e0a7ed-aaef-428f-9f3f-4a4be90471a3)

# Warning
YOOLO v8 models are not provided.
To run script, you need to ensure that the model files are located in the same directory.

# References
James Gallagher. (Mar 10, 2023). Represent Chess Boards Digitally with Computer Vision. Roboflow Blog: https://blog.roboflow.com/chess-boards/
