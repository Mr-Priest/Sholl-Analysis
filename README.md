# Sholl-Analysis
Sholl analysis to quantify intersections and total protrusion length within 20um


Run the py file on your computer. Python version: 3.9.10



1. Select the images you want to analyze. Replace "Example.tif" with your image path.
   Use Grey scale, single channel, 8 bit image as input.
   
image_path = "**EXAMPLE.tif**"  # Replace with your image path

2. Run the python file, the image you select will appear
   Click on the soma of the OPCs, a circle will appear. Scroll and drag to change the position of the image.
  do not choose OPCs that are adjacent to others or near unwantted signals
  Press "q" when finished, the analysis will start. 

<img width="1897" height="1013" alt="image" src="https://github.com/user-attachments/assets/a0ddea0f-689d-4388-91dc-81c082441f46" />


3. Excel and Sholl plot will be saved to folder "excel" and "plots".
The plot shows the "original image", "skeletonized image with circles" and the "original image with circles"

<img width="2559" height="873" alt="image" src="https://github.com/user-attachments/assets/eea8f7a9-4972-4066-93b2-73de34257aa0" />

<img width="1680" height="150" alt="image" src="https://github.com/user-attachments/assets/f9ef9fa8-9fe6-45d8-8bf0-eeb0140fb227" />

