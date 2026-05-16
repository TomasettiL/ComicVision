import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import cv2
import os

def main():
    #img2 = Image.open(r"rukia.png")
    #SobelImage(img2)
    """
    img = cv2.imread("rukia.png", cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (3,3), 1)
    edges = cv2.Canny(img, threshold1=20, threshold2=90)
    edges_blur = cv2.Canny(blurred, threshold1=50, threshold2=150)
    edges_inv = 255 - edges
    cv2.imwrite("edges_canny.png", edges)
    cv2.imwrite("edges_canny_blur.png", edges_blur)
    cv2.imwrite("edges_canny_inv.png", edges_inv)
    """
    img = cv2.imread("rukia.png", cv2.IMREAD_GRAYSCALE)
    # Sobel in X-direction (horizontal edges)
    sobel_x = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    # Sobel in Y-direction (vertical edges)
    sobel_y = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    # Compute gradient magnitude
    #magnitude = cv2.addWeighted(np.abs(sobel_x), 0.5, np.abs(sobel_y), 0.5, 0)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    # Normalize to 0-255 for display
    #magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype('uint8')
    inverted = 255 - magnitude
    cv2.imwrite("sobel_magnitude.png", magnitude)
    cv2.imwrite("sobel_magnitude_inv.png", inverted)
    
    blurred = cv2.GaussianBlur(magnitude, (3,3), 1)
    # Canny thresholds: tune for your image
    canny_lower = 50
    canny_upper = 100
    edges = cv2.Canny(blurred, canny_lower, canny_upper)
    # Invert Canny edges: black edges on white
    edges_black = 255 - edges
    # --- Save final Canny result ---
    cv2.imwrite("canny_after_sobel.png", edges_black)

    input_path = "bleach1.mp4"
    output_path = "bleach_edges.mp4"
    output_path2 = "bleach_edges_sobel.mp4"
    output_path3 = "bleach_edges_sobel_b.mp4"
    output_path4 = "bleach_edges_sobcan.mp4"

    canny_lower = 40
    canny_upper = 100
    gaussian_kernel = (3,3)
    sigma = 1

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames1 = []
    frames2 = []
    frames3 = []
    frames4 = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, gaussian_kernel, sigma)
        edges = cv2.Canny(blurred, canny_lower, canny_upper)
        black_edges = 255 - edges
        # --- Square edges safely ---
        edges_float = black_edges.astype(np.float32)  # convert to float
        edges_squared = edges_float ** 2              # square pixel values
        edges_squared = np.clip(edges_squared, 0, 255)  # cap at 255
        edges_bold = edges_squared.astype(np.uint8)      # back to uint8
        #edges_soft = cv2.GaussianBlur(edges_bold, (3,3), 1)
        edges_bgr1 = cv2.cvtColor(edges_bold, cv2.COLOR_GRAY2BGR)
        frames1.append(edges_bgr1)

        #Sobel
        # Sobel in X-direction (horizontal edges)
        sobel_x = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        # Sobel in Y-direction (vertical edges)
        sobel_y = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        # Compute gradient magnitude
        #magnitude = cv2.addWeighted(np.abs(sobel_x), 0.5, np.abs(sobel_y), 0.5, 0)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        # Normalize to 0-255 for display
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = magnitude.astype('uint8')
        #inverted = 255 - magnitude
        #edges_soft = cv2.GaussianBlur(magnitude, (3,3), 1)
        # --- Square Sobel edges safely ---
        #edges_float = edges_soft.astype(np.float32)
        #edges_squared = edges_float ** 1.5
        #edges_squared = np.clip(edges_squared, 0, 255)
        #edges_bold = edges_squared.astype(np.uint8)
        #edges_soft = cv2.GaussianBlur(edges_bold, (3,3), 1)
        inverted = 255 - magnitude
        edges_bgr2 = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
        frames2.append(edges_bgr2)

        edges_bgr3 = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
        frames3.append(edges_bgr3)

        #canny after sobel
        blurred = cv2.GaussianBlur(magnitude, (5,5), 1.5)
        # Canny thresholds: tune for your image
        canny_lower = 50
        canny_upper = 120
        edges = cv2.Canny(blurred, canny_lower, canny_upper)
        #blurred_edges = cv2.GaussianBlur(edges, (3,3), 1)
        # Invert Canny edges: black edges on white
        edges_black = 255 - edges
        edges_bgr4 = cv2.cvtColor(edges_black, cv2.COLOR_GRAY2BGR)
        frames4.append(edges_bgr4)

    cap.release()
    print(f"Processed {len(frames1)} frames.")
    height, width, _ = frames1[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames1:
        out.write(frame)
    out.release()
    print(f"Saved edge-detected video to {output_path}")

    print(f"Processed {len(frames2)} frames.")
    height, width, _ = frames2[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path2, fourcc, fps, (width, height))
    for frame in frames2:
        out.write(frame)
    out.release()
    print(f"Saved edge-detected video to {output_path2}")

    print(f"Processed {len(frames3)} frames.")
    height, width, _ = frames3[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path3, fourcc, fps, (width, height))
    for frame in frames3:
        out.write(frame)
    out.release()
    print(f"Saved edge-detected video to {output_path3}")

    print(f"Processed {len(frames4)} frames.")
    height, width, _ = frames4[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path4, fourcc, fps, (width, height))
    for frame in frames4:
        out.write(frame)
    out.release()
    print(f"Saved edge-detected video to {output_path4}")



if __name__ == '__main__':
    main()