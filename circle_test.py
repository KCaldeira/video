import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

def test_radial_symmetry(image, center_x, center_y, max_radius=None, num_angles=36):
    """
    Test radial symmetry around a specified point in an image.
    
    Parameters:
    - image: Input image (grayscale)
    - center_x, center_y: Center point coordinates
    - max_radius: Maximum radius to test (defaults to distance to nearest image edge)
    - num_angles: Number of angles to sample (default 36, meaning every 10 degrees)
    
    Returns:
    - symmetry_score: Value between 0 and 1 indicating degree of radial symmetry
    - radial_profile: Array of average pixel values at each radius
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    height, width = image.shape
    center_x, center_y = int(center_x), int(center_y)
    
    # Calculate maximum radius if not specified
    if max_radius is None:
        max_radius = min(
            min(center_x, width - center_x),
            min(center_y, height - center_y)
        )
    
    # Create coordinate grid
    y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
    
    # Calculate radial distances
    r = np.sqrt(x*x + y*y)
    
    # Calculate angles
    theta = np.arctan2(y, x)
    
    # Initialize arrays for symmetry calculation
    radial_profile = np.zeros(max_radius)
    symmetry_scores = np.zeros(max_radius)
    
    # For each radius
    for radius in range(1, max_radius):
        # Create mask for this radius
        mask = (r >= radius-0.5) & (r < radius+0.5)
        
        if np.any(mask):
            # Get pixel values at this radius
            values = image[mask]
            angles = theta[mask]
            
            # Calculate mean value at this radius
            radial_profile[radius] = np.mean(values)
            
            # Calculate symmetry score
            # Compare values at opposite angles
            symmetry_sum = 0
            count = 0
            
            for angle in np.linspace(0, np.pi, num_angles):
                # Find points at this angle and its opposite
                angle_mask1 = (angles >= angle-0.1) & (angles < angle+0.1)
                angle_mask2 = (angles >= angle+np.pi-0.1) & (angles < angle+np.pi+0.1)
                
                if np.any(angle_mask1) and np.any(angle_mask2):
                    val1 = np.mean(values[angle_mask1])
                    val2 = np.mean(values[angle_mask2])
                    symmetry_sum += 1 - abs(val1 - val2) / 255.0
                    count += 1
            
            if count > 0:
                symmetry_scores[radius] = symmetry_sum / count
    
    # Overall symmetry score is the average of all radius scores
    overall_symmetry = np.mean(symmetry_scores)
    
    return overall_symmetry, radial_profile

def houghCircleTransform(dp=2, minDist=10, param1=100, param2=2, minRadius=0, maxRadius=0):
    root = os.getcwd()
    imgPath = os.path.join(root, "Screenshot 2025-04-25 132550.png")
    imgRGB = cv.cvtColor(cv.imread(imgPath), cv.COLOR_BGR2RGB)
    imgGray = cv.cvtColor(imgRGB, cv.COLOR_BGR2GRAY)

    #imgGray = cv.medianBlur(imgGray, 21)

    # Process each set of circles and calculate confidence
    for imageChannel, color in [(imgGray, 'Gray'), (imgRGB[0], 'Red'), (imgRGB[1], 'Green'), (imgRGB[2], 'Blue')]:
        circle_set = cv.HoughCircles(imageChannel, cv.HOUGH_GRADIENT, dp, minDist=minDist, param1=param1, 
                                    param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if circle_set is not None:
            print(f"\n{color} channel circles:")
            print(f"Number of circles: {circle_set.shape[1]}")
            
            # Calculate confidence for each circle
            for i, circle in enumerate(circle_set[0]):
                x, y, r = circle
                
                # Test radial symmetry at this point
                symmetry_score, radial_profile = test_radial_symmetry(imgGray, x, y, max_radius=int(r*1.5))
                
                # Create a mask for the circle
                mask = np.zeros_like(imgGray)
                cv.circle(mask, (int(x), int(y)), int(r), 255, 1)
                
                # Get edge points using Canny
                edges = cv.Canny(imgGray, 50, 150)
                
                # Count edge points that lie on the circle
                edge_points = np.sum((mask > 0) & (edges > 0))
                
                # Calculate confidence based on edge support and symmetry
                circumference = 2 * np.pi * r
                edge_confidence = edge_points / (circumference + 1e-6)  # Avoid division by zero
                combined_confidence = (edge_confidence + symmetry_score) / 2
                
                print(f"Circle {i+1}: Center=({x:.1f}, {y:.1f}), Radius={r:.1f}")
                print(f"Edge Confidence: {edge_confidence:.3f}, Symmetry Score: {symmetry_score:.3f}, Combined: {combined_confidence:.3f}")
                
                # Draw circle with color intensity based on combined confidence
                color_intensity = int(255 * combined_confidence)
                cv.circle(imgRGB, (int(x), int(y)), int(r), (color_intensity, color_intensity, color_intensity), 2)
        else:
            print(f"\nNo circles found in {color} channel")

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

if __name__ == '__main__':
    houghCircleTransform(dp=2, minDist=100, param1=50, param2=30, minRadius=200, maxRadius=1500)
