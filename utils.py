import cv2
import math
import numpy as np

def distance(p1, p2):
    """Calcule la distance entre deux points p1 et p2."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_paralelogramme(points):
    """Détermine si les quatre points forment presque un paralélogramme."""
    
    if len(points) != 4:
        raise ValueError("Il faut exactement quatre points.")
    
    hypothenuseAD = 0
    
    A = 0
    D = 0
    
    for i in range(4):
        for j in range(i+1,4):
            if distance(points[i], points[j]) > hypothenuseAD:
                hypothenuseAD = distance(points[i], points[j])
                A = i
                D = j
    if(hypothenuseAD<250):
        return False
    
    if A == 0:
        if D == 1:
            B = 2
            C = 3
        if D == 2:
            B = 1
            C = 3
        if D == 3:
            B = 2
            C = 1
    if A == 1:
        if D ==2 :
            B = 0
            C = 3
        if D ==3:
            B = 2
            C = 0
    if A == 2:
        if D == 3:
            B = 1
            C = 0
       
    milieu_AD = ((points[A][0]+points[D][0])/2,(points[A][1]+points[D][1])/2)
    milieu_BC = ((points[B][0]+points[C][0])/2,(points[B][1]+points[C][1])/2)
    
    if abs(milieu_AD[0] - milieu_BC[0]) < hypothenuseAD*3/100 and abs(milieu_AD[1] - milieu_BC[1]) < hypothenuseAD*3/100:
        return True
    
    return False

def draw_rectangle_with_corners(image, corners):
    
    image_with_card = image.copy()
    
    # Dessiner le rectangle détecté
    cv2.drawContours(image_with_card, [corners], -1, (0, 255, 0), 2)
                
    # Dessiner les coins
    cv2.circle(image_with_card, tuple(corners[0][0]), 5, (255, 0, 0), -1)
    cv2.circle(image_with_card, tuple(corners[1][0]), 5, (255, 0, 0), -1)
    cv2.circle(image_with_card, tuple(corners[2][0]), 5, (255, 0, 0), -1)
    cv2.circle(image_with_card, tuple(corners[3][0]), 5, (255, 0, 0), -1)
    
    return image_with_card

# Swaps the values of at two indexes in the given array
def swap(arr, ind1, ind2):
    temp = arr[ind1]
    arr[ind1] = arr[ind2]
    arr[ind2] = temp

# Returns sorted array and array of indexes of locations of original values
# Selection sort is used as efficieny won't matter as much for n = 3 or 4
def sortVals(vals):
    indexes = list(range(len(vals)))
    for i in range(len(vals)):
        index = i
        minval = vals[i]
        for j in range(i, len(vals)):
            if vals[j] < minval:
                minval = vals[j]
                index = j
        swap(vals, i, index)
        swap(indexes, i, index)
    return vals, indexes

def reorderCorners(corners):
    # Copy corner values into xvals and yvals
    xvals = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
    yvals = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]]

    # Sort yvals and get indexes of original values in sorted array
    yvals, idxs = sortVals(yvals)

    # Change xvals to same order as yvals
    temp = xvals.copy()
    for i in range(len(idxs)):
        xvals[i] = temp[idxs[i]]

    # Check if card is horizontal or vertical and make sure [0, 0] is point closest to top left of image (smallest x)
    if yvals[0] == yvals[1]:
        if xvals[1] < xvals[0]:
            # yvals are same so only swap xvals
            tempx = xvals[0]
            xvals[0] = xvals[1]
            xvals[1] = tempx

    # Find distance from corner with min y to corners
    dist1 = math.sqrt((xvals[1] - xvals[0]) ** 2 + (yvals[1] - yvals[0]) ** 2)
    dist2 = math.sqrt((xvals[2] - xvals[0]) ** 2 + (yvals[2] - yvals[0]) ** 2)
    dist3 = math.sqrt((xvals[3] - xvals[0]) ** 2 + (yvals[3] - yvals[0]) ** 2)
    dists = [dist1, dist2, dist3]

    # Sort distances and get indexes of original values in sorted array
    distSorted, idxsDist = sortVals(dists.copy())

    # Reformat index array to be 4 values, not necessary but makes code easier to read
    idxsDist.insert(0, 0)
    idxsDist[1] += 1
    idxsDist[2] += 1
    idxsDist[3] += 1

    # Check if card is vertical/horizontal
    if yvals[0] == yvals[1]:
        if dists[0] == distSorted[0]:  # If card is vertical; corner [0, 0] is top left of card
            topleft = [xvals[idxsDist[0]], yvals[idxsDist[0]]]  # Same as [xvals[0], yvals[0]]
            topright = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            bottomright = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
            bottomleft = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
        else:  # If card is horizontal; corner [0, 0] is top right of card
            topleft = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            topright = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            bottomright = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
            bottomleft = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
    else:  # Else card is tilted in some other orientation
        if xvals[idxsDist[1]] == min(xvals):  # Left-most point is the closest to the point with the smallest y value
            # Left-most point is top left corner
            topleft = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            topright = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            bottomright = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
            bottomleft = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
        else:  # Corner [0, 0] is the top left corner
            topleft = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            topright = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            bottomright = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
            bottomleft = [xvals[idxsDist[2]], yvals[idxsDist[2]]]

    return [[topleft], [topright], [bottomleft], [bottomright]]