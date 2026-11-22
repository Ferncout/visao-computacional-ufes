"""

Challenge 03 - Camera Calibration with ChArUco
Nome: Fernanda Coutinho Correa Calazans

"""

import glob
import time
import zipfile

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

########################################################
# Complementary functions for ploting points and vectors with Y-axis swapped with Z-axis
def set_plot(ax=None,figure = None,figsize=(9,8),limx=[-2,2],limy=[-2,2],limz=[-2,2]):
    if figure ==None:
        figure = plt.figure(figsize=(9,8))
    if ax==None:
        ax = plt.axes(projection='3d')

    ax.set_xlim(limx)
    ax.set_xlabel("x axis")
    ax.set_ylim(limy)
    ax.set_ylabel("y axis")
    ax.set_zlim(limz)
    ax.set_zlabel("z axis")
    return ax

#adding quivers to the plot
def draw_arrows(point,base,axis,length=1.5):
    # Plot vector of x-axis
    axis.quiver(point[0],point[1],point[2],base[0,0],base[1,0],base[2,0],color='red',pivot='tail',   length=length)
    # Plot vector of y-axis
    axis.quiver(point[0],point[1],point[2],base[0,1],base[1,1],base[2,1],color='green',pivot='tail', length=length)
    # Plot vector of z-axis
    axis.quiver(point[0],point[1],point[2],base[0,2],base[1,2],base[2,2],color='blue',pivot='tail',  length=length)

    return axis


### Setting printing options
np.set_printoptions(precision=8,suppress=True)

#########################################################

"""# Read images and detect the ChArUco corners"""

# Flag to read from zip file
ZipFile = True
ZipFileName = 'canon_charuco.zip'

# Image path
img_path = './canon_charuco/'

if ZipFile:
    # Read zip file and unzip images
    zip_ref = zipfile.ZipFile(ZipFileName,'r')
    zip_ref.extractall('./')
    zip_ref.close()

# Flag to show or not the detections
SHOW_MATPLOTLIB = False
SHOW_CV = True

# ChArUco board parameters (units in millimeters)
# Board type 9x7 -> 9 squares in x and 7 squares in y
squares_x     = 9
squares_y     = 7
square_size   = 45   # 4.5 cm
marker_size   = 33   # 3.3 cm
dictionary_id = cv2.aruco.DICT_4X4_50

# Minimum number of ChArUco corners detected in an image to use it in the calibration
MIN_CORNERS = 4

# Create the dictionary, the ChArUco board and the ChArUco detector
arucoDict = cv2.aruco.getPredefinedDictionary(dictionary_id)
board = cv2.aruco.CharucoBoard((squares_x,squares_y), square_size, marker_size, arucoDict)
charucoParams = cv2.aruco.CharucoParameters()
# Accept images where only one ArUco marker is found to interpolate the ChArUco corners
charucoParams.minMarkers = 1
arucoParams = cv2.aruco.DetectorParameters()
charucoDetector = cv2.aruco.CharucoDetector(board, charucoParams, arucoParams)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in the world frame
imgpoints = [] # 2d points in image plane.

# Read images for calibration (the images have the extension .JPG)
# (set() removes duplicates: on Windows glob is not case sensitive, so both patterns match the same files)
images = sorted(set(glob.glob(img_path+'*.JPG') + glob.glob(img_path+'*.jpg')))
# Print the number of images to see if all of them were read
print('Number of images read: ', len(images))

# Read images, detect the markers and the ChArUco corners and plot the detections
if SHOW_MATPLOTLIB:
    fig = plt.figure(figsize=(10,10))
    plt.ion()
    plt.show()

# Counter of images where at least one ArUco marker was detected
num_markers_detected = 0

for fname in images:
    # Read image
    img = cv2.imread(fname)
    # Verify if the image has landscape format. If not, rotate image
    imsize = img.shape
    if (imsize[1] < imsize[0]):
            img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Detect the ArUco markers and the ChArUco corners
    charucoCorners, charucoIds, markerCorners, markerIds = charucoDetector.detectBoard(gray)

    if markerIds is not None:
        num_markers_detected += 1

    # If enough ChArUco corners were found, store object points and image points
    if charucoIds is not None and len(charucoIds) >= MIN_CORNERS:

        # Match the detected ChArUco corners to their 3D position in the board (object points)
        # and to their 2D position in the image (image points)
        objp, imgp = board.matchImagePoints(charucoCorners, charucoIds)

        # Store object points and the corresponding image points
        objpoints.append(objp)
        imgpoints.append(imgp)

        # Draw and display the markers and the ChArUco corners
        if SHOW_MATPLOTLIB or SHOW_CV:
            img = cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
            img = cv2.aruco.drawDetectedCornersCharuco(img, charucoCorners.reshape(-1,1,2), charucoIds, (0,0,255))

        # When using Colab, we can not use cv2.imshow. So we are showing images
        # with matplotlib.pyplot

        if SHOW_MATPLOTLIB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            fig.canvas.draw()
            # This will run the GUI event
            # loop until all UI events
            # currently waiting have been processed
            fig.canvas.flush_events()
            time.sleep(0.01)
        if SHOW_CV:
            cv2.imshow('img',img)
            cv2.waitKey(100)


if SHOW_CV:
    cv2.destroyAllWindows()

if SHOW_MATPLOTLIB:
    plt.ioff()

# Print the number of images where the markers/corners could be detected
print('Number of images where the markers were detected', num_markers_detected)
print('Number of images used in the calibration (>= %d ChArUco corners)' % MIN_CORNERS, len(objpoints))

"""# Run Camera Calibration
##Print Intrinsic Matrix
##Print radial distortion coeficients
"""

# Run camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Print calibration results
print('Calibration Matrix: \n', mtx)
print('Radial distortion coeficients: \n', dist)
print('Ret: \n', ret)

# Organize the extrinsic parameters (rotation and translation) in 3xN arrays, where N is
# the number of images
transl = np.hstack(tvecs)
rot = np.hstack(rvecs)

print(transl.shape)
print(rot.shape)

"""## Undistort one image as example
## Calculate reprojection error
"""

img_file = images[0]
img = cv2.imread(img_file)
# Verify if the image has landscape format. If not, rotate image
if (img.shape[1] < img.shape[0]):
    img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
# Convert to RGB to show with matplotlib.pyplot
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
h,  w = img.shape[:2]
print('Image size: ',h,' ',w)


# Undistort one image as example
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# When using Colab, we can not use cv2.imshow. So we are showing images
# with matplotlib.pyplot
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Original Image")
plt.imshow(img)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Undistorted Image")
plt.imshow(dst)
plt.show()
#cv2.imshow('undistorted image',dst)
#cv2.waitKey(10)

# Calculate the reprojection error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    # matchImagePoints returns Nx2 points, while projectPoints returns Nx1x2
    imgpoints2 = imgpoints2.reshape(-1,2)
    error = cv2.norm(imgpoints[i].reshape(-1,2),imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print ("Total reprojection error: ", mean_error/len(objpoints))

# Uncomment if you are showing images using cv2.imshow
#cv2.destroyAllWindows()

"""# Show Extrinsic Parameters
## Considering the calibration pattern fixed at the origin and moving the camera.
"""

# Initialize figure
axis0 = set_plot(limx=[-400,800],limy=[-400,800],limz=[-800,0])
axis0.set_title('Camera Calibration - Fixed Board / Moving Camera')
# Create base vector values
e1 = np.array([[1],[0],[0],[0]]) # X
e2 = np.array([[0],[1],[0],[0]]) # Y
e3 = np.array([[0],[0],[1],[0]]) # Z
base = np.hstack((e1,e2,e3))
#origin point
origin =np.array([[0],[0],[0],[1]])
# Create camera frame and world frame
cam  = np.hstack([base,origin])
world  = np.hstack([base,origin])

# All the inner corners of the ChArUco board (3D points in the board frame, Z=0)
board_points = board.getChessboardCorners()

# Plot camera frames considering that the calibration patern is on the XY-plane and Z=0
for i in range(rot.shape[1]):  

    R,_ = cv2.Rodrigues(rot[:,i])
    t = transl[:,i]
    Rt = np.eye(4)
    Rt[0:3,0:3] = R
    Rt[0:3,-1] = t
    # To get the camera's rotation and translation we have to invert the
    # Extrinsic Parameters
    M = np.linalg.inv(Rt)
    # Transform the camera frame from the origin to the positon defined from
    # the extrinsic parameters
    new_cam = M@cam
    axis0 = draw_arrows(new_cam[:,-1],new_cam[:,0:3],axis0,80)

# Plot the calibration pattern as it was fixed and all the
# relative positions of the camera
X,Y = np.meshgrid(np.unique(board_points[:,0]),np.unique(board_points[:,1]))
#axis0 = draw_arrows(world[:,-1],world[:,0:3],world,80)
Z = np.zeros(X.shape)


axis0.plot_wireframe(X,Y,Z)
axis0.view_init(elev=-60,azim=-111,roll=23)
plt.show()

"""##Considering the camera fixed at the origin and moving the calibration pattern"""

# Initialize figure
axis1 = set_plot(limx=[-400,400],limy=[-400,400],limz=[-100,800])
axis1.set_title('Camera Calibration - Fixed Camera / Moving Board')
# Create base vector values
e1 = np.array([[1],[0],[0],[0]]) # X
e2 = np.array([[0],[1],[0],[0]]) # Y
e3 = np.array([[0],[0],[1],[0]]) # Z
base = np.hstack((e1,e2,e3))
#origin point
origin =np.array([[0],[0],[0],[1]])
# Create camera frame and world frame
cam  = np.hstack([base,origin])
world  = np.hstack([base,origin])
axis1 = draw_arrows(cam[:,-1],cam[:,0:3],axis1,80)


calib_points = board_points.T
#add a vector of ones to the chessboard points to represent them in homogeneous coordinates
calib_points = np.vstack([calib_points, np.ones(np.size(calib_points,1))])



# Plot calibration patter considering the camera fixed at (0,0,0)
for i in range(rot.shape[1]):

    # Set up the transformation matrix
    R,_ = cv2.Rodrigues(rot[:,i])
    t = transl[:,i]
    Rt = np.eye(4)
    Rt[0:3,0:3] = R
    Rt[0:3,-1] = t

    chessboard = Rt@calib_points
    new_world = Rt@world
    axis1.scatter(chessboard[0,:],chessboard[1,:],chessboard[2,:])
    #axis1 = draw_arrows(new_world[:,-1],new_world[:,0:3],axis1,80)


axis1.view_init(elev=-60,azim=-142,roll=53)

plt.show()
