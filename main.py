import cv2
from fpdf import FPDF
from imutils import rotate_bound
import os
from datetime import datetime
from skimage.measure import compare_ssim

start = datetime.now()

file_name = "Databases"
file_extension = ".mp4"

video = cv2.VideoCapture(file_name + file_extension)
# Set the sampling rate (lower value means more frames processed but higher execution time)
sample_rate = 500.0  # in ms
curr_time = 0.0

total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)

previous_frame = None
# PDF file parameters
pdf = FPDF(orientation='L', unit='pt', format=(720*1.1, 1280*1.1))
pdf.set_auto_page_break(0)

while video.isOpened():

    check, frame = video.read()
    # Check if it has been open correctly
    if not check:
        break

    name = "frame" + str(video.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg"
    frame = rotate_bound(frame, 0)
    # If it is the first frame add it to the pdf
    if previous_frame is None:
        cv2.imwrite(name, frame)
        pdf.add_page()
        pdf.image(name)
        previous_frame = frame
        os.remove(name)
        continue
    # Absolute difference of frames (multichannel for colorful images)
    (score, diff) = compare_ssim(frame, previous_frame, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")
    # Check that the difference is relevant
    if score <= 0.95:
        # Save an image from the frame
        cv2.imwrite(name, frame)
        # print("SSIM: {}".format(score))
        previous_frame = frame
        # Create a new page that will contain the image
        pdf.add_page()
        pdf.image(name)
        # Delete the images
        os.remove(name)

    # Set the next frame
    video.set(cv2.CAP_PROP_POS_MSEC, curr_time + sample_rate)
    curr_time += sample_rate

video.release()

pdf.output(file_name + ".pdf", "F")
print("Done! Time: " + str(datetime.now() - start))
