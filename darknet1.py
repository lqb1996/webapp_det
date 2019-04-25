from ctypes import *
import math
import random
import cv2
from PIL import Image,ImageFont,ImageDraw
from skimage import io
from timeit import default_timer as timer

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/lihanyu/pycodes/beifen/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE
from timeit import default_timer as timer
get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

#''''''
ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE
#''''''''

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, img, thresh=.5, hier_thresh=.5, nms=.45):
    #im = load_image(image, 0, 0)
    im=nparray_to_image(img)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

#""""""""""
def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image

#''''''

# def showPicResult(image):
#     img = cv2.imread(image)
#     cv2.imwrite(out_img, img)
#     video_dir = '/home/lihanyu/pycodes/darknet/data/oou.avi'
#     fps=20
#     img_size=(841,1023)
#     #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # opencv3.0
#     fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#     videoWriter = cv2.VideoWriter(video_dir,fourcc, fps, img_size)
#     #print(out_img)
#     for i in range(len(r)):
#         x1=r[i][2][0]-r[i][2][2]/2
#         y1=r[i][2][1]-r[i][2][3]/2
#         x2=r[i][2][0]+r[i][2][2]/2
#         y2=r[i][2][1]+r[i][2][3]/2
#         im = cv2.imread(out_img)
#
#         cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
#         cv2.putText(arr, label, (int(xmin), int(ymin)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,color=(0, 255, 255), thickness=1)
#         #This is a method that works well.
#         cv2.imwrite(out_img, im)
#     out_img1=cv2.imread(out_img)
#     videoWriter.write(out_img1)
#     videoWriter.release()
#     # print("finish")
#         #print(out_img)
#     #cv2.imshow('yolo_image_detector', cv2.imread(out_img))
#     #cv2.imshow('yolo_image_detector', out_img)


    
if __name__ == "__main__":
    net = load_net(b"/home/lihanyu/pycodes/beifen/darknet/cfg/yolov3.cfg", b"/home/lihanyu/pycodes/beifen/darknet/yolov3.weights", 0)
    meta = load_meta(b"/home/lihanyu/pycodes/beifen/darknet/cfg/coco.data")

    # out_img = ("/home/lihanyu/pycodes/beifen/darknet/data/test_result.jpg")
    # video_tmp = ("/home/lihanyu/pycodes/beifen/darknet/data/video_tmp.jpg")
    origin_video = ('/home/lihanyu/pycodes/beifen/darknet/data/1.wmv')
    #video_dir = ('/home/lihanyu/pycodes/darknet/data/ooo.mp4')

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # VideoW = cv2.VideoWriter(video_dir, fourcc, 30.0, (1280, 720))

    count=0
    cap = cv2.VideoCapture(origin_video)
    #img=cv2.imread(origin_video)
    while cap.isOpened():
        ret, img = cap.read()
        if ret == False:
            break
        boxes = detect(net, meta, img)
        print(boxes)
        #free_image(im)
        for i in range(len(boxes)):
            score = boxes[i][1]
            label = boxes[i][0]
            # label=boxes[i][0]
            xmin = boxes[i][2][0] - boxes[i][2][2] / 2
            ymin = boxes[i][2][1] - boxes[i][2][3] / 2
            xmax = boxes[i][2][0] + boxes[i][2][2] / 2
            ymax = boxes[i][2][1] + boxes[i][2][3] / 2;

            print(xmin,ymin,xmax,ymax)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 1)
            cv2.putText(img, str(label), (int(xmin), int(ymin+50)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(0, 0, 255), thickness=1)

        cv2.imshow("Canvas", img)  # 13
        cv2.waitKey( )
    #VideoW.write(img)
    # VideoW.write(arr)
#cv2.destroyAllWindows()

    # if count == 100:
    #     break

    # img_arr = Image.fromarray(img)
    # img_goal = img_arr.save(video_tmp)
    #showPicResult(video_tmp)
#cap.release()

    # cap = cv2.VideoCapture(video_dir)
    # re,er=cap.read()
    # if re==True:
    #     print(er)

