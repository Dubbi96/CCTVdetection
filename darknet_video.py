from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import argparse
from ctypes import *
import math
import random
import os
import matplotlib.pyplot as plt
import pandas as pd

b = time.localtime()

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
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `"+winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

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

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    im = load_image(image, 0, 0)
    if debug: print("Loaded image")
    ret = detect_image(net, meta, im, thresh, hier_thresh, nms, debug)
    free_image(im)
    if debug: print("freed image")
    return ret

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    #import cv2
    #custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
    #custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    #custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    #import scipy.misc
    #custom_image = scipy.misc.imread(image)
    #im, arr = array_to_image(custom_image)		# you should comment line below: free_image(im)
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    letter_box = 0
    #predict_image_letterbox(net, im)
    #letter_box = 1
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res

def performDetect(imagePath='./image_0/{0}_{1}.png'.format(b.tm_hour,b.tm_min), thresh= 0.25, configPath = "./cfg/yolov4_custom_test.cfg", weightPath = "./yolov4_custom_train_last.weights",metaPath= "./cfg/obj.data", showImage= True, makeImageOnly = False, initOnly= False):
    b = time.localtime()
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if initOnly:
        print("Initialized detector")
        return None
    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `"+os.path.abspath(imagePath)+"`")
    # Do the detection
    #detections = detect(netMain, metaMain, imagePath, thresh)	# if is used cv2.imread(image)
    detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    if showImage:
        try:
            from skimage import io, draw
            import numpy as np
            image = io.imread(imagePath)
            print("*** "+str(len(detections))+" Results, color coded by confidence ***")
            imcaption = []
            for detection in detections:
                label = detection[0]
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape
                # x = shape[1]
                # xExtent = int(x * bounds[2] / 100)
                # y = shape[0]
                # yExtent = int(y * bounds[3] / 100)
                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2]/2)
                yCoord = int(bounds[1] - bounds[3]/2)
                boundingBox = [
                    [xCoord, yCoord],
                    [xCoord, yCoord + yExtent],
                    [xCoord + xEntent, yCoord + yExtent],
                    [xCoord + xEntent, yCoord]
                ]
                # Wiggle it around to make a 3px border
                rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
                rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
                boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
                draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
                draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
                draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
                draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
            if not makeImageOnly:
                io.imshow(image)
                io.show()
            detections = {
                "detections": detections,
                "image": image,
                "caption": "\n<br/>".join(imcaption)
            }
        except Exception as e:
            print("Unable to show image: "+str(e))
    return detections

# !python darknet_video -v -c -w 
# augument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v" ,"--video", type=str, required=True, 
                help="Path to input video")
ap.add_argument("-c","--config", default="./config.json",
               help="Path to yolo config file")
ap.add_argument("-w","--weights",type=str, required=True,
                help="Path to yolo weight")
ap.add_argument("-l","--label",type=str, default="./data/classes.names",
                help="Path to label file")
ap.add_argument("-m","--meta",type=str, default="./data/yolov4.data",
                help="Path to metaPath")
ap.add_argument("-o","--output",type=str, default="./output.mp4",
                help="Path to output file")

args = vars(ap.parse_args())

def check_argument(args):
    assert os.path.isfile(args["video"]) == True, "Can't find " + args["video"]
    assert os.path.isfile(args["config"]) == True, "Can't find " + args["config"]
    assert os.path.isfile(args["weights"]) == True, "Can't find " + args["weights"]
    assert os.path.isfile(args["meta"]) == True, "Can't find " + args["meta"]
    assert os.path.isfile(args["label"]) == True, "Can't find " + args["label"]

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        color = [int(c) for c in COLORS[LABELS.index(detection[0].decode())]]
        #print(color, type(color))
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
    return img

def parkedcars(detections,img,B):
    Alpah = []
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt3 = ((xmin + xmax)/2 , (ymin + ymax)/2)
        Alpah = [pt3[0],pt3[1],detection[0].decode() ,
                     str(round(detection[1] * 100, 2))]
        C = [pt3[0],pt3[1]]
        if Alpah[2] == 'car' or Alpah[2] == 'truck':
                B.append(C)
                print(C)
    return img

def performBatchDetect(thresh= 0.25,imagePath ='./image_0/{0}_{1}.png'.format(b.tm_hour,b.tm_min), configPath = "./cfg/yolov4_custom_test.cfg", weightPath = "./yolov4_custom_train_last.weights",metaPath= "./cfg/obj.data", hier_thresh=.5, nms=.45, batch_size=3):
    import cv2
    import numpy as np
    b = time.localtime()
    # NB! Image sizes should be the same
    # You can change the images, yet, be sure that they have the same width and height
    img_samples = ['./image_0/{0}_{1}.png'.format(b.tm_hour,b.tm_min), './image_0/{0}_{1}.png'.format(b.tm_hour,b.tm_min), './image_0/{0}_{1}.png'.format(b.tm_hour,b.tm_min)]
    image_list = [cv2.imread(k) for k in img_samples]

    net = load_net_custom(configPath.encode('utf-8'), weightPath.encode('utf-8'), 0, batch_size)
    meta = load_meta(metaPath.encode('utf-8'))
    pred_height, pred_width, c = image_list[0].shape
    net_width, net_height = (network_width(net), network_height(net))
    img_list = []
    for custom_image_bgr in image_list:
        custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
        custom_image = cv2.resize(
            custom_image, (net_width, net_height), interpolation=cv2.INTER_NEAREST)
        custom_image = custom_image.transpose(2, 0, 1)
        img_list.append(custom_image)

    arr = np.concatenate(img_list, axis=0)
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(net_width, net_height, c, data)

    batch_dets = network_predict_batch(net, im, batch_size, pred_width,
                                                pred_height, thresh, hier_thresh, None, 0, 0)
    batch_boxes = []
    batch_scores = []
    batch_classes = []
    for b in range(batch_size):
        num = batch_dets[b].num
        dets = batch_dets[b].dets
        if nms:
            do_nms_obj(dets, num, meta.classes, nms)
        boxes = []
        scores = []
        classes = []
        for i in range(num):
            det = dets[i]
            score = -1
            label = None
            for c in range(det.classes):
                p = det.prob[c]
                if p > score:
                    score = p
                    label = c
            if score > thresh:
                box = det.bbox
                left, top, right, bottom = map(int,(box.x - box.w / 2, box.y - box.h / 2,
                                            box.x + box.w / 2, box.y + box.h / 2))
                boxes.append((top, left, bottom, right))
                scores.append(score)
                classes.append(label)
                boxColor = (int(255 * (1 - (score ** 2))), int(255 * (score ** 2)), 0)
                cv2.rectangle(image_list[b], (left, top),
                          (right, bottom), boxColor, 2)
        cv2.imwrite(os.path.basename(img_samples[b]),image_list[b])

        batch_boxes.append(boxes)
        batch_scores.append(scores)
        batch_classes.append(classes)
    free_batch_detections(batch_dets, batch_size)
    return batch_boxes, batch_scores, batch_classes 

netMain = None
metaMain = None
altNames = None
   


def YOLO():
    B = []
    int_time = time.time()

    global metaMain, netMain, altNames, COLORS, LABELS
    
    videoPath = args["video"]
    configPath = args["config"]
    weightPath = args["weights"]
    metaPath = args["meta"]
    labelsPath = args["label"]
    outputPath = args ["output"]
    
    check_argument(args)
    
    LABELS = open(labelsPath).read().strip().split("\n")
    #print(LABELS, len(LABELS))
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(videoPath)
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "{}".format(outputPath), cv2.VideoWriter_fourcc(*"DIVX"), 24.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("[INFO] Start the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    print("[INFO] Start processing video...")                                    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()

        if not ret:
          break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = parkedcars(detections, frame_resized,B)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if time.time() - int_time > 300:
            b = time.localtime()
            print("5min passed")
            np.savetxt('./txt/{0}_{1}.txt'.format(b.tm_hour,b.tm_min),B,fmt='%s')
            A = np.genfromtxt('./txt/{0}_{1}.txt'.format(b.tm_hour,b.tm_min),dtype='str', delimiter = ' ')
            Bb = A[:,0:2].astype(np.float32)
            C = pd.DataFrame(Bb)
            Bbb = C[0] * 10000 + C[1]
            E = pd.DataFrame(Bbb)
            D = E[0].value_counts()
            G = np.savetxt('./txt/{0}_{1}_counts.txt'.format(b.tm_hour,b.tm_min),D,fmt='%i')
            F = np.genfromtxt('./txt/{0}_{1}_counts.txt'.format(b.tm_hour,b.tm_min),dtype='int', delimiter = ' ')
            if np.any(F > 825):
                prv_time = time.time()
                b = time.localtime()
                aa = plt.scatter(Bb[:,0],Bb[:,1],s = 1)
                aa = plt.axis('off')
                plt.savefig('./image_0/{0}_{1}.png'.format(b.tm_hour,b.tm_min))
                print( time.time() - prv_time)
                src = cv2.imread('./image_0/{0}_{1}.png'.format(b.tm_hour,b.tm_min))
                plt.cla()
                print('Parked Car exists')
                performBatchDetect()
                B = []
                int_time = time.time()

        #print(1/(time.time()-prev_time))
        out.write(image)
        #cv2.imshow('Demo', image)
        cv2.waitKey(3)
    cap.release()
    out.release()
    print('[INFO] Save processed video as "{}"'.format(outputPath))                                    
if __name__ == "__main__":
    YOLO()
