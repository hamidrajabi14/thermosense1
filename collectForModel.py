import cv2
import numpy as np
import threading
import time
import tos
import sys

LINE_WIDTH = 1;
LINE_COLOR = (255,0,0)
overlay = np.zeros((8,8))

lock = threading.Lock();

class gridData(tos.Packet):
    def __init__(self, payload = None):
        tos.Packet.__init__(self,
              [('data',  'blob', None)],
                               payload)
if len(sys.argv) < 2:
    print("Usage:", sys.argv[0], "serial@/dev/ttyUSB0:115200")
    sys.exit()


def handleSerial(am, bg):
   while True:
       p = am.read()
       if p:
          grid = gridData(p.data)

          node = (grid.data[0] & 0x7f)
          pir = ((grid.data[0] & 0x80) != 0)
          pir = bool(pir);

          if node != 15:
              continue;
          if len(grid.data) < 64:
             #print "Not for us"
             continue;

          #print("===========");
          #print("Node ID: " + str(node) + " PIR: " + str(pir) + ", len " + str(len(grid.data)))
          snap = np.asarray(grid.data[1:65], dtype=np.float64).reshape((8,8))/4
          #print "Made snap";
          #print snap
          return (snap, pir)


def readLoop():
    #Start Reading from Serial
    am = tos.AM()
    bg = np.zeros((8,8))
    (bg, pir) = handleSerial(am, bg);
    #print "Handled";

    # About 10 seconds
    PIR_THRESHOLD = 10;
    COLOR_THRESHOLD = 1;

    ## ALEXIS ##
    # Increase to reduce noise - range 1-3
    SENS = 1.5
    pirCounter = 0;
    ALPHA = 0.85
    global overlay

    while True:
        try:
            (new_bg, pir) = handleSerial(am, bg);
            #print "Handled";
            if not pir:
                pirCounter +=1;
            else:
                pirCounter = 0;
            # Add background values
            if new_bg is not None and pirCounter >= PIR_THRESHOLD:
                # Blend it together
                #print('Updating Background')
                bg = ((1-ALPHA)*bg + ALPHA*new_bg)
            else:
                #print('Need %d more idle snaps'%(PIR_THRESHOLD -pirCounter));
                pass;

            # Background subtraction
            lock.acquire();
            if new_bg is not None:
                overlay = np.abs(bg - new_bg)
                #print(overlay)
                overlay = np.fliplr(np.flipud(np.greater(overlay, np.ones((8,8))*SENS)))
                print " ".join([str(x) for x in overlay.flatten().astype(int)]);
                #print(overlay)
            lock.release();
        except Exception, e:
            print e.message, e.args
            #print e.msg

def fill(frame, x, y):
    block_width = np.size(frame, 1)/8
    block_height = np.size(frame, 0)/8

    frame[(y * block_height):((y+1)*block_height),
        (x * block_width):((x+1)*block_width)] = (0, 0, 255)

    return frame;

def cameraLoop():
    name = "Thermosense"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    camera_index = 0
    capture = cv2.VideoCapture(camera_index)
    global overlay

    while True:
        ret, frame = capture.read()
        # cv2.imshow(name, frame)
#
#         c = cv2.waitKey(1)
#         if(c=="e"):
#             exit()
#         continue;
        width = np.size(frame, 1)
        height = np.size(frame, 0)

        gridWidth = width/8;
        gridHeight = height/8;

        final = frame
        blank_image = np.zeros((height, width, 3), np.uint8)
        blank_image[:] = (0, 0, 0)
        #print "Drawing to Camera!"

        oSum = 0;
        lock.acquire();
        for x,row in enumerate(overlay):
            for y,col in enumerate(row):
                if col:
                    blank_image = fill(blank_image, y, x)
                    oSum = oSum + 1;
        lock.release();

		# Drawing lines on the image
        for row in range(1, 8):
            # Rows
            h = row*gridHeight
            cv2.line(frame,(0,h),(width,h) ,LINE_COLOR, LINE_WIDTH)

            # Columns
            w = row*gridWidth
            cv2.line(frame,(w,0),(w,height), LINE_COLOR, LINE_WIDTH)


        final = cv2.addWeighted(frame, 1, blank_image, 0.3, 0)
        #cv2.imshow(name, cv2.resize(final, (min(width, height), min(width,height))))
        cv2.imshow(name, final)

        c = cv2.waitKey(1)
        if(c=="e"):
            exit()

def main():
    # Camera Stuff
    #cameraThread = threading.Thread(name="Pizza", target=cameraLoop)
    #cameraThread.setDaemon(True)
    #cameraThread.start()

    # Grid Eye Stuff
    gridThread = threading.Thread(name="Pizza2", target=readLoop)
    gridThread.setDaemon(True)
    gridThread.start()

    # Current version of opencv doesn't allow camera handling to be off the main thread
    cameraLoop();

    while True:
        time.sleep(10)

if __name__ == '__main__':
    main()
