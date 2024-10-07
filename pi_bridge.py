#!/usr/bin/env python
import sys
import tos
import numpy as np

class gridData(tos.Packet):
    def __init__(self, payload = None):
        tos.Packet.__init__(self,
              [('data',  'blob', None)],
                               payload)
if len(sys.argv) < 2:
    print("Usage:", sys.argv[0], "serial@/dev/ttyUSB0:115200")
    sys.exit()


def handleSerial(am, bg):
   p = am.read()
   if p:
      grid = gridData(p.data)

      node = (grid.data[0] & 0x7f)
      pir = ((grid.data[0] & 0x80) != 0)
      pir = bool(pir);

      print("===========");
      print("Node ID: " + str(node) + " PIR: " + str(pir))
      snap = np.asarray(grid.data[1:65], dtype=np.float64).reshape((8,8))/4
      print (snap)
      return (snap, pir)


def main():
    #Start Reading from Serial
    am = tos.AM()
    bg = np.zeros((8,8))
    (bg, pir) = handleSerial(am, bg);

    PIR_THRESHOLD = 6;
    COLOR_THRESHOLD = 1;
    SENS = 2
    pirCounter = 0;
    ALPHA = 0.25

    while True:
        try:
            (new_bg, pir) = handleSerial(am, bg);
            if not pir:
                print('Empty')
                pirCounter +=1;
            else:
                pirCounter =0;
            # Add background values
            if new_bg is not None and pirCounter >= PIR_THRESHOLD:
                # Blend it together
                bg = ((1-ALPHA)*bg + ALPHA*new_bg)/2

            # Background subtraction
            if new_bg is not None:
                truth = np.abs(bg - new_bg)
                truth  = np.greater(truth, np.ones((8,8))*SENS)
                print(truth)
        except Exception as e:
            print (e)



if __name__ == '__main__':
    main()
