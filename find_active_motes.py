#!/usr/bin/env python
import numpy as np
import time
import sys
import tos

STALE_THRESHOLD = 60    # We kick a mote after 60 seconds

class gridData(tos.Packet):
    def __init__(self, payload = None):
        tos.Packet.__init__(self,
              [('data',  'blob', None)],
                               payload)
if len(sys.argv) < 2:
    print("Usage:", sys.argv[0], "serial@/dev/ttyUSB0:115200")
    sys.exit()


def handleSerial(am):
   p = am.read()
   if p:
      grid = gridData(p.data)

      node = (grid.data[0] & 0x7f)

      return node


def print_active_nodes(active_nodes):
    # Sort list with respect to node id
    active_nodes.sort(key=lambda tup: tup[0])

    for node in active_nodes:
        print ("{}\t".format(node[0]).expandtabs(3))

    # Provide our newline
    print ("")

def main():
    #Start Reading from Serial
    am = tos.AM()
    active_nodes = []

    node = handleSerial(am);

    while True:
        try:
            node = handleSerial(am);

            # Try to locate id in our list, return None otherwise
            node_idx = next((idx for (idx, id) in enumerate(active_nodes) if id[0] == node), None)

            # If this node is in our history, remove it
            if node_idx is not None:
                active_nodes.pop(node_idx)

            # In either case, we add this node with fresh timestamp to our history
            active_nodes.append((node, time.time()))

            # Step through all nodes, evict any who hasn't been seen in STALE_THRESHOLD seconds
            for idx, (node, update_time) in reversed(list(enumerate(active_nodes))):
                if time.time() > update_time + STALE_THRESHOLD:
                    active_nodes.pop(idx)


            print ("{} active motes:\t\t".format(len(active_nodes)))
            print_active_nodes(active_nodes)

        except Exception as e:
            print (e)



if __name__ == '__main__':
    main()
