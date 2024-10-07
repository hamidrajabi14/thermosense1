import tos


class gridData(tos.Packet):
    def __init__(self, payload=None):
        tos.Packet.__init__(self, [('data',  'blob', None)], payload)


def main():
    am = tos.AM()
    while True:
        p = am.read()
        print(f"{p}\n")
        # print(type(p))

if __name__ == '__main__':
    main()
