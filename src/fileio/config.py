def write_config(fname, com_port, cam_order):
    file = open(fname, "w")

    file.write(com_port + "\n")
    file.write("{}".format(cam_order))

    file.close()


def read_config(fname):
    file = open(fname, "r+")

    com_str = file.readline()
    cam_str = file.readline()

    file.close()

    com_port = com_str.strip()  # remove newline character at the end

    cam_str = cam_str[1: -1]
    cam_order = [int(s) for s in cam_str.split(", ")]

    return com_port, cam_order
