def write_config(fname, config):
    lines = ["{}:{}\n".format(key, val) for key, val in config.items()]

    file = open(fname, "w")
    file.writelines(lines)
    file.close()


def read_config(fname):
    # read all lines from file
    file = open(fname, "r+")
    lines = file.readlines()
    file.close()

    config = {}

    # interpret
    for line in lines:
        line = line.split(":")
        key = line[0].strip()  # strip to remove any whitespace
        val = line[1].strip()

        if key == "cam_order" or key == "active_DEAs":
            val = val[1: -1]
            val = [int(s) for s in val.split(", ")]
        elif val == "True":
            val = True
        elif val == "False":
            val = False
        else:
            try:  # try to convert to numeric value
                val = int(val)
            except ValueError:
                pass  # not a number  -> just keep it as a string value

        config[key] = val

    return config
