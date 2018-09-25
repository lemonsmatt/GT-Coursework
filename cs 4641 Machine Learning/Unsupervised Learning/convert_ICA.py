if __name__ == '__main__':
    
    dataset = "winequality-white"
    f = open(dataset + "/data.arff", "r")
    data = []
    start = False
    for line in f:
        if start:
            data.append("'\\'" + line.split("'\\'")[1])
        if "@data" in line:
            start = True
            print(line)
    for i in range(0,10):
        print(data[i])
    for filename in ["/ica_1.0E-2.arff", "/ica_1.0E-4.arff", "/ica_1.0E-7.arff"]:
        f = open(dataset + filename, "r")
        start = False
        lines = []
        i = 0
        while not start:
            line = f.readline()
            if "@data" in line:
                start = True
            lines.append(line)
        for line, val in zip(f, data):
            lines.append(line.split("\n")[0] + "," + val)

        #f = open(dataset + filename, "w")
        #for line in lines:
            #f.write(line)


    dataset = "magic04"
    f = open(dataset + "/data.arff", "r")
    data = []
    start = False
    for line in f:
        if start:
            data.append(line.split(",")[-1])
        if "@data" in line:
            start = True
            print(line)
    for i in range(0,10):
        print(data[i])
    for filename in ["/ica_1.0E-2.arff", "/ica_1.0E-4.arff", "/ica_1.0E-7.arff"]:
        f = open(dataset + filename, "r")
        start = False
        lines = []
        i = 0
        while not start:
            line = f.readline()
            if "@data" in line:
                start = True
            lines.append(line)
        for line, val in zip(f, data):
            lines.append(line.split("\n")[0] + "," + val)

        f = open(dataset + filename, "w")
        for line in lines:
            f.write(line)

