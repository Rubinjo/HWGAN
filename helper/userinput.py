def cmd_in(args):
    out = ""
    words = args[1:]
    for i in range(len(words)):
        out += words[i]
        if i != len(words) - 1:
            out += " "
    return out

def getArg(arg):
    if arg[0] == '-':
        identifier = arg[1:]
        if identifier == 'data':
            return True, 'data'
        elif identifier == 'text':
            return True, 'text'
        elif identifier == 'sample':
            return True, 'sample'
        elif identifier == 'ocr':
            return True, 'ocr'
    return False, 'none'

def getDataAndText(args):
    args = args[1:]
    # Establish default values
    data = "emnist"
    sample = 0
    splitText = "chars"
    ocr = False
    # Loop through given arguments
    for idx, arg in enumerate(args):
        # Check if argument corrosponds to an expected argument
        isarg, sort = getArg(arg)
        if isarg:
            if sort == 'data':
                try:
                    data = args[idx + 1]
                except:
                    continue
            elif sort == 'sample':
                try:
                    sample = int(args[idx + 1])
                except:
                    continue
            elif sort == "text":
                try:
                    splitText = args[idx + 1]
                except:
                    continue
            elif sort == 'ocr':
                try:
                    ocr = args[idx + 1]
                except:
                    continue
    return data, sample, splitText, ocr