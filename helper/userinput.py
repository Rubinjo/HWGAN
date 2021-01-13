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
        return False, 'none'

def getDataAndText(args):
        args = args[1:]
        textParsing = False
        expectingData = False
        data = 'emnist'
        text = ""
        for arg in args:
                isarg, sort = getArg(arg)
                if isarg:
                        if sort == 'data':
                                expectingData = True
                                textParsing = False
                                continue
                        elif sort == "text":
                                textParsing = True
                                expectingData = False
                                continue
                if expectingData:
                        data = arg
                        expectingData = False
                        continue
                if textParsing:
                        text += arg
                        text += " "
        if text != "":
                text = text[:len(text) - 1]
        return data, text