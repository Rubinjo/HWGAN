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
                if identifier == 'user':
                        return True, 'user'
                elif identifier == 'text':
                        return True, 'text'
        return False, 'none'

def getUserAndText(args):
        args = args[1:]
        textParsing = False
        expectingUser = False
        user = 'emnist'
        text = ""
        for arg in args:
                isarg, sort = getArg(arg)
                if isarg:
                        if sort == 'user':
                                expectingUser = True
                                textParsing = False
                                continue
                        elif sort == "text":
                                textParsing = True
                                expectingUser = False
                                continue
                if expectingUser:
                        user = arg
                        expectingUser = False
                        continue
                if textParsing:
                        text += arg
                        text += " "
        if text != "":
                text = text[:len(text) - 1]
        return user, text