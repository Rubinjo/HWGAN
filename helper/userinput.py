def cmd_in(args):
    out = ""
    words = args[1:]
    for i in range(len(words)):
        out += words[i]
        if i != len(words) - 1:
            out += " "
    return out
