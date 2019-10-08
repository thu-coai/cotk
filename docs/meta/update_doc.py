import os
import argparse
import sys

def get_location(text):
    lines = text.split("\n")
    res = []
    for line in lines:
        if not line.startswith("~~ location"):
            break
        _, _, key, path = line.split()
        res.append((key, path))
    return res

def render(text, key):
    lines = text.split("\n")
    res = []

    in_flag = True
    for line in lines:
        if not line.startswith("~~"):
            if in_flag:
                res.append(line)
            continue
        args = line.split()
        if args[1] == "location":
            continue
        elif args[1] == "contentstart":
            if args[2] != key:
                in_flag = False
        elif args[1] == "contentend":
            if args[2] != key:
                in_flag = True
        elif args[1] == "include":
            res.extend(render(open(args[2], 'r', encoding='utf-8').read(), key))
        else:
            raise RuntimeError("Unknown tags %s" % args[1])
    return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--check', action='store_true', dest='check')
    cargs = parser.parse_args()

    for filename in os.listdir("./"):
        if not filename.endswith(".md") and not filename.endswith(".rst"):
            continue
        print(filename)
        file = open(filename, "r", encoding='utf-8').read()

        locations = get_location(file)
        for key, path in locations:
            text = render(file, key)
            if cargs.check:
                realtext = open(path, 'r', encoding='utf-8').read().split("\n")
                if realtext[-1] != "" or len(text) + 1 != len(realtext):
                    print("Line number")
                    print("\t%s: %d" % (path, len(text)))
                    print("\t%s: %d" % (filename, len(realtext) - 1))
                    raise ValueError("It seems docs [%s] is not synced with metadocs [%s]. Please change metadocs and then run update_docs !" % (path, filename))
                for i, (t, rt) in enumerate(zip(text, realtext)):
                    if t != rt:
                        print("At Line %d:" % i)
                        print("\t%s: %s" % (path, t))
                        print("\t%s: %s" % (filename, rt))
                        raise ValueError("It seems docs [%s] is not synced with metadocs [%s]. Please change metadocs and then run update_docs !" % (path, filename))
            else:
                open(path, 'w', encoding='utf-8').write("\n".join(text) + "\n")
                print("render %s with %s" % (path, key))
