import time

def sleeptime(hour, min, sec):
    return hour * 3600 + min * 60 + sec

def readMemInfo():
    res = {'total': 0, 'free': 0, 'buffers': 0, 'cached': 0}
    f = open('/proc/meminfo')
    lines = f.readlines()
    f.close()
    i = 0
    for line in lines:
        if i == 4:
            break
        line = line.lstrip()
        memItem = line.lower().split()
        if memItem[0] == 'memtotal:':
            res['total'] = int(memItem[1])
            i = i + 1
            continue
        elif memItem[0] == 'memfree:':
            res['free'] = int(memItem[1])
            i = i + 1
            continue
        elif memItem[0] == 'buffers:':
            res['buffers'] = int(memItem[1])
            i = i + 1
            continue
        elif memItem[0] == 'cached:':
            res['cached'] = int(memItem[1])
            i = i + 1
            continue
    return res

def calcMemUsage(counters):
    used = counters['total'] - counters['free'] - counters['buffers'] - counters['cached']
    total = counters['total']
    return used * 100 / total

if __name__ == '__main__':
    second = sleeptime(0, 5, 0)
    while True:
        time.sleep(second)
        counters = readMemInfo()
        print(calcMemUsage(counters))

