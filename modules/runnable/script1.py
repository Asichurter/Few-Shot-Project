# 本脚本用于清洗微软的数据集，从bytes文件中提取二进制数据

import os
from functools import reduce
import threading

path = 'C:/train/'
dest_path = 'C:/microsoftPE/train/'

file_index_lock = threading.Lock()
file_index = 0

def check_0size_file(ID, delete=False):
    for f in os.listdir(dest_path):
        if os.path.getsize(dest_path+f) == 0:
            print(f)
            if delete:
                os.remove(dest_path+f)

def strl_2_bytes(line):

    n_line = line[9:]   # filter the sequence number
    n_line = n_line.split(' ') # split the bytes
    # print(n_line)
    try:
        n_line = list(map(lambda x: int(x, 16), n_line))
        # print(n_line)
    except ValueError:
        # print(line)
        return bytearray()
    return bytearray(n_line)

def convert(ID):
    files = os.listdir(path)
    file_index_lock.acquire()
    global file_index
    while file_index < len(files):
        this_index = file_index
        file_index += 1
        file_index_lock.release()

        full_name = files[this_index]
        f_name = full_name.split('.')[0]
        print('thread %d, %d / %d %s' % (ID, this_index + 1, len(os.listdir(path)) + 1, full_name))
        if not os.path.exists(dest_path + f_name + '.pe') or \
                os.path.getsize(dest_path + f_name + '.pe') == 0:  # 支持中断后继续
            with open(path + full_name, 'r') as rf, open(dest_path + f_name + '.pe', 'wb') as wf:
                lines = rf.readlines()
                lines = list(map(strl_2_bytes, lines))
                lines = reduce(lambda x, y: x + y, lines)

                wf.write(lines)

        file_index_lock.acquire()

    file_index_lock.release()
    print('Thraed %s, exit!' % (ID))

class ConvertThread(threading.Thread):
    def __init__(self, ID, name, function, **kwargs):
        super(ConvertThread, self).__init__()
        self.ID = ID
        self.Name = name
        self.F = function
        self.kwargs = kwargs

    def run(self):
        self.F(self.ID, **self.kwargs)

if __name__ == '__main__':
    thread_num = 1
    threads = []

    for i in range(thread_num):
        threads.append(ConvertThread(i+1, 'Thread_%d'%(i+1), check_0size_file, delete=False))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print('Done!')


        # for i, full_name in enumerate(os.listdir(path)):




#
# with open(file_1,'r') as f, open(write_path, 'wb') as wf:
#     print(file_1)
#     lines = f.readlines()#[f.readline(500)]
#     lines = list(map(strl_2_bytes, lines))
#     lines = reduce(lambda x,y:x+y, lines)
#     # print(lines)
#     print('writing...')
#     wf.write(lines)




