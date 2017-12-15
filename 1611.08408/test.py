from reader import *

Reader = DataReader(data_root='/media/Disk/wangfuyu/data/cxr/801/',
                    txt='/media/Disk/wangfuyu/data/cxr/801/trainJM_id.txt')

while(1):
    a, b, c, d = Reader.next()
    print a.shape, b.shape, c.shape, d.shape