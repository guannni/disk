import os

# path='D:\\guan2019\\1_disk\\1.2_pics\\AccFixedFreDiff\\1_3\\70_0\\' #'D:\\guan2019\\1_disk\\1.2_pics\\AccFixedFreDiff\\1_4\\585\\585_1\\'

path = 'D:\\guan2019\\1_disk\\1.2_pics\\PeakFixedFreDiff\\1_1\\50_1\\'# ''D:\\guan2019\\2_ball\\1_pic\\120Hz\\6.0\\6_0\\'#'D:\\guan2019\\2_ball\\1_pic\\60Hz\\3.5\\3.5\\'
#获取该目录下所有文件，存入列表中
f =os.listdir(path)#

n=0

for i in f:
    # print(os.path.splitext(i)[0],f[n])
    if int(os.path.splitext(i)[0]) <10:
        oldname = path + f[n]
        print(oldname)
        newname = path + '00000' + str(os.path.splitext(i)[0]) + '.jpg'
        print(str(os.path.splitext(i)[0]))
        os.rename(oldname, newname)
    elif int(os.path.splitext(i)[0]) < 100 and int(os.path.splitext(i)[0]) >= 10:
        oldname = path + f[n]
        newname = path + '0000' + str(os.path.splitext(i)[0]) + '.jpg'
        print(newname)
        os.rename(oldname, newname)
    elif int(os.path.splitext(i)[0]) < 1000 and int(os.path.splitext(i)[0]) >= 100:
        oldname = path + f[n]
        newname = path + '000' + str(os.path.splitext(i)[0]) + '.jpg'
        print(newname)
        os.rename(oldname, newname)
    elif int(os.path.splitext(i)[0]) < 10000 and int(os.path.splitext(i)[0]) >= 1000:
        oldname = path + f[n]
        newname = path + '00' + str(os.path.splitext(i)[0]) + '.jpg'
        print(newname)
        os.rename(oldname, newname)
    elif int(os.path.splitext(i)[0]) < 100000 and int(os.path.splitext(i)[0]) >= 10000:
        oldname = path + f[n]
        newname = path + '0' + str(os.path.splitext(i)[0]) + '.jpg'
        print(newname)
        os.rename(oldname, newname)

    n+=1
