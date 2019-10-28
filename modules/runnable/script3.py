# GIST特征
import os

def convert_img_to_gist_file():
    s_path = 'D:/peimages/New/cluster/train/'
    d_path = 'D:/peimages/New/cluster/gist/train/'
    gist_exe_path = 'D:/DL/GIST-global-Image-Descripor-master/GIST-global-Image-Descripor-master/gist-tool/gist.exe'
    shell_temp = '%s -i %s -o %s'

    for s_dir in os.listdir(s_path):
        print(s_dir)
        dir_path = d_path+s_dir+'/'
        os.mkdir(dir_path)
        os.system(shell_temp%(gist_exe_path,s_path+s_dir,dir_path))

def cumulate_gist_data():
    s_path = 'D:/peimages/New/cluster/gist/train/'
    d_path = 'D:/peimages/New/cluster/gist/train.npy'

    # for c in os.listdir()

if __name__ == '__main__':
    # convert_img_to_gist_file()
    s_path = 'D:/peimages/New/cluster/gist/train/'
    for c in os.listdir(s_path):
        with open(s_path+c+'/gist.txt', 'r') as f:
            lines = f.readlines()
            print(c,len(lines[0]))

