import os

# 输出字符串指示正在使用的平台。window 则用'nt'表示，对于Linux/Unix用户，它是'posix'。
print(os.name)
# 函数得到当前工作目录，即当前Python脚本工作的目录路径。
print(os.getcwd())
# 返回指定目录下的所有文件和目录名
print(os.listdir(os.getcwd()))

print(os.path.split(r"D:\csi_nexmon\os_demo.py"))
# os.path.isfile()和os.path.isdir()函数分别检验给出的路径是一个文件还是目录。
print(os.path.isdir(os.getcwd()))
print(os.path.isfile(os.path.split(r"D:\csi_nexmon\os_demo.py")[1]))
# os.path.exists()函数用来检验给出的路径是否真的存在
print(os.path.exists('D:\\csi_nexmon\\abc.txt'))
print(os.path.exists('D:\\csi_nexmon\\demo.py'))
# os.path.abspath(name): 获得绝对路径
print(os.path.abspath("./matlab/example.pcap"))
# os.path.getsize(name): 获得文件大小，如果name是目录返回0L，如果name为文件，则返回文件的字节数
print(os.path.getsize("./matlab/example.pcap"))
# os.path.splitext(): 分离文件名与扩展名
print(os.path.splitext("example.pcap"))
# os.path.join(path,name): 连接目录与文件名或目录
print(os.path.join('D:\\csi_nexmon', 'sys_usage_demo.py'))
# os.path.basename(path): 返回文件名
print(os.path.basename(r"D:\csi_nexmon\sys_usage_demo.py"))
# os.path.dirname(path): 返回文件路径
print(os.path.dirname(r"D:\csi_nexmon\sys_usage_demo.py"))
