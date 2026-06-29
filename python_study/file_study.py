# 创建文件
file = open('C:\\Users\\NLDS\\Desktop\\1.txt', 'w')
file.write('hello world')
file.close()

file = open('C:\\Users\\NLDS\\Desktop\\1.txt', 'r')
file_content = file.readlines()
file.close()
print(file_content)