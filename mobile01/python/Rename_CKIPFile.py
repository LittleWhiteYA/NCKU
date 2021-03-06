import os
import sys

def special_handle(str1):
    str1 = ''.join(str1.split('?'))
    str1 = ''.join(str1.split('\\'))
    str1 = ''.join(str1.split('/'))
    str1 = ''.join(str1.split(':'))
    str1 = ''.join(str1.split('|'))
    str1 = ''.join(str1.split('<'))
    str1 = ''.join(str1.split('>'))
    str1 = ''.join(str1.split('"'))
    str1 = ''.join(str1.split('*'))
    str1 = ''.join(str1.split('\n'))
    str1 = ''.join(str1.split('\x1b')) #Esc
    str1 = ''.join(str1.split('\t'))
    str1 = ''.join(str1.split('\b'))
    str1 = ''.join(str1.split('\x0b'))
    str1 = ''.join(str1.split('\x1c'))
    str1 = ''.join(str1.split('\r'))
    return str1
type1 = sys.argv[1]
path = '/home/lu/mobile01/data/'+type1+'_data/'
path2 = '/home/lu/mobile01/data/tmp_'+type1+'_data/'
data_dir = os.listdir(path)


for filename in data_dir:
    if filename[0] == '_':
        continue

    file = open(path+filename, 'rb')
    origin_url_len = int.from_bytes(file.read(8), byteorder='big')
    topic_len = int.from_bytes(file.read(8), byteorder='big')
    start_url_len = int.from_bytes(file.read(8), byteorder='big')
    
    origin_url = file.read(origin_url_len).decode('utf8')
    topic = file.read(topic_len).decode('utf8')
    start_url = file.read(start_url_len).decode('utf8')
    file.close()
    if topic == '':
        print(filename)
        print("no topic!")
        print("~~~~~~~~~~~~~~~~~~~~")
        continue
    elif start_url.rfind('&p=') == -1:
        page = start_url
        new_name = path2+special_handle(topic)+".txt"
    else:
        page = start_url[start_url.rfind('&p=')+3:]
        new_name = path2+special_handle(topic)+"_"+page+".txt"
    print(topic)
    print(start_url)
    print(new_name)
    
    isfile = os.path.isfile(new_name)
    if isfile:
        print("File exist! "+new_name)
    else:
        os.rename(path+filename, new_name)
    
    #if topic == '':
    #    os.remove(new_name)
    print("=================")
    #break




