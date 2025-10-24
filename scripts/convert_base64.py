#* 功能说明import base64
import base64
import tarfile
import argparse
import shutil
import io
from io import StringIO
from pathlib import Path
import re

#* python convert_base64.py -c ./dir_to_tar ./base64.txt
#* python convert_base64.py -z ./dir_to_extract ./base64.txt

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--create', nargs=2, help='convert file to base64')
parser.add_argument('-z', '--zip', nargs=2, help='convert base64 to file')

str_tag_start = '-----BEGIN PGP MESSAGE-----'
str_tag_end = '-----END PGP MESSAGE-----'
def dir_to_tar(input_file_path):
    #生成一个在内存中的文件对象，并返回
    input_file = io.BytesIO()

    tar = tarfile.open(fileobj=input_file, mode=f'w:gz')
    print("\033[32m压缩内容中\033[0m")
    tar.add(input_file_path)
    tar.close()
    print("\033[32m压缩完毕\033[0m")

    return input_file

def echo_to_terminal(text):
    width_line = 64
    print(str_tag_start)
    escape = lambda x: re.sub(r"^b'|'$", '', x)

    for i in range(0, len(text), width_line):
        print(escape(str(text[i:i+width_line])))

    print(str_tag_end)

def gather_from_log(str_encoded):
    ret_1 = re.search(str_tag_start, str_encoded)
    ret_2 = re.search(str_tag_end, str_encoded)
    str_marked = str_encoded[ret_1.end()+1: ret_2.start()-1]

    #* 解码
    str_marked_decoded = base64.b64decode(str_marked)
    return str_marked_decoded

# 使用示例
def tar_to_dir(str_bytes, output_file_path):
    #* 解压文件
    input_file = io.BytesIO(str_bytes)
    tar = tarfile.open(fileobj=input_file, mode='r:gz')
    tar.extractall(output_file_path)

if __name__ == '__main__':
    args = parser.parse_args()
    # test()

    if (args.create is None) ^ (args.zip is None):
        # print(args)

        if args.create is not None:

            file_tar = dir_to_tar(args.create[1])
            str_tar = base64.b64encode(file_tar.getvalue())
            Path(args.create[0]).write_bytes(str_tar)
            echo_to_terminal(str_tar)

        if args.zip is not None:

            str_zip = Path(args.zip[0]).read_text(encoding='utf-16')
            str_zip = gather_from_log(str_zip)
            tar_to_dir(str_zip, args.zip[1])