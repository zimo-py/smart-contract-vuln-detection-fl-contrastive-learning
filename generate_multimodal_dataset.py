import io
import re

import math
import pandas as pd
import numpy as np
from PIL import Image
from hexbytes import HexBytes

def remove_comment(source_code):
    source_code = re.sub(r"//\*.*", "", source_code)
    source_code = re.sub(r"#.*", "", source_code)
    # Remove multi-line comments
    source_code = re.sub(r"/\*.*?\*/", "", source_code, flags=re.DOTALL)
    source_code = re.sub(r"\"\"\".*?\"\"\"/", "", source_code, flags=re.DOTALL)

    source_code = re.sub(r"//.*", "", source_code)

    # Remove redundant spaces and tabs
    source_code = re.sub(r"[\t ]+", " ", source_code)

    # Remove empty lines
    source_code = re.sub(r"^\s*\n", "", source_code, flags=re.MULTILINE)
    return source_code

def get_RGB_image(bytecode):
    image = np.frombuffer(bytecode, dtype=np.uint8)
    length = int(math.ceil(len(image)/3))
    image = np.pad(image, pad_width=(0, length*3 - len(image)))
    image = image.reshape((-1, 3))
    sqrt_len = int(math.ceil(math.sqrt(image.shape[0])))
    image = np.pad(image,  pad_width=((0, sqrt_len**2 - image.shape[0]),(0,0)))
    image = image.reshape((sqrt_len, sqrt_len, 3))
    image = Image.fromarray(image)
    # print('image', image)
    return image

def generate_image(example):
    code = HexBytes(example)
    example = get_RGB_image(code)
    return example

# 示例函数：将 PIL.Image 转换为字节
def image_to_bytes(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')  # 将图像保存为字节流
    return img_byte_arr.getvalue()

if __name__ == '__main__':
    df = pd.read_parquet('src/data/mwritescode/slither-audited-smart-contracts/data/test/test.parquet')
    print(df)
    # image = generate_image(df['bytecode'])
    df.insert(3, 'image', df['bytecode'])
    df = df[df['bytecode'] != '0x']
    df = df.reset_index(drop=True)
    print(df)
    for i in range(15921):
        # print('bytecode', df.loc[i, 'bytecode'])
        # print('image', df.loc[i, 'image'])
        # print(remove_comment(df.loc[i, 'source_code']))
        # print(get_RGB_image(HexBytes(df.loc[i, 'bytecode'])))
        df.loc[i, 'source_code'] = remove_comment(df.loc[i, 'source_code'])
        image = generate_image(HexBytes(df.loc[i, 'image']))
        df.loc[i, 'image'] = image_to_bytes(image)
        print(df.loc[i, 'image'])
    df.to_parquet('src/data/mwritescode/slither-audited-smart-contracts/data/test/test_3.parquet', index=False)