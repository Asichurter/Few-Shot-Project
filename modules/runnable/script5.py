import requests
import os

# save_path = 'C:/Users/Asichurter/Desktop/dl_malwares/'
#
# url = 'https://www.virustotal.com/vtapi/v2/file/download'
# apikey = 'c424abc9c8d7102cfaf9cf2d8f01fb95f4ddfd81a563d6e07738fa960b501d87'
# hashes = [  "63956d6417f8f43357d9a8e79e52257e"
#             "6f7bde7a1126debf0cc359a54953efc1"
#             "7520c8f9534ca818726a4feaebf49e2b"
#             "e435a536968941854bcec3b902c439f6"
#             "e93049e2df82ab26f35ad0049173cb14"
#             "4235e2d487958ff377f0f92b266591f0"
#             "e4647acec12b82944f5df603dc682660"
#             "6524a10da9701301b2582f12cc66f90c"
#             "14a3f5108958b61c6bdc2de17c785a89"
#             "1515a80662d5bd0d8a6fb9ecfeedb652"
#             "94e4b62861ab7a4d3246a4888e9025b5"
#             "cfb9fbcd2bb1ca2d326720971f385a4b"
#             "a89153d58a70f143ed1fd3b89f26a90f"
#             "323037966ab54ce841f528870908e259"
#             "5af2ee5a9e61b194f6cb076775237980"
#             "ac79fefb5ddfe4f20061bca398884233"
#             "52411226cbdd24441966c08f959ad5dc"
#             "3742f0a58ca91a0c56c74f49dd22ab0b"
#             "485a4912b2d639694f836451a2b30435"
#             "b29bea9ae0292d8a6a18219b63a62787"]
#
# for i,hash_val in enumerate(hashes):
#     print(i, hash_val)
#     params = {'apikey': apikey, 'hash': hash_val}
#
#     response = requests.get(url, params=params)
#     print(response)
#     downloaded_file = response.content
#
#     with open(save_path+str(i)+'.pe', 'wb') as f:
#         f.write(downloaded_file)

url = 'https://www.virustotal.com/gui/file/968c37e74571c6f3bf8f2749c9e1d0ea6999eb503de2a9a6cc78c68530559c6d/detection.html'
response = requests.get(url)

