# 调用VirusTotal 的api来扫描和生成报告

import requests
import os
import json
import time
import sys

scan_url = 'https://www.virustotal.com/vtapi/v2/file/scan'
report_url = 'https://www.virustotal.com/vtapi/v2/file/report'

folder_path = 'D:/BaiduNetdiskDownload/VirusShare_00177/'
labels_path = 'D:/BaiduNetdiskDownload/VirusShare_labels/'
json_save_path = 'D:/BaiduNetdiskDownload/VirusShare_jsons/'

AVClass_path = 'D:/BaiduNetdiskDownload/avclass-master/'
# AVClass_path = '../../modules/avclass-master/'
AVClass_template = 'python ' + AVClass_path + 'avclass_labeler.py %s %s > %s'


apikey = 'c424abc9c8d7102cfaf9cf2d8f01fb95f4ddfd81a563d6e07738fa960b501d87'
scan_params = {'apikey': apikey}

report_platform = 'McAfee'

start_index = len(os.listdir(json_save_path))
end_index = 40000

assert start_index < end_index, '尝试使用一个更大的end下标值！目前的起始下标:%d' % start_index

print('Begin to scan...')
samples_list = os.listdir(folder_path)
last_stamp = time.time()
while start_index < end_index:
    # print(start_index, time.time()%100000)
    print(start_index)
    f = samples_list[start_index]
    if (os.path.exists(json_save_path+f+'.json') and os.path.getsize(json_save_path+f+'.json') != 0):
        start_index += 1
        last_stamp = time.time()
        continue

    files_cfg = {'file': ('test', open(folder_path+f, 'rb'))}

    try:
        print('scanning...')
        response = requests.post(scan_url, files=files_cfg, params=scan_params)
    except BaseException as e:
        print(f, ': api request exceeds!', ' error:', str(e))
        print('waiting...')
        time.sleep(60)
        continue

    # print(type(response.json()))
    scan_info = response.json()
    # print(response.json())

    report_params = {'apikey': apikey, 'resource': scan_info['md5']}
    try:
        print('fetching report...')
        report = requests.get(report_url, params=report_params)
        report = report.json()#['scans']
    except BaseException as e:
        print(f, ': api request exceeds!', ' error:', str(e))
        print('waiting...')
        time.sleep(60)
        continue
    # print(report)
    print(report['verbose_msg'])
    if report['response_code'] == 1:
        with open('%s.json'%(json_save_path+f),'w') as fp:
            json.dump(report, fp)
    else:
        sys.stderr.write('%s wrong response code %d'%(f, report['response_code']))

    print('time consuming: %.2f'%(time.time()-last_stamp))
    last_stamp = time.time()

    # time.sleep(20.5)
    start_index += 1
        # report = report['scans']
        # if report[report_platform]['detected']:
        #     print('%s:\n'%report_platform,report[report_platform]['result'])
        #     # print(report[report_platform])
        # else:
        #     print(f, ' fail to be detected by %s'%report_platform)

    # scan_ids[f] = scan_info['md5']


# print('Begin to fetch reports...')
# for f,id in scan_ids.items():



        # print(AVClass_template % ('-vt',
        #                           folder_path + 'VirusShare_00002a26eadf58972a336ea3d17c2a20.json',
        #                           'VirusShare_00002a26eadf58972a336ea3d17c2a2.labels'))
# print('Begin to convert...')
# # print((labels_path+'malware.labels'))
# os.system(AVClass_template % ('-vtdir',
#                               json_save_path,
#                               (labels_path+'malware.labels'))) #labels_path + f + '.labels'))





