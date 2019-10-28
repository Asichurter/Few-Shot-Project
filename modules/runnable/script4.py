import requests
import os
import json

scan_url = 'https://www.virustotal.com/vtapi/v2/file/scan'
report_url = 'https://www.virustotal.com/vtapi/v2/file/report'

folder_path = 'C:/Users/Asichurter/Desktop/malwares/samples/'
labels_path = 'C:/Users/Asichurter/Desktop/malwares/labels/'
json_save_path = 'C:/Users/Asichurter/Desktop/malwares/jsons/'

AVClass_path = 'D:/BaiduNetdiskDownload/avclass-master/'
# AVClass_path = '../../modules/avclass-master/'
AVClass_template = 'python ' + AVClass_path + 'avclass_labeler.py %s %s > %s'


apikey = 'c424abc9c8d7102cfaf9cf2d8f01fb95f4ddfd81a563d6e07738fa960b501d87'
scan_params = {'apikey': apikey}

scan_ids = {}

report_platform = 'McAfee'

print('Begin to scan...')
for f in os.listdir(folder_path):
    files_cfg = {'file': ('test', open(folder_path+f, 'rb'))}

    response = requests.post(scan_url, files=files_cfg, params=scan_params)

    # print(type(response.json()))
    scan_info = response.json()
    # print(response.json())
    scan_ids[f] = scan_info['md5']
    print(scan_info)

print('Begin to fetch reports...')
for f,id in scan_ids.items():
    report_params = {'apikey': apikey, 'resource': id}
    try:
        report = requests.get(report_url, params=report_params)
        report = report.json()#['scans']
    except json.decoder.JSONDecodeError:
        print(f, report)
        continue
    # print(report)
    print(report['verbose_msg'])
    if report['response_code'] == 1:
        with open('%s.json'%(json_save_path+f),'w') as fp:
            json.dump(report, fp)
        report = report['scans']
        if report[report_platform]['detected']:
            print('%s:\n'%report_platform,report[report_platform]['result'])
            # print(report[report_platform])
        else:
            print(f, ' fail to be detected by %s'%report_platform)

        # print(AVClass_template % ('-vt',
        #                           folder_path + 'VirusShare_00002a26eadf58972a336ea3d17c2a20.json',
        #                           'VirusShare_00002a26eadf58972a336ea3d17c2a2.labels'))
print('Begin to convert...')
# print((labels_path+'malware.labels'))
os.system(AVClass_template % ('-vtdir',
                              json_save_path,
                              (labels_path+'malware.labels'))) #labels_path + f + '.labels'))





