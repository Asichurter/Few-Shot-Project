# This module is modified from code in paper "Transfer learning for Image-Based Malware classification"
# It aims to extract code features that will be input into ML classifier
# This module uses module "pefile" in github: https://github.com/erocarrera/pefile

import pefile
import array
import math

COLUMNS = [
    "Machine",
    "SizeOfOptionalHeader",
    "Characteristics",
    "MajorLinkerVersion",
    "MinorLinkerVersion",
    "SizeOfCode",
    "SizeOfInitializedData",
    "SizeOfUninitializedData",
    "AddressOfEntryPoint",
    "BaseOfCode",
    "BaseOfData",
    "ImageBase",
    "SectionAlignment",
    "FileAlignment",
    "MajorOperatingSystemVersion",
    "MinorOperatingSystemVersion",
    "MajorImageVersion",
    "MinorImageVersion",
    "MajorSubsystemVersion",
    "MinorSubsystemVersion",
    "SizeOfImage",
    "SizeOfHeaders",
    "CheckSum",
    "Subsystem",
    "DllCharacteristics",
    "SizeOfStackReserve",
    "SizeOfStackCommit",
    "SizeOfHeapReserve",
    "SizeOfHeapCommit",
    "LoaderFlags",
    "NumberOfRvaAndSizes",
    "SectionsNb",
    "SectionsMeanEntropy",
    "SectionsMinEntropy",
    "SectionsMaxEntropy",
    "SectionsMeanRawsize",
    "SectionsMinRawsize",
    "SectionMaxRawsize",
    "SectionsMeanVirtualsize",
    "SectionsMinVirtualsize",
    "SectionMaxVirtualsize",
    "ImportsNbDLL",
    "ImportsNb",
    "ImportsNbOrdinal",
    "ExportNb",
    "ResourcesNb",
    "ResourcesMeanEntropy",
    "ResourcesMinEntropy",
    "ResourcesMaxEntropy",
    "ResourcesMeanSize",
    "ResourcesMinSize",
    "ResourcesMaxSize",
    "LoadConfigurationSize",
    "VersionInformationSize",
]


def get_entropy(data):
    if len(data) == 0:
        return 0.0
    occurences = array.array('L', [0] * 256)
    for x in data:
        occurences[x if isinstance(x, int) else ord(x)] += 1

    entropy = 0
    for x in occurences:
        if x:
            p_x = float(x) / len(data)
            entropy -= p_x * math.log(p_x, 2)

    return entropy


def get_resources(pe):
    """Extract resources :
    [entropy, size]"""
    resources = []
    if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
        try:
            for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                if hasattr(resource_type, 'directory'):
                    for resource_id in resource_type.directory.entries:
                        if hasattr(resource_id, 'directory'):
                            for resource_lang in resource_id.directory.entries:
                                data = pe.get_data(resource_lang.data.struct.OffsetToData,
                                                   resource_lang.data.struct.Size)
                                size = resource_lang.data.struct.Size
                                entropy = get_entropy(data)
                                resources.append([entropy, size])
        except Exception as e:
            return resources
    return resources


def get_version_info(pe):
    """Return version infos"""
    res = {}
    for fileinfo in pe.FileInfo:
        if fileinfo.Key == 'StringFileInfo':
            for st in fileinfo.StringTable:
                for entry in st.entries.items():
                    res[entry[0]] = entry[1]
        if fileinfo.Key == 'VarFileInfo':
            for var in fileinfo.Var:
                res[var.entry.items()[0][0]] = var.entry.items()[0][1]
    if hasattr(pe, 'VS_FIXEDFILEINFO'):
        res['flags'] = pe.VS_FIXEDFILEINFO.FileFlags
        res['os'] = pe.VS_FIXEDFILEINFO.FileOS
        res['type'] = pe.VS_FIXEDFILEINFO.FileType
        res['file_version'] = pe.VS_FIXEDFILEINFO.FileVersionLS
        res['product_version'] = pe.VS_FIXEDFILEINFO.ProductVersionLS
        res['signature'] = pe.VS_FIXEDFILEINFO.Signature
        res['struct_version'] = pe.VS_FIXEDFILEINFO.StrucVersion
    return res


def extract_infos(fpath, return_type='list'):
    i = 0
    res = {}
    try:
        pe = pefile.PE(fpath)
    except pefile.PEFormatError as e:
        print(e.value)
        return None
    res[COLUMNS[i]] = pe.FILE_HEADER.Machine
    i += 1
    res[COLUMNS[i]] = pe.FILE_HEADER.SizeOfOptionalHeader
    i += 1
    res[COLUMNS[i]] = pe.FILE_HEADER.Characteristics
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.MajorLinkerVersion
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.MinorLinkerVersion
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SizeOfCode
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SizeOfInitializedData
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.BaseOfCode
    i += 1
    try:
        res[COLUMNS[i]] = pe.OPTIONAL_HEADER.BaseOfData
        i += 1
    except AttributeError:
        res[COLUMNS[i]] = 0
        i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.ImageBase
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SectionAlignment
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.FileAlignment
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.MajorImageVersion
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.MinorImageVersion
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SizeOfImage
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SizeOfHeaders
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.CheckSum
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.Subsystem
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.DllCharacteristics
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SizeOfStackReserve
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SizeOfStackCommit
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.LoaderFlags
    i += 1
    res[COLUMNS[i]] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes
    i += 1
    res[COLUMNS[i]] = len(pe.sections)
    i += 1
    entropy = list(map(lambda x: x.get_entropy(), pe.sections))
    res[COLUMNS[i]] = sum(entropy) / float(len(entropy)) if len(entropy) != 0 else 0
    i += 1
    res[COLUMNS[i]] = min(entropy) if len(entropy) != 0 else 0
    i += 1
    res[COLUMNS[i]] = max(entropy) if len(entropy) != 0 else 0
    i += 1
    raw_sizes = list(map(lambda x: x.SizeOfRawData, pe.sections))
    res[COLUMNS[i]] = sum(raw_sizes) / float(len(raw_sizes)) if len(raw_sizes) != 0 else 0
    i += 1
    res[COLUMNS[i]] = min(raw_sizes) if len(raw_sizes) != 0 else 0
    i += 1
    res[COLUMNS[i]] = max(raw_sizes) if len(raw_sizes) != 0 else 0
    i += 1
    virtual_sizes = list(map(lambda x: x.Misc_VirtualSize, pe.sections))
    res[COLUMNS[i]] = sum(virtual_sizes) / float(len(virtual_sizes)) if len(virtual_sizes) != 0 else 0
    i += 1
    res[COLUMNS[i]] = min(virtual_sizes) if len(virtual_sizes) != 0 else 0
    i += 1
    res[COLUMNS[i]] = max(virtual_sizes) if len(virtual_sizes) != 0 else 0
    i += 1
    # Imports
    try:
        res[COLUMNS[i]] = len(pe.DIRECTORY_ENTRY_IMPORT)
        i += 1
        imports = sum([x.imports for x in pe.DIRECTORY_ENTRY_IMPORT], [])
        res[COLUMNS[i]] = len(imports)
        i += 1
        res[COLUMNS[i]] = len(list(filter(lambda x: x.name is None, imports)))
        i += 1
    except AttributeError:
        res[COLUMNS[i]] = 0
        i += 1
        res[COLUMNS[i]] = 0
        i += 1
        res[COLUMNS[i]] = 0
        i += 1
    # Exports
    try:
        res[COLUMNS[i]] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
        i += 1
    except AttributeError:
        # No export
        res[COLUMNS[i]] = 0
        i += 1
    # Resources
    resources = get_resources(pe)
    res[COLUMNS[i]] = len(resources)
    i += 1
    if len(resources) > 0:
        entropy = list(map(lambda x: x[0], resources))
        res[COLUMNS[i]] = sum(entropy) / float(len(entropy)) if len(entropy) != 0 else 0
        i += 1
        res[COLUMNS[i]] = min(entropy) if len(entropy) != 0 else 0
        i += 1
        res[COLUMNS[i]] = max(entropy) if len(entropy) != 0 else 0
        i += 1
        sizes = list(map(lambda x: x[1], resources))
        res[COLUMNS[i]] = sum(sizes) / float(len(sizes)) if len(sizes) != 0 else 0
        i += 1
        res[COLUMNS[i]] = min(sizes) if len(sizes) != 0 else 0
        i += 1
        res[COLUMNS[i]] = max(sizes) if len(sizes) != 0 else 0
        i += 1
    else:
        res[COLUMNS[i]] = 0
        i += 1
        res[COLUMNS[i]] = 0
        i += 1
        res[COLUMNS[i]] = 0
        i += 1
        res[COLUMNS[i]] = 0
        i += 1
        res[COLUMNS[i]] = 0
        i += 1
        res[COLUMNS[i]] = 0
        i += 1

    # Load configuration size
    try:
        res[COLUMNS[i]] = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size
        i += 1
    except AttributeError:
        res[COLUMNS[i]] = 0
        i += 1

    # Version configuration size
    try:
        version_infos = get_version_info(pe)
        res[COLUMNS[i]] = len(version_infos.keys())
        i += 1
    except AttributeError:
        res[COLUMNS[i]] = 0
        i += 1
    if return_type == 'list':
        return list(res.values())
    elif return_type == 'dict':
        return res
    else:
        raise Exception('指定了无效的返回类型参数: ' + return_type)


if __name__ == '__main__':
    path1 = r'D:/pe/backdoor1/Backdoor.Win32.Hupigon.zah'
    path2 = r'C:/Windows/assembly/GAC/ADODB/7.0.3300.0__b03f5f7f11d50a3a/ADODB.dll'
    features = extract_infos(path1)
    features_ = extract_infos(path2)


