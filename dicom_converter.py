import datetime
import gzip
from pathlib import Path
from pydicom import uid, datadict
from pydicom.dataset import FileDataset, FileMetaDataset
from lxml import etree
from typing import Dict, Any
import os

def read_fbp_params(ct_dir: Path) -> Dict[str, Any]:
    files = list(ct_dir.glob("*"))
    if len(files) < 2:
        raise FileNotFoundError("Не найден второй файл в директории.")

    second_file = files[1]
    new_name = second_file.with_suffix(".xml")
    os.rename(second_file, new_name)

    xml_file = next(ct_dir.glob("*.xml"), None)
    print(xml_file)
    if xml_file is None:
        raise FileNotFoundError("В указанной директории не найдено XML файла.")

    with gzip.open(xml_file, "rb") as f:
        tree = etree.fromstring(f.read())

    assert tree.tag == "FBPParams"
    params = tree.find("LibParams")
    assert params is not None
    assert params.find("ZoomFactor").text == "1"

    rows = int(params.find('VolSizeX').text)
    assert rows > 0
    columns = int(params.find('VolSizeY').text)
    assert columns > 0
    bits = int(params.find('ImageInBitDepth').text)
    assert bits > 8 and bits <= 16
    value_max = int(params.find('VoxelValueMax').text)
    assert value_max < (2 << bits)

    retval = {
        'ContentDate': "19700101",
        'ContentTime': "00:00:00",
        'Rows': rows,
        'Columns': columns,
        'BitsStored': bits,
        'HighBit': bits - 1,
        'PixelSpacing': [float(params.find('VoxelSizeX').text), float(params.find('VoxelSizeY').text)],
        'SliceThickness': float(params.find('VoxelSizeZ').text),
        'SpacingBetweenSlices': float(params.find('VoxelSizeZ').text),
        'ImagePositionPatient': [float(params.find('CenterX').text), float(params.find('CenterY').text), float(params.find('CenterZ').text)],
        'SliceLocation': float(params.find('CenterZ').text),
        'StudyID': params.find("ScanID").text,
        'StudyDate': "19700101",
        'StudyTime': "00:00:00",
        'FileSetID': params.find("ScanID").text,
        'WindowCenter': (0 + value_max + 1) // 2,
        'WindowWidth': value_max + 1,
        'RescaleIntercept': -1000,
        'RescaleType': 'HU',
        'RescaleSlope': 1.0,
    }
    return retval



def glx2dicom(src_dir: Path, dst_dir: Path, dicom_attrs) -> None:
    uid_prefix = None 
    attrs = dicom_attrs.copy()

    media_storage_sop_class_uid = attrs.pop('MediaStorageSOPClassUID', None)
    sop_class_uid = attrs.pop('SOPClassUID', media_storage_sop_class_uid)
    if not sop_class_uid:
        sop_class_uid = uid.CTImageStorage
    elif (media_storage_sop_class_uid and sop_class_uid != media_storage_sop_class_uid):
        raise ValueError('SOPClassUID != MediaStorageSOPClassUID', (sop_class_uid, media_storage_sop_class_uid))

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    ds = FileDataset(None, {}, file_meta=file_meta, is_implicit_VR=False, is_little_endian=True)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    media_storage_sop_instace_uid = attrs.pop('MediaStorageSOPInstanceUID', None)
    sop_instance_uid = attrs.pop('SOPInstanceUID', media_storage_sop_instace_uid)
    if not sop_instance_uid:
        sop_instance_uid = uid.generate_uid(uid_prefix)
    elif (media_storage_sop_instace_uid and sop_instance_uid != media_storage_sop_instace_uid):
        raise ValueError('SOPInstanceUID != MediaStorageSOPInstanceUID', (sop_instance_uid, media_storage_sop_instace_uid))

    for tag in ['StudyInstanceUID', 'SeriesInstanceUID', 'FrameOfReferenceUID']:
        value = attrs.pop(tag, None) or uid.generate_uid(uid_prefix)
        setattr(ds, tag, value)

    instance_creation_date = attrs.pop('InstanceCreationDate', '')
    instance_creation_time = attrs.pop('InstanceCreationTime', '')
    if (not instance_creation_date) and (not instance_creation_date):
        dt = datetime.datetime.now()
        instance_creation_date = dt.strftime("%Y%m%d")
        instance_creation_time = dt.strftime("%H%M%S.%f")
    ds.InstanceCreationDate = instance_creation_date
    ds.InstanceCreationTime = instance_creation_time

    for k, v in attrs.items():
        tag1 = (datadict.tag_for_keyword(k) & 0xffff0000) >> 16
        obj = ds
        if tag1 == 0x0002:
            obj = file_meta
        setattr(obj, k, v)

    transfer_syntax_uid = attrs['TransferSyntaxUID']
    photometric_interpretation = attrs['PhotometricInterpretation']
    slice_spacing = attrs['SpacingBetweenSlices']
    z_center = attrs['ImagePositionPatient'][2]
    pixel_bytes, bits_allocated_remnant = divmod(attrs['BitsAllocated'], 8)
    assert bits_allocated_remnant == 0
    data_length = attrs['Rows']*attrs['Columns']*pixel_bytes

    src = sorted(src_dir.glob(src_dir.name + '_[0-9]*[0-9]'))
    if not len(src) > 0:
        raise RuntimeError(f'No files found in {src_dir:r}', src_dir)
    z_base = z_center - 0.5*slice_spacing*(len(src) + 1)

    i = 1
    for f in src:
        print(f'{i:3d} {f}')

        ds.PhotometricInterpretation = photometric_interpretation
        file_meta.TransferSyntaxUID = transfer_syntax_uid
        file_meta.MediaStorageSOPInstanceUID = uid.generate_uid(uid_prefix)
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = str(i)  
        z = z_base + i*slice_spacing
        ds.ImagePositionPatient[2] = z
        ds.SliceLocation = z  
        with gzip.open(f, "rb") as img:
            data = img.read()
        assert len(data) == data_length
        ds.PixelData = data

        ds.compress(uid.RLELossless, encoding_plugin='gdcm')

        output_path = dst_dir / f"image_{i:03d}.dcm"
        ds.save_as(output_path, write_like_original=False)

        i += 1