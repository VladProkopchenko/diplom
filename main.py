from pathlib import Path
from pydicom import uid, datadict
from tkinter import Tk, filedialog
from dicom_converter import read_fbp_params
from dicom_converter import glx2dicom
from png_converter import process_files
from model_vizualizer import load_png_series
from model_vizualizer import create_3d_model
from model_vizualizer import create_vtk_image_data
from model_vizualizer import reduce_polygons
from model_vizualizer import visualize
from model_vizualizer import save_as_obj
import os



def main():
    
    Tk().withdraw()
    current_dir = Path(__file__).parent

    src_dir = Path(filedialog.askdirectory(title="Выберите директорию"))
    dicom_directory = current_dir / "dicom series"
    png_directory = current_dir / "png series"
    model_directory = current_dir / "obj model\\model.obj"

    
    #input_folder = "C:\\Users\\Admin\\Desktop\\diplom\\test2"
    #output_folder = "C:\\Users\\Admin\\Desktop\\aa"
    file_list = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]

    default_dicom_attrs = {
        'ImplementationClassUID': '2.25.88075347621178512596802224299896711910',
        'SpecificCharacterSet': 'ISO_IR 100',
        'TransferSyntaxUID': uid.ExplicitVRLittleEndian,
        'PatientName': 'Vasya Pupkin',
        'PatientID': '1234456789',
        'StudyDescription': 'Dental CBCT Study32372304',
        'Modality': 'CT',
        'SeriesNumber': "1",
        'SeriesDescription': "CT series",
        'ImageType': ['DERIVED', 'PRIMARY'],
        'NumberOfFrames': 1,
        'ImageOrientationPatient': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        'SamplesPerPixel': 1,
        'PhotometricInterpretation': 'MONOCHROME2',  
        'PixelRepresentation': 1,  
        'BitsAllocated': 16 
    }

    glx_attrs = read_fbp_params(src_dir)

    dicom_attrs = default_dicom_attrs.copy()
    dicom_attrs.update(glx_attrs)

    glx2dicom(src_dir, dicom_directory, dicom_attrs)
    
    process_files(file_list,dicom_directory, png_directory)
    
    volume = load_png_series(png_directory)
    print(volume.shape)

    spacing = (1, 1, 1)
    vtk_data = create_vtk_image_data(volume, spacing)
    print(vtk_data.GetDimensions()) 

    model = create_3d_model(vtk_data)

    reduced_model = reduce_polygons(model, reduction_factor=0.5)

    visualize(reduced_model)

    #output_file = "C:\\Users\\Admin\\Desktop\\diplom\\STL\\tooth_segment_reduced.obj"
    save_as_obj(reduced_model, model_directory)
    print(f"Сохранено в {model_directory}")


if __name__ == '__main__':
    main()

