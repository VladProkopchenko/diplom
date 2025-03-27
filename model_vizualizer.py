import os
import numpy as np
import vtk
from PIL import Image
from vtk.util import numpy_support
from scipy.ndimage import gaussian_filter

# Фильтрация данных
def apply_gaussian_filter(volume, sigma=1):
    return gaussian_filter(volume, sigma=sigma)

# Конвертация в grayscale
def convert_to_grayscale(image):
    return image.convert("L")

# Изменение размера изображений
def resize_image(image, target_size):
    return image.resize(target_size, Image.LANCZOS)

# Загрузка PNG файлов в объём
def load_png_series(directory, target_size=(512, 512)):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.png')]
    
    if not files:
        raise ValueError(f"Нет файлов в папке: {directory}")
    else:
        print(f"Найдено {len(files)} файлов.")

    files.sort()
    
    images = [np.array(convert_to_grayscale(resize_image(Image.open(f), target_size))) for f in files]
    volume = np.stack(images, axis=0)

    # Применение фильтрации
    volume = apply_gaussian_filter(volume)
    return volume

# Создание vtkImageData
def create_vtk_image_data(volume, spacing=(1, 1, 0.5)):
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(volume.shape[2], volume.shape[1], volume.shape[0])
    vtk_data.SetSpacing(spacing)
    vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    
    flat_volume = volume.flatten()
    vtk_array = numpy_support.numpy_to_vtk(flat_volume, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_data.GetPointData().SetScalars(vtk_array)
    
    return vtk_data

# Создание модели с помощью алгоритма Marching Cubes
def create_3d_model(vtk_data, threshold=100):
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_data)
    marching_cubes.SetValue(0, threshold)  
    marching_cubes.Update()

    return marching_cubes.GetOutput()

# Сглаживание модели
def smooth_model(model, iterations=20, relaxation_factor=0.1):
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(model)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetRelaxationFactor(relaxation_factor)
    smoother.Update()
    return smoother.GetOutput()

# Уменьшение количества полигонов
def reduce_polygons(model, reduction_factor=0.5):
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(model)
    decimate.SetTargetReduction(reduction_factor)  
    decimate.Update()

    return decimate.GetOutput()

# Удаление мелких компонентов
def remove_small_components(model, threshold=100):
    connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
    connectivity_filter.SetInputData(model)
    connectivity_filter.SetExtractionModeToLargestRegion()
    connectivity_filter.SetScalarRange(threshold, float('inf'))
    connectivity_filter.Update()
    return connectivity_filter.GetOutput()

# Визуализация модели
def visualize(model):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(model)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0, 0)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.Render()
    render_window_interactor.Start()

# Сохранение модели в STL
def save_as_stl(model, output_file):
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(output_file)
    stl_writer.SetInputData(model)
    stl_writer.Write()

# Сохранение модели в OBJ
def save_as_obj(model, output_file):
    obj_writer = vtk.vtkOBJWriter()
    obj_writer.SetFileName(output_file)
    obj_writer.SetInputData(model)
    obj_writer.Write()


