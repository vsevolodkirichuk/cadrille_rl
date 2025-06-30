import cadquery as cq
import tempfile
import os
import trimesh
import open3d
import skimage
import numpy as np
from PIL import Image, ImageOps


def code_to_image(code: str, width=512, height=512, num_views=1) -> Image:
    """
    Принимает CadQuery-код в виде строки и возвращает изображение модели.
    Использует подход из dataset.py с Open3D для более стабильной работы.

    Args:
        code: CadQuery код в виде строки
        width: ширина изображения
        height: высота изображения
        num_views: количество видов (1, 2 или 4)
    """
    namespace = {}
    try:
        print(f"[DEBUG] Executing code...")
        # Выполняем код и получаем Workplane
        exec(code, {"cq": cq}, namespace)

        print(f"[DEBUG] Namespace keys: {list(namespace.keys())}")
        print(f"[DEBUG] Namespace values types: {[(k, type(v)) for k, v in namespace.items()]}")

        # Ищем результат - предпочитаем 'result', 'r', затем последний Workplane
        model = None
        workplanes = [(k, v) for k, v in namespace.items() if isinstance(v, cq.Workplane)]

        if not workplanes:
            raise ValueError(
                f"В коде не найден объект Workplane. Найденные объекты: {[(k, type(v)) for k, v in namespace.items()]}")

        # Приоритет поиска: result -> r -> последний по алфавиту
        for preferred_name in ['result', 'r']:
            for key, value in workplanes:
                if key == preferred_name:
                    model = value
                    print(f"[DEBUG] Found preferred Workplane: {key}")
                    break
            if model is not None:
                break

        # Если не нашли предпочтительные, берем последний
        if model is None:
            key, model = workplanes[-1]  # последний в списке
            print(f"[DEBUG] Using last Workplane: {key}")

        # Проверяем, что в Workplane есть геометрия
        try:
            solids = model.solids()
            if not solids.objects:
                print(f"[DEBUG] Workplane {key} has no solids, trying other workplanes...")
                # Пробуем другие workplanes
                for k, v in workplanes:
                    if k != key:
                        test_solids = v.solids()
                        if test_solids.objects:
                            model = v
                            print(f"[DEBUG] Found workplane with solids: {k}")
                            break
                else:
                    raise ValueError("Все Workplane объекты пусты")
            else:
                print(f"[DEBUG] Workplane {key} has {len(solids.objects)} solids")
        except Exception as e:
            print(f"[DEBUG] Error checking solids: {e}")
            # Продолжаем с текущей моделью

        # Экспортируем в STL через временный файл
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            print(f"[DEBUG] Exporting to STL: {tmp_path}")
            # Экспортируем модель в STL - используем правильный метод
            cq.exporters.export(model, tmp_path)

            print(f"[DEBUG] Loading mesh from STL...")
            # Загружаем как trimesh
            loaded = trimesh.load(tmp_path)

            # Проверяем тип загруженного объекта
            if isinstance(loaded, trimesh.Scene):
                print(f"[DEBUG] Loaded Scene with {len(loaded.geometry)} geometries")
                # Извлекаем первую геометрию из сцены
                if loaded.geometry:
                    mesh = list(loaded.geometry.values())[0]
                    print(f"[DEBUG] Extracted mesh from scene")
                else:
                    raise ValueError("Scene contains no geometry")
            else:
                mesh = loaded
                print(f"[DEBUG] Loaded mesh directly")

            print(f"[DEBUG] Mesh loaded: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

            # Нормализуем и центрируем меш (как в dataset.py)
            # Вычисляем bounding box для нормализации
            bounds = mesh.bounds
            size = bounds[1] - bounds[0]
            max_size = np.max(size)
            print(f"[DEBUG] Mesh bounds: {bounds}, max_size: {max_size}")

            # Масштабируем чтобы поместился в единичный куб
            if max_size > 0:
                scale_factor = 1.0 / max_size
                mesh.apply_scale(scale_factor)
                print(f"[DEBUG] Applied scale: {scale_factor}")

            # Центрируем в точке (0.5, 0.5, 0.5)
            center = mesh.bounds.mean(axis=0)
            mesh.apply_translation([0.5, 0.5, 0.5] - center)
            print(f"[DEBUG] Applied translation: {[0.5, 0.5, 0.5] - center}")

            # Конвертируем в Open3D меш
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)

            #print(f"[DEBUG] Converting to Open3D mesh...")
            o3d_mesh = open3d.geometry.TriangleMesh()
            o3d_mesh.vertices = open3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = open3d.utility.Vector3iVector(faces)
            o3d_mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)  # Желтоватый цвет как в dataset.py
            o3d_mesh.compute_vertex_normals()
            #print(f"[DEBUG] Open3D mesh created successfully")

            # Определяем углы обзора
            fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
            images = []

            #print(f"[DEBUG] Rendering {num_views} views...")
            for i, front in enumerate(fronts[:num_views if num_views <= 4 else 4]):
                try:
                    #print(f"[DEBUG] Rendering view {i} with front={front}")
                    image = mesh_to_image_o3d(
                        o3d_mesh,
                        camera_distance=-0.9,
                        front=front,
                        width=width,
                        height=height,
                        img_size=min(width, height)
                    )
                    if image is not None:
                        images.append(image)
                        #print(f"[DEBUG] View {i} rendered successfully")
                    else:
                        print(f"Warning: mesh_to_image_o3d returned None for view {i}")
                        # Создаем белую заглушку
                        images.append(Image.new("RGB", (min(width, height), min(width, height)), (255, 255, 255)))
                except Exception as e:
                    print(f"Error rendering view {i}: {e}")
                    # Создаем белую заглушку
                    images.append(Image.new("RGB", (min(width, height), min(width, height)), (255, 255, 255)))

            # Проверяем что у нас есть изображения
            if not images:
                print("No images generated, returning white placeholder")
                return Image.new("RGB", (width, height), (255, 255, 255))

            # Добавляем черную рамку как в dataset.py
            #print(f"[DEBUG] Adding borders to {len(images)} images...")
            try:
                images = [ImageOps.expand(image, border=3, fill='black') for image in images]
                #print(f"[DEBUG] Borders added successfully")
            except Exception as e:
                print(f"[DEBUG] Error adding borders: {e}")
                # Используем изображения без рамок
                pass

            # Компонуем изображения в зависимости от num_views
            #print(f"[DEBUG] Composing final image for {num_views} views...")
            if num_views == 1:
                final_image = images[0]
            elif num_views == 2:
                final_image = Image.fromarray(np.hstack((
                    np.array(images[0]), np.array(images[1])
                )))
            elif num_views == 4:
                final_image = Image.fromarray(np.vstack((
                    np.hstack((np.array(images[0]), np.array(images[1]))),
                    np.hstack((np.array(images[2]), np.array(images[3])))
                )))
            else:
                final_image = images[0]

            print(f"[DEBUG] Final image created: {final_image.size}")
            return final_image

        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        print(f"[code_to_image ERROR]: {e}")
        # Возвращаем белую заглушку
        return Image.new("RGB", (width, height), (255, 255, 255))


def mesh_to_image_o3d(mesh, camera_distance=-1.8, front=[1, 1, 1], width=500, height=500, img_size=128):
    """
    Рендерит Open3D меш в изображение. Скопировано из dataset.py.
    """
    try:
        vis = open3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        vis.add_geometry(mesh)

        lookat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        front_array = np.array(front, dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)

        eye = lookat + front_array * camera_distance
        right = np.cross(up, front_array)
        right_norm = np.linalg.norm(right)
        if right_norm > 0:
            right /= right_norm
        else:
            right = np.array([1, 0, 0])  # fallback

        true_up = np.cross(front_array, right)
        rotation_matrix = np.column_stack((right, true_up, front_array)).T
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation_matrix
        extrinsic[:3, 3] = -rotation_matrix @ eye

        view_control = vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        camera_params.extrinsic = extrinsic
        view_control.convert_from_pinhole_camera_parameters(camera_params)

        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        if image is None:
            print("Warning: vis.capture_screen_float_buffer returned None")
            return None

        image = np.asarray(image)
        if image.size == 0:
            print("Warning: captured image is empty")
            return None

        image = (image * 255).astype(np.uint8)
        image = skimage.transform.resize(
            image,
            output_shape=(img_size, img_size),
            order=2,
            anti_aliasing=True,
            preserve_range=True).astype(np.uint8)

        return Image.fromarray(image)

    except Exception as e:
        print(f"Error in mesh_to_image_o3d: {e}")
        return None