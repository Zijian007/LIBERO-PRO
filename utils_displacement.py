from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import os
from libero.libero import get_libero_path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import libero.libero.envs.bddl_utils as BDDLUtils
import tempfile
from libero.libero.envs.bddl_utils import generate_bddl_from_parsed_problem
import random
from pathlib import Path
from typing import List, Dict, Tuple, Union
import copy
import zipfile
import pickle


def resize_image_no_tf(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.
    This version doesn't use TensorFlow.
    """
    assert isinstance(resize_size, tuple)
    
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(img.astype(np.uint8))
    
    # Resize using PIL with Lanczos resampling (similar to tf's lanczos3)
    resized_pil = pil_img.resize(resize_size, Image.LANCZOS)
    
    # Convert back to numpy array
    resized_img = np.array(resized_pil)
    
    # Ensure values are in valid range [0, 255] and uint8 type
    resized_img = np.clip(resized_img, 0, 255).astype(np.uint8)
    
    return resized_img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    # img = obs["agentview_image"]
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    # img = resize_image_no_tf(img, resize_size)
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    return img


def modify_object_region_position(parsed_problem, object_name, direction, displacement):
    """
    修改parsed_problem中指定物体所在区域的位置
    
    Args:
        parsed_problem: 解析后的BDDL问题字典
        object_name: 要修改的物体名称
        direction: 控制方向，可以是'x', 'y', 或'xy'
        displacement: 位移量，如果direction是'xy'，则应该是[dx, dy]的列表
    
    Returns:
        修改后的parsed_problem字典
    """
    # 从initial_state中找到物体对应的区域
    region_key = None
    for state in parsed_problem["initial_state"]:
        if len(state) >= 3 and state[1] == object_name:
            region_key = state[2]
            break
    
    if region_key is None:
        print(f"Error: Object '{object_name}' not found in initial_state")
        return parsed_problem
    
    if region_key not in parsed_problem["regions"]:
        print(f"Error: Region '{region_key}' not found in regions")
        return parsed_problem
    
    # 获取区域信息
    region_info = parsed_problem["regions"][region_key]
    print(f"Old ranges: {region_info['ranges']}")
    
    # 修改ranges中的位置
    for i, range_coords in enumerate(region_info["ranges"]):
        x_min, y_min, x_max, y_max = range_coords
        
        if direction == 'x':
            # 只在x方向移动
            dx = displacement
            new_range = [x_min + dx, y_min, x_max + dx, y_max]
        elif direction == 'y':
            # 只在y方向移动
            dy = displacement
            new_range = [x_min, y_min + dy, x_max, y_max + dy]
        elif direction == 'xy':
            # 在x和y方向都移动
            if not isinstance(displacement, (list, tuple)) or len(displacement) != 2:
                print("Error: For 'xy' direction, displacement should be [dx, dy]")
                return parsed_problem
            dx, dy = displacement
            new_range = [x_min + dx, y_min + dy, x_max + dx, y_max + dy]
        else:
            print("Error: Direction should be 'x', 'y', or 'xy'")
            return parsed_problem
        
        # 更新range
        region_info["ranges"][i] = new_range
    
    print(f"Successfully modified region '{region_key}' for object '{object_name}'")
    print(f"New ranges: {region_info['ranges']}")
    
    return parsed_problem


def select_objects_to_perturb(all_objects: List[str], 
                              obj_of_interest: List[str], 
                              perturb_interest: bool, 
                              num_objects: int,
                              seed: int = None) -> List[str]:
    """
    选择要扰动的物体
    
    Args:
        all_objects: 所有物体名称列表
        obj_of_interest: 关键物体名称列表
        perturb_interest: 是否扰动关键物体
        num_objects: 要扰动的物体数量
        seed: 随机种子
    
    Returns:
        选中的物体名称列表
    """
    if seed is not None:
        random.seed(seed)
    
    if perturb_interest:
        # 从关键物体中选择
        candidate_objects = obj_of_interest
    else:
        # 从非关键物体中选择
        candidate_objects = [obj for obj in all_objects if obj not in obj_of_interest]
    
    # 确保不超过候选物体数量
    num_to_select = min(num_objects, len(candidate_objects))
    
    if num_to_select == 0:
        print(f"警告: 没有可选择的物体")
        return []
    
    selected = random.sample(candidate_objects, num_to_select)
    return selected


def get_perturbation_values(direction: str, 
                           displacement: Union[float, List[float]], 
                           use_random: bool = False,
                           random_range: Dict = None,
                           seed: int = None) -> Tuple[str, Union[float, List[float]]]:
    """
    获取扰动方向和位移值
    
    Args:
        direction: 扰动方向 ('x', 'y', 'xy')
        displacement: 位移量
        use_random: 是否使用随机扰动
        random_range: 随机扰动范围
        seed: 随机种子
    
    Returns:
        (方向, 位移值)
    """
    if seed is not None:
        random.seed(seed)
    
    if not use_random:
        return direction, displacement
    
    # 使用随机扰动
    if direction == "x":
        dx = random.uniform(random_range["x"][0], random_range["x"][1])
        return "x", dx
    elif direction == "y":
        dy = random.uniform(random_range["y"][0], random_range["y"][1])
        return "y", dy
    elif direction == "xy":
        dx = random.uniform(random_range["x"][0], random_range["x"][1])
        dy = random.uniform(random_range["y"][0], random_range["y"][1])
        return "xy", [dx, dy]
    else:
        raise ValueError(f"Unknown direction: {direction}")


def apply_perturbations(parsed_problem: Dict,
                       objects_to_perturb: List[str],
                       direction: str,
                       displacement: Union[float, List[float]]) -> Dict:
    """
    对多个物体应用扰动
    
    Args:
        parsed_problem: 解析后的 BDDL 问题
        objects_to_perturb: 要扰动的物体列表
        direction: 扰动方向
        displacement: 位移量
    
    Returns:
        修改后的 parsed_problem
    """
    modified_problem = copy.deepcopy(parsed_problem)
    
    for obj_name in objects_to_perturb:
        print(f"  扰动物体: {obj_name}")
        modified_problem = modify_object_region_position(
            modified_problem, 
            obj_name, 
            direction, 
            displacement
        )
    
    return modified_problem


def save_perturbed_bddl(parsed_problem: Dict,
                       problem_folder: str,
                       original_bddl_filename: str,
                       suffix: str) -> str:
    """
    保存扰动后的 BDDL 文件
    
    Args:
        parsed_problem: 修改后的 parsed_problem
        problem_folder: 原始 problem_folder (如 "libero_10")
        original_bddl_filename: 原始 BDDL 文件名 (如 "KITCHEN_SCENE3_xxx.bddl")
        suffix: 添加到 problem_folder 后的后缀 (如 "_displacement")
    
    Returns:
        保存的文件路径
    """
    # 构建输出目录: get_libero_path("bddl_files") / (problem_folder + suffix)
    output_dir = os.path.join(get_libero_path("bddl_files"), problem_folder + suffix)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成 BDDL 内容
    bddl_content = generate_bddl_from_parsed_problem(parsed_problem)
    
    # 使用原始文件名
    filepath = os.path.join(output_dir, original_bddl_filename)
    
    # 保存文件
    with open(filepath, 'w') as f:
        f.write(bddl_content)
    
    print(f"  ✓ 已保存到: {filepath}")
    return filepath


def generate_init_states_for_bddl(
    bddl_file_path: str,
    output_dir: str,
    num_inits: int = 50,
    height: int = 128,
    width: int = 128
) -> str:
    """
    为单个 BDDL 文件生成初始状态
    
    Args:
        bddl_file_path: BDDL 文件路径
        output_dir: 输出目录
        num_inits: 生成的初始状态数量
        height: 相机高度
        width: 相机宽度
    
    Returns:
        生成的 .pruned_init 文件路径
    """
    bddl_path = Path(bddl_file_path)
    task_base_name = bddl_path.stem
    
    all_initial_states = []
    
    print(f"  生成任务 {task_base_name} 的初始状态...")
    
    for i in range(num_inits):
        env = None
        try:
            env_args = {
                "bddl_file_name": str(bddl_file_path),
                "camera_heights": height,
                "camera_widths": width,
            }
            env = OffScreenRenderEnv(**env_args)
            
            initial_state = env.get_sim_state()
            all_initial_states.append(initial_state)
            
        except Exception as e:
            print(f"    生成第 {i+1} 个状态时出错: {e}")
        
        finally:
            if env is not None and hasattr(env, 'close'):
                env.close()
    
    # 保存初始状态
    output_filename = f"{task_base_name}.pruned_init"
    output_filepath = os.path.join(output_dir, output_filename)
    
    try:
        with zipfile.ZipFile(output_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            all_initial_states = np.array(all_initial_states)
            pickled_states_list = pickle.dumps(all_initial_states)
            zipf.writestr("archive/data.pkl", pickled_states_list)
            zipf.writestr("archive/version", b"1")
        
        print(f"    ✓ 成功保存 {len(all_initial_states)} 个状态到: {output_filepath}")
        return output_filepath
        
    except Exception as e:
        print(f"    ✗ 保存状态列表时出错: {e}")
        return None