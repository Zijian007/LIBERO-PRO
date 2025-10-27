from bddl.parsing import *

import itertools
import numpy as np

import tempfile
import os


pi = np.pi


def get_regions(t, regions, group):
    group.pop(0)
    while group:
        region = group.pop(0)
        region_name = region[0]
        target_name = None
        region_dict = {
            "target": None,
            "ranges": [],
            "extra": [],
            "yaw_rotation": [0, 0],
            "rgba": [0, 0, 1, 0],
        }
        for attribute in region[1:]:
            if attribute[0] == ":target":
                assert len(attribute) == 2
                region_dict["target"] = attribute[1]
                target_name = attribute[1]
            elif attribute[0] == ":ranges":
                for rect_range in attribute[1]:
                    assert (
                        len(rect_range) == 4
                    ), f"Dimension of rectangular range mismatched!!, supposed to be 4, only found {len(rect_range)}"
                    region_dict["ranges"].append([float(x) for x in rect_range])
            elif attribute[0] == ":yaw_rotation":
                # print(attribute[1])
                for value in attribute[1]:
                    region_dict["yaw_rotation"] = [eval(x) for x in value]
            elif attribute[0] == ":rgba":
                assert (
                    len(attribute[1]) == 4
                ), f"Missing specification for rgba color, supposed to be 4 dimension, but only got  {attribute[1]}"
                region_dict["rgba"] = [float(x) for x in attribute[1]]
            else:
                raise NotImplementedError
        regions[target_name + "_" + region_name] = region_dict


def get_scenes(t, scene_properties, group):
    group.pop(0)
    while group:
        scene_property = group.pop(0)
        scene_properties_dict = {}
        for attribute in region[1:]:
            if attribute[0] == ":floor":
                assert len(attribute) == 2
                scene_properties_dict["floor_style"] = attribute[1]
            elif attribute[0] == ":wall":
                assert len(attribute) == 2
                scene_properties_dict["wall_style"] = attribute[1]
            else:
                raise NotImplementedError


def get_problem_info(problem_filename):
    domain_name = "unknown"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                domain_name = "robosuite"
            elif t == ":language":
                group.pop(0)
                language_instruction = group
    return {
        "problem_name": problem_name,
        "domain_name": domain_name,
        "language_instruction": " ".join(language_instruction),
    }


def robosuite_parse_problem(problem_filename):
    domain_name = "robosuite"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        objects = {}
        obj_of_interest = []
        initial_state = []
        goal_state = []
        fixtures = {}
        regions = {}
        scene_properties = {}
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                if domain_name != group[-1]:
                    raise Exception("Different domain specified in problem file")
            elif t == ":requirements":
                pass
            elif t == ":objects":
                group.pop(0)
                object_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        objects[group.pop(0)] = object_list
                        object_list = []
                    else:
                        object_list.append(group.pop(0))
                if object_list:
                    if not "object" in objects:
                        objects["object"] = []
                    objects["object"] += object_list
            elif t == ":obj_of_interest":
                group.pop(0)
                while group:
                    obj_of_interest.append(group.pop(0))
            elif t == ":fixtures":
                group.pop(0)
                fixture_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        fixtures[group.pop(0)] = fixture_list
                        fixture_list = []
                    else:
                        fixture_list.append(group.pop(0))
                if fixture_list:
                    if not "fixture" in fixtures:
                        fixtures["fixture"] = []
                    fixtures["fixture"] += fixture_list
            elif t == ":regions":
                get_regions(t, regions, group)
            elif t == ":scene_properties":
                get_scenes(t, scene_properties, group)
            elif t == ":language":
                group.pop(0)
                language_instruction = group

            elif t == ":init":
                group.pop(0)
                initial_state = group
            elif t == ":goal":
                package_predicates(group[1], goal_state, "", "goals")
            else:
                print("%s is not recognized in problem" % t)
        return {
            "problem_name": problem_name,
            "fixtures": fixtures,
            "regions": regions,
            "objects": objects,
            "scene_properties": scene_properties,
            "initial_state": initial_state,
            "goal_state": goal_state,
            "language_instruction": language_instruction,
            "obj_of_interest": obj_of_interest,
        }
    else:
        raise Exception(
            f"Problem {behavior_activity} {activity_definition} does not match problem pattern"
        )


def generate_bddl_from_parsed_problem(parsed_problem):
    """
    将 parsed_problem 字典转换回 BDDL 格式的字符串
    
    Args:
        parsed_problem: robosuite_parse_problem() 返回的字典
        
    Returns:
        BDDL 格式的字符串
    """
    lines = []
    
    # 开始定义
    lines.append(f"(define (problem {parsed_problem['problem_name']})")
    lines.append("    (:domain robosuite)")
    
    # 语言指令
    if parsed_problem.get("language_instruction"):
        lang_instr = parsed_problem["language_instruction"]
        if isinstance(lang_instr, list):
            lang_instr = " ".join(lang_instr)
        lines.append(f"    (:language {lang_instr})")
    
    # Regions
    if parsed_problem.get("regions"):
        lines.append("    (:regions")
        for region_full_name, region_info in parsed_problem["regions"].items():
            # 分离 target 和 region 名称 (e.g., "floor_bin_region" -> "bin_region")
            target = region_info["target"]
            if region_full_name.startswith(target + "_"):
                region_name = region_full_name[len(target) + 1:]
            else:
                region_name = region_full_name
            
            lines.append(f"        ({region_name}")
            lines.append(f"            (:target {target})")
            
            # Ranges
            if region_info.get("ranges"):
                lines.append("            (:ranges (")
                for range_coords in region_info["ranges"]:
                    coords_str = " ".join(str(c) for c in range_coords)
                    lines.append(f"                ({coords_str})")
                lines.append("            ))")
            
            # Yaw rotation
            if region_info.get("yaw_rotation"):
                yaw = region_info["yaw_rotation"]
                if yaw != [0, 0] and yaw != [0.0, 0.0]:
                    lines.append("            (:yaw_rotation (")
                    yaw_str = " ".join(str(y) for y in yaw)
                    lines.append(f"                ({yaw_str})")
                    lines.append("            ))")
            
            lines.append("        )")
        lines.append("    )")
    
    # Fixtures
    if parsed_problem.get("fixtures"):
        fixture_parts = []
        for fixture_type, fixture_list in parsed_problem["fixtures"].items():
            for fixture_name in fixture_list:
                fixture_parts.append(fixture_name)
            fixture_parts.append("-")
            fixture_parts.append(fixture_type)
        lines.append(f"    (:fixtures {' '.join(fixture_parts)})")
    
    # Objects
    if parsed_problem.get("objects"):
        object_parts = []
        for obj_type, obj_list in parsed_problem["objects"].items():
            for obj_name in obj_list:
                object_parts.append(obj_name)
            object_parts.append("-")
            object_parts.append(obj_type)
        lines.append(f"    (:objects {' '.join(object_parts)})")
    
    # Objects of interest
    if parsed_problem.get("obj_of_interest"):
        obj_interest_str = " ".join(parsed_problem["obj_of_interest"])
        lines.append(f"    (:obj_of_interest {obj_interest_str})")
    
    # Initial state
    if parsed_problem.get("initial_state"):
        lines.append("    (:init")
        for state in parsed_problem["initial_state"]:
            if isinstance(state, list):
                state_str = " ".join(str(s) for s in state)
                lines.append(f"        ({state_str})")
            else:
                lines.append(f"        {state}")
        lines.append("    )")
    
    # Goal state
    if parsed_problem.get("goal_state"):
        lines.append("    (:goal")
        lines.append("        (and")
        
        def format_goal(goal_item, indent=12):
            goal_lines = []
            if isinstance(goal_item, dict):
                for key, value in goal_item.items():
                    if isinstance(value, list):
                        for v in value:
                            goal_lines.extend(format_goal(v, indent))
                    elif isinstance(value, dict):
                        goal_lines.extend(format_goal(value, indent))
                    else:
                        goal_lines.append(" " * indent + f"({key} {value})")
            elif isinstance(goal_item, list):
                item_str = " ".join(str(x) for x in goal_item)
                goal_lines.append(" " * indent + f"({item_str})")
            return goal_lines
        
        for goal in parsed_problem["goal_state"]:
            lines.extend(format_goal(goal))
        
        lines.append("        )")
        lines.append("    )")
    
    # 结束
    lines.append(")")
    
    return "\n".join(lines)


def create_env_from_parsed_problem(parsed_problem, camera_heights=128, camera_widths=128, **kwargs):
    """
    从 parsed_problem 直接创建环境
    
    Args:
        parsed_problem: robosuite_parse_problem() 返回的字典（可能已经被修改）
        camera_heights: 相机高度
        camera_widths: 相机宽度
        **kwargs: 传递给 OffScreenRenderEnv 的其他参数
        
    Returns:
        创建好的环境对象
        
    Example:
        >>> parsed = BDDLUtils.robosuite_parse_problem("task.bddl")
        >>> # 修改 parsed 字典...
        >>> env = BDDLUtils.create_env_from_parsed_problem(parsed)
        >>> obs = env.reset()
    """

    from libero.libero.envs import OffScreenRenderEnv
    # 生成 BDDL 内容
    bddl_content = generate_bddl_from_parsed_problem(parsed_problem)
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bddl', delete=False) as f:
        f.write(bddl_content)
        temp_bddl_file = f.name
    
    # 创建环境参数
    env_args = {
        "bddl_file_name": temp_bddl_file,
        "camera_heights": camera_heights,
        "camera_widths": camera_widths,
        **kwargs
    }
    
    # 创建环境
    env = OffScreenRenderEnv(**env_args)
    
    # 保存临时文件路径，以便后续删除
    setattr(env, '_temp_bddl_file', temp_bddl_file)
    
    return env
