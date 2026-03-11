import os
import json
import tarfile
import shutil
from pathlib import Path
import polars as pl
from tqdm import tqdm

def extract_tar_file(tar_path, extract_dir):
    """
    解压 tar.gz 文件到指定目录
    
    Args:
        tar_path: tar.gz 文件路径
        extract_dir: 解压目标目录
    """
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Tar 文件不存在: {tar_path}")
    
    print(f"正在解压 {tar_path} 到 {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    # 检查是否已经解压过
    # thingi10k 的 npz variant 解压后应该在 extract_dir/npz/ 目录下
    npz_dir = os.path.join(extract_dir, 'npz')
    if os.path.exists(npz_dir) and len([f for f in os.listdir(npz_dir) if f.endswith('.npz')]) > 0:
        print(f"检测到已解压的文件（找到 {len([f for f in os.listdir(npz_dir) if f.endswith('.npz')])} 个 .npz 文件），跳过解压步骤。")
        return extract_dir
    
    # 解压文件
    with tarfile.open(tar_path, 'r:gz') as tar:
        # 显示进度
        members = tar.getmembers()
        for member in tqdm(members, desc="解压文件"):
            tar.extract(member, extract_dir)
    
    # 检查解压后的目录结构
    # 查找 npz 文件或 npz 目录
    npz_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    
    if npz_files:
        print(f"找到 {len(npz_files)} 个 .npz 文件")
        # 如果文件不在 npz 子目录中，可能需要重新组织
        if not os.path.exists(npz_dir):
            # 检查文件是否在根目录或其他目录
            sample_file = npz_files[0]
            sample_dir = os.path.dirname(sample_file)
            if sample_dir != npz_dir:
                print(f"检测到文件在 {sample_dir}，创建 npz 目录结构...")
                os.makedirs(npz_dir, exist_ok=True)
                # 移动或链接文件到 npz 目录
                for npz_file in npz_files:
                    filename = os.path.basename(npz_file)
                    target = os.path.join(npz_dir, filename)
                    if not os.path.exists(target):
                        if os.path.exists(npz_file):
                            shutil.move(npz_file, target)
    
    print(f"解压完成！")
    return extract_dir

def process_with_local_metadata(metadata_dir, npz_dir, output_dir, min_facets=5000, variant='npz'):
    """
    直接使用本地 CSV 文件处理数据集，不依赖 thingi10k 的下载机制
    
    Args:
        metadata_dir: CSV 文件所在目录
        npz_dir: npz 文件所在目录
        output_dir: 输出目录
        min_facets: 最小面数阈值
        variant: 数据集变体
    """
    print("使用本地 CSV 文件处理数据集...")
    
    # 确定 geometry_data 文件名
    if variant == 'tetwild':
        geometry_file = os.path.join(metadata_dir, 'tetwild_geometry_data.csv')
    else:
        geometry_file = os.path.join(metadata_dir, 'geometry_data.csv')
    
    contextual_file = os.path.join(metadata_dir, 'contextual_data.csv')
    summary_file = os.path.join(metadata_dir, 'input_summary.csv')
    tag_file = os.path.join(metadata_dir, 'tag_data.csv')
    
    # 检查文件是否存在
    required_files = [geometry_file, contextual_file, summary_file, tag_file]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"必需的 CSV 文件不存在: {f}")
    
    # 读取几何数据
    print("读取 geometry_data.csv...")
    geometry_schema = {
        "file_id": pl.Int32,
        "num_vertices": pl.Int32,
        "num_faces": pl.Int32,
        "num_geometrical_degenerated_faces": pl.Int32,
        "num_combinatorial_degenerated_faces": pl.Int32,
        "num_connected_components": pl.Int32,
        "num_boundary_edges": pl.Int32,
        "num_duplicated_faces": pl.Int32,
        "euler_characteristic": pl.Int32,
        "num_self_intersections": pl.Int32,
        "num_coplanar_intersecting_faces": pl.Int32,
        "vertex_manifold": pl.Int32,
        "edge_manifold": pl.Int32,
        "oriented": pl.Int32,
        "total_area": pl.Float64,
        "min_area": pl.Float64,
        "p25_area": pl.Float64,
        "median_area": pl.Float64,
        "p75_area": pl.Float64,
        "p90_area": pl.Float64,
        "p95_area": pl.Float64,
        "max_area": pl.Float64,
        "min_valance": pl.Int32,
        "p25_valance": pl.Int32,
        "median_valance": pl.Int32,
        "p75_valance": pl.Int32,
        "p90_valance": pl.Int32,
        "p95_valance": pl.Int32,
        "max_valance": pl.Int32,
        "min_dihedral_angle": pl.Float64,
        "p25_dihedral_angle": pl.Float64,
        "median_dihedral_angle": pl.Float64,
        "p75_dihedral_angle": pl.Float64,
        "p90_dihedral_angle": pl.Float64,
        "p95_dihedral_angle": pl.Float64,
        "max_dihedral_angle": pl.Float64,
        "min_aspect_ratio": pl.Float64,
        "p25_aspect_ratio": pl.Float64,
        "median_aspect_ratio": pl.Float64,
        "p75_aspect_ratio": pl.Float64,
        "p90_aspect_ratio": pl.Float64,
        "p95_aspect_ratio": pl.Float64,
        "max_aspect_ratio": pl.Float64,
        "PWN": pl.Int32,
        "solid": pl.Int32,
        "ave_area": pl.Float64,
        "ave_valance": pl.Float64,
        "ave_dihedral_angle": pl.Float64,
        "ave_aspect_ratio": pl.Float64,
    }
    
    geometry_df = pl.read_csv(geometry_file, schema_overrides=geometry_schema, ignore_errors=True)
    if "self_intersecting" not in geometry_df.columns:
        geometry_df = geometry_df.with_columns(
            (pl.col("num_self_intersections") > 0).cast(pl.Boolean).alias("self_intersecting")
        )
    
    # 读取上下文数据
    print("读取 contextual_data.csv...")
    contextual_schema = {
        "Thing ID": pl.Int32,
        "Date": pl.Datetime,
        "Category": pl.String,
        "Sub-category": pl.String,
        "Name": pl.String,
        "Author": pl.String,
        "License": pl.String,
    }
    contextual_df = pl.read_csv(contextual_file, schema_overrides=contextual_schema, ignore_errors=True)
    
    # 读取摘要数据
    print("读取 input_summary.csv...")
    summary_schema = {
        "ID": pl.Int32,
        "Thing ID": pl.Int32,
    }
    summary_df = pl.read_csv(summary_file, schema_overrides=summary_schema, ignore_errors=True)
    
    # 读取标签数据
    print("读取 tag_data.csv...")
    tag_schema = {
        "Thing ID": pl.Int32,
        "Tag": pl.String,
    }
    tag_df = pl.read_csv(tag_file, schema_overrides=tag_schema, ignore_errors=True)
    
    # 合并数据
    print("合并数据...")
    # 合并 geometry 和 summary
    df = geometry_df.join(summary_df, left_on="file_id", right_on="ID", how="left")
    # 合并上下文数据
    df = df.join(contextual_df, on="Thing ID", how="left")
    # 合并标签数据
    tag_agg = tag_df.group_by("Thing ID").agg(pl.col("Tag").alias("Tags"))
    df = df.join(tag_agg, on="Thing ID", how="left")
    
    # 填充空值
    fill_exprs = []
    if "License" in df.columns:
        fill_exprs.append(pl.col("License").fill_null("unknown"))
    if "Author" in df.columns:
        fill_exprs.append(pl.col("Author").fill_null("unknown"))
    if "Category" in df.columns:
        fill_exprs.append(pl.col("Category").fill_null("unknown"))
    if "Sub-category" in df.columns:
        fill_exprs.append(pl.col("Sub-category").fill_null("unknown"))
    if "Name" in df.columns:
        fill_exprs.append(pl.col("Name").fill_null("unknown"))
    if "Tags" in df.columns:
        fill_exprs.append(pl.col("Tags").fill_null(pl.lit([])))
    
    if fill_exprs:
        df = df.with_columns(fill_exprs)
    
    # 筛选面数 >= min_facets 的记录
    print(f"筛选面数 >= {min_facets} 的模型...")
    filtered_df = df.filter(pl.col("num_faces") >= min_facets)
    
    print(f"找到 {len(filtered_df)} 个符合条件的模型")
    
    # 转换为字典列表
    filtered_entries = []
    missing_files = 0
    
    for row in tqdm(filtered_df.iter_rows(named=True), total=len(filtered_df), desc="处理模型"):
        file_id = row["file_id"]
        # 构建文件路径
        npz_file = os.path.join(npz_dir, f"{file_id}.npz")
        
        # 检查文件是否存在
        if not os.path.exists(npz_file):
            # 尝试在整个解压目录中查找文件
            found = False
            search_root = os.path.dirname(npz_dir) if os.path.exists(os.path.dirname(npz_dir)) else npz_dir
            for root, dirs, files in os.walk(search_root):
                if f"{file_id}.npz" in files:
                    npz_file = os.path.join(root, f"{file_id}.npz")
                    found = True
                    break
            if not found:
                missing_files += 1
                # 只在调试时打印警告，避免输出过多
                if missing_files <= 10:
                    print(f"警告: 文件不存在 {npz_file}, file_id={file_id}")
                continue
        
        # 处理标签
        tags = row.get("Tags", [])
        if tags is None:
            tags = []
        elif isinstance(tags, list) and len(tags) > 0 and isinstance(tags[0], str):
            tags = tags
        elif isinstance(tags, str):
            tags = [tags]
        else:
            tags = []
        
        filtered_entries.append({
            'file_id': file_id,
            'thing_id': row.get("Thing ID"),
            'file_path': npz_file,
            'author': row.get("Author", "unknown"),
            'license': row.get("License", "unknown"),
            'category': row.get("Category", "unknown"),
            'subcategory': row.get("Sub-category", "unknown"),
            'name': row.get("Name", "unknown"),
            'tags': tags,
            'num_vertices': row.get("num_vertices"),
            'num_facets': row.get("num_faces"),  # num_faces 就是 num_facets
            'num_components': row.get("num_connected_components"),
            'num_boundary_edges': row.get("num_boundary_edges"),
            'closed': row.get("num_boundary_edges", 1) == 0,
            'solid': row.get("solid", 0) == 1,
            'vertex_manifold': row.get("vertex_manifold", 0) == 1,
            'edge_manifold': row.get("edge_manifold", 0) == 1,
            'oriented': row.get("oriented", 0) == 1,
            'self_intersecting': row.get("self_intersecting", False),
            'euler': row.get("euler_characteristic"),
        })
    
    if missing_files > 0:
        print(f"警告: 共有 {missing_files} 个模型的文件未找到")
    
    # 保存筛选结果的信息到 JSON 文件
    info_file = os.path.join(output_dir, 'filtered_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump({
            'min_facets': min_facets,
            'total_count': len(filtered_entries),
            'variant': variant,
            'metadata_dir': metadata_dir,
            'npz_dir': npz_dir,
            'missing_files': missing_files,
            'entries': filtered_entries
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n筛选完成！")
    print(f"符合条件的模型数量: {len(filtered_entries)}")
    print(f"模型信息已保存到: {info_file}")
    print(f"数据目录: {output_dir}")
    
    return filtered_entries

def process_thingi10k_from_tar(tar_path, output_dir, min_facets=5000, variant='npz'):
    """
    从本地 tar 文件处理 thingi10k 数据集，筛选面数超过指定阈值的模型
    
    Args:
        tar_path: 本地 tar.gz 文件路径
        output_dir: 数据保存目录
        min_facets: 最小面数阈值，默认 5000
        variant: 数据集变体，'npz', 'raw', 或 'tetwild'
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 解压 tar 文件
    extract_dir = os.path.join(output_dir, 'extracted')
    extract_tar_file(tar_path, extract_dir)
    
    # 检查解压后的目录结构
    # 根据 thingi10k 的代码，npz variant 应该在 extract_dir/npz/ 目录下
    npz_dir = os.path.join(extract_dir, 'npz')
    if not os.path.exists(npz_dir):
        # 尝试查找可能的目录结构
        print("检查解压后的目录结构...")
        npz_files_found = []
        for root, dirs, files in os.walk(extract_dir):
            if 'npz' in dirs:
                npz_dir = os.path.join(root, 'npz')
                print(f"找到 npz 目录: {npz_dir}")
                break
            # 检查是否有 .npz 文件
            npz_files = [f for f in files if f.endswith('.npz')]
            if npz_files:
                npz_files_found.append(root)
        
        # 如果找到了 npz 文件但不在 npz 目录中
        if npz_files_found and not os.path.exists(npz_dir):
            # 使用第一个包含 npz 文件的目录
            npz_dir = npz_files_found[0]
            print(f"在 {npz_dir} 找到 .npz 文件，使用此目录")
    
    # 检查本地 metadata 目录
    metadata_dir = os.path.join(output_dir, 'metadata')
    if os.path.exists(metadata_dir):
        # 检查必需的 CSV 文件是否存在
        required_files = [
            'contextual_data.csv',
            'input_summary.csv',
            'tag_data.csv',
        ]
        if variant == 'tetwild':
            required_files.append('tetwild_geometry_data.csv')
        else:
            required_files.append('geometry_data.csv')
        
        all_files_exist = all(os.path.exists(os.path.join(metadata_dir, f)) for f in required_files)
        
        if all_files_exist:
            print(f"✓ 找到本地 metadata 文件在: {metadata_dir}")
            print("使用本地 CSV 文件处理数据集...")
            return process_with_local_metadata(metadata_dir, npz_dir, output_dir, min_facets, variant)
        else:
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(metadata_dir, f))]
            print(f"警告: metadata 目录存在但缺少以下文件: {missing_files}")
            print("尝试使用 thingi10k 库下载元数据（需要网络连接）...")
    else:
        print(f"警告: 未找到本地 metadata 目录: {metadata_dir}")
        print("尝试使用 thingi10k 库下载元数据（需要网络连接）...")
    
    # 如果本地没有 metadata，尝试使用 thingi10k 库（需要网络）
import thingi10k
    print(f"初始化 thingi10k 数据集，缓存目录: {output_dir}")
    print(f"使用 variant: {variant}")
    
    thingi10k.init(variant=variant, cache_dir=output_dir, force_redownload=False)
    
    # 筛选面数超过 min_facets 的模型
    print(f"筛选面数 >= {min_facets} 的模型...")
    
    filtered_entries = []
    filtered_count = 0
    
    # 遍历筛选后的数据集
    try:
        for entry in tqdm(thingi10k.dataset(num_facets=(min_facets, None)), desc="处理模型"):
            # 检查文件路径是否存在
            file_path = entry.get('file_path')
            if file_path and not os.path.exists(file_path):
                # 尝试在解压目录中查找文件
                file_id = entry.get('file_id')
                if file_id:
                    # 尝试多个可能的路径
                    possible_paths = [
                        os.path.join(extract_dir, 'npz', f"{file_id}.npz"),
                        os.path.join(npz_dir, f"{file_id}.npz"),
                        file_path,
                    ]
                    for pp in possible_paths:
                        if os.path.exists(pp):
                            file_path = pp
                            break
                
                if not os.path.exists(file_path):
                    print(f"警告: 文件不存在 {file_path}, file_id={file_id}")
                    continue
            
            filtered_entries.append({
                'file_id': entry.get('file_id'),
                'thing_id': entry.get('thing_id'),
                'file_path': str(file_path) if file_path else None,
                'author': entry.get('author'),
                'license': entry.get('license'),
                'category': entry.get('category'),
                'subcategory': entry.get('subcategory'),
                'name': entry.get('name'),
                'num_vertices': entry.get('num_vertices'),
                'num_facets': entry.get('num_facets'),
                'num_components': entry.get('num_components'),
                'num_boundary_edges': entry.get('num_boundary_edges'),
                'closed': entry.get('closed'),
                'solid': entry.get('solid'),
                'vertex_manifold': entry.get('vertex_manifold'),
                'edge_manifold': entry.get('edge_manifold'),
                'oriented': entry.get('oriented'),
            })
            filtered_count += 1
    except Exception as e:
        print(f"处理数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 保存筛选结果的信息到 JSON 文件
    info_file = os.path.join(output_dir, 'filtered_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump({
            'min_facets': min_facets,
            'total_count': filtered_count,
            'variant': variant,
            'tar_path': tar_path,
            'entries': filtered_entries
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n筛选完成！")
    print(f"符合条件的模型数量: {filtered_count}")
    print(f"模型信息已保存到: {info_file}")
    print(f"数据目录: {output_dir}")
    
    return filtered_entries

if __name__ == '__main__':
    # 设置路径
    output_dir = '/mnt/lab/data/yeruisi/data/compression/thingi10k'
    tar_path = '/mnt/lab/data/yeruisi/data/compression/thingi10k/Thingi10K-002.tar.gz'
    
    # 从本地 tar 文件处理并筛选面数超过 5000 的模型
    process_thingi10k_from_tar(tar_path, output_dir, min_facets=5000, variant='npz') 