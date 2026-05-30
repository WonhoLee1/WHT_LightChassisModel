# -*- coding: utf-8 -*-
"""
capture_modal_shapes.py
=======================
WHT Solver와 CalculiX의 고유 모드 해석 결과를 PyVista 오프스크린 캡쳐 및 Pillow Side-by-Side
이미지 결합 방식을 통해 모드 형상(Mode Shape)을 시각적으로 1:1 정밀 비교하는 배치 자동화 도구입니다.
"""

import os
import sys
import time
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add workspace to path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_solver import WHTSolver
from wht_converter.wht_models import WHTMetadata
from wht_visualizer.wht_visualizer import WHTVisualizer

# Add AutoCalculix to path
if "D:/PythonCodeStudy/AutoCalculix" not in sys.path:
    sys.path.append("D:/PythonCodeStudy/AutoCalculix")

from src.core.frd_converter import FrdToVtuConverter
from src.core.dat_parser import CalculixDatParser
import pyvista as pv

def draw_header_overlay(img: Image.Image, left_text: str, right_text: str) -> Image.Image:
    """
    Side-by-Side 이미지의 상단에 라벨 텍스트를 세련되게 렌더링하여 추가합니다.
    """
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # 폰트 로드 시도 (Arial Bold 등 Windows 기본 폰트 적용, 실패 시 기본 폰트 폴백)
    font = None
    font_paths = [
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\malgunbd.ttf",
        "C:\\Windows\\Fonts\\malgun.ttf",
        "C:\\Windows\\Fonts\\consola.ttf"
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 26)
                break
            except Exception:
                pass
    
    if font is None:
        font = ImageFont.load_default()

    # 상단에 반투명한 블랙 바 레이아웃 배치
    overlay_height = 60
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([0, 0, width, overlay_height], fill=(15, 15, 15, 220))
    img = Image.alpha_composite(img.convert('RGBA'), overlay)
    
    # 텍스트 다시 그리기
    draw = ImageDraw.Draw(img)
    
    # 왼쪽 라벨 (WHT Solver)
    draw.text((30, 15), left_text, fill=(255, 255, 255), font=font)
    
    # 오른쪽 라벨 (CalculiX)
    draw.text((width // 2 + 30, 15), right_text, fill=(255, 255, 255), font=font)
    
    # 중앙 구분선
    draw.line([width // 2, 0, width // 2, height], fill=(60, 60, 60), width=2)
    
    return img.convert('RGB')

def capture_single_model_mode(grid: pv.UnstructuredGrid, disp_array_name: str, 
                              scale_factor: float, output_path: str, title: str):
    """
    단일 모델의 모달 쉐입을 PyVista 오프스크린 플로터로 동일 뷰에서 정교하게 렌더링 후 캡쳐합니다.
    """
    # 1. 변형 적용 (Warp by vector)
    disp = grid.point_data[disp_array_name]
    grid.point_data["disp_vec"] = disp
    grid.point_data["disp_mag"] = np.linalg.norm(disp, axis=1)
    
    warped = grid.warp_by_vector("disp_vec", factor=scale_factor)
    
    # 2. PyVista Plotter 초기화 (오프스크린 및 strict paraview theme 준수)
    pv.set_plot_theme("dark")
    plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
    plotter.set_background('black')
    
    # 3. 메쉬 추가 (Rule-aligned: edge color darkgray, 1개 colorbar)
    plotter.add_mesh(
        warped, 
        scalars="disp_mag", 
        cmap="jet", 
        show_edges=True, 
        edge_color="darkgray",
        scalar_bar_args={
            "title": "Displacement Magnitude (Normalized)",
            "title_font_size": 12,
            "label_font_size": 10,
            "font_family": "arial",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.15,
            "height": 0.7,
            "width": 0.08,
            "color": "white"
        }
    )
    
    # 4. 좌표축 추가 (Rule-aligned)
    plotter.add_axes(color='white')
    
    # 5. 카메라 뷰 잠금 (ISO 및 전체 모델 바인딩 리셋)
    plotter.view_isometric()
    plotter.reset_camera()
    
    # 6. 렌더링 및 캡쳐 실행
    plotter.screenshot(output_path)
    plotter.close()

def main():
    print("="*80)
    print("   Starting Automated Batch Mode Shape Visualizer & Compare Pipeline")
    print("="*80)
    
    curr_dir = Path(__file__).resolve().parent
    mesh_inp = curr_dir / "ccx_iter016_Modal_Analysis_mesh.inp"
    master_inp = curr_dir / "ccx_iter016_Modal_Analysis.inp"
    frd_file = curr_dir / "ccx_iter016_Modal_Analysis.frd"
    ccx_dat_path = curr_dir / "ccx_iter016_Modal_Analysis.dat"
    
    # 디렉토리 생성
    image_dir = curr_dir / "results" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Init] Output images directory: {image_dir}")

    # 1. CalculiX FRD to VTU 변환
    print("\n[Step 1] Converting CalculiX FRD to VTU files...")
    converter = FrdToVtuConverter()
    converter.convert(frd_file)
    
    # VTU 파일들 매칭 리스트 확보
    vtu_files = sorted(list(curr_dir.glob("ccx_iter016_Modal_Analysis.*.vtu")))
    print(f"   -> Found {len(vtu_files)} CalculiX VTU mode shape files.")
    if len(vtu_files) == 0:
        print("[Error] CalculiX VTU 파일들이 정상적으로 생성되지 않았습니다.")
        return

    # 2. WHT Solver 실행
    print("\n[Step 2] Running WHT Solver to obtain exact Modal Results...")
    model = WHTMeshModel(name="Chassis_Benchmark")
    model.add_material(1, 210000.0, 0.3, 7.85e-9)
    model.add_property(1, "PSHELL", 0.6, 1) # 0.6mm thickness
    
    from benchmark_whtsolver_modal import load_ccx_mesh_into_wht
    load_ccx_mesh_into_wht(str(mesh_inp), model, default_pid=1)
    
    solver = WHTSolver(model)
    wht_res = solver.solve_modal(num_modes=20, method='auto', exclude_rigid_body=False)
    
    metadata = WHTMetadata(
        solver_name="WHTSolver",
        solver_version="0.1.0",
        analysis_type="modal",
        coordinate_system="cartesian",
        unit_length="mm",
        unit_force="N",
        unit_mass="tonne",
        unit_time="s"
    )
    wht_rd = wht_res.to_wht_result_data(metadata, model)
    print(f"   -> WHT Solver completed. Total {wht_res.frequencies.shape[0]} modes obtained.")

    # 3. CalculiX dat 주파수 파싱
    ccx_freqs = []
    if ccx_dat_path.exists():
        parser = CalculixDatParser()
        ccx_freqs = parser.extract_frequencies(ccx_dat_path)
        print(f"   -> Parsed {len(ccx_freqs)} CalculiX frequencies.")
    else:
        print("[Warning] CalculiX .dat 결과 파일을 찾을 수 없어 주파수 라벨이 제한적일 수 있습니다.")

    # 4. 모드 배치 루프 캡쳐 (탄성 모드 7차부터 15차까지 집중 캡쳐)
    # 1-based index 기준: Mode 7 ~ Mode 15
    target_modes = list(range(7, 16))
    
    # 모델의 Bounding Box Span 계산하여 일관된 Scale Factor 가중화
    nodes_np = wht_rd.nodes
    bb_min = np.min(nodes_np, axis=0)
    bb_max = np.max(nodes_np, axis=0)
    bb_span = np.linalg.norm(bb_max - bb_min)
    print(f"[Info] Model bounding box span: {bb_span:.2f} mm")

    print("\n[Step 3] Executing batch captures for Mode 7 ~ Mode 15...")
    temp_wht = curr_dir / "temp_wht.png"
    temp_ccx = curr_dir / "temp_ccx.png"

    for mode in target_modes:
        print(f" - Rendering and capturing Mode {mode}...")
        
        wht_freq = wht_res.frequencies[mode - 1]
        ccx_freq = ccx_freqs[mode - 1]["hz"] if len(ccx_freqs) >= mode else wht_freq
        
        # 4-1. WHT Solver 격자 및 변위 필드 구성
        wht_grid = WHTVisualizer._make_pv_grid(wht_rd.nodes, wht_rd.connectivity, wht_rd.offsets, wht_rd.cell_types)
        wht_disp = wht_rd.point_data["ModeShape"][mode - 1]
        wht_grid.point_data["U"] = wht_disp
        
        # 스케일 팩터 계산: 모델 스팬의 약 8%를 최대 모드 변형량으로 설정하여 가독성 최대화
        max_u = np.max(np.linalg.norm(wht_disp, axis=1))
        if max_u < 1e-8: max_u = 1.0
        scale_factor = (bb_span * 0.08) / max_u
        
        # WHT 캡쳐
        capture_single_model_mode(wht_grid, "U", scale_factor, str(temp_wht), f"WHT Solver Mode {mode}")
        
        # 4-2. CalculiX 변형 캡쳐
        ccx_vtu_path = curr_dir / f"ccx_iter016_Modal_Analysis.{mode:02d}.vtu"
        if not ccx_vtu_path.exists():
            # ccx2paraview의 포맷은 01, 02 등 두 자릿수 형식을 가짐
            ccx_vtu_path = curr_dir / f"ccx_iter016_Modal_Analysis.{mode:02d}.vtu"
            
        if not ccx_vtu_path.exists():
            print(f"   [Error] CalculiX VTU file not found for Mode {mode} at {ccx_vtu_path}")
            continue
            
        ccx_grid = pv.read(str(ccx_vtu_path))
        
        # CalculiX 캡쳐
        capture_single_model_mode(ccx_grid, "U", scale_factor, str(temp_ccx), f"CalculiX Mode {mode}")
        
        # 4-3. Side-by-Side 이미지 병합 및 텍스트 오버레이
        img_wht = Image.open(temp_wht)
        img_ccx = Image.open(temp_ccx)
        
        # 가로 결합 이미지 크기 정의
        combined_width = img_wht.width + img_ccx.width
        combined_height = img_wht.height
        
        combined_img = Image.new('RGB', (combined_width, combined_height))
        combined_img.paste(img_wht, (0, 0))
        combined_img.paste(img_ccx, (img_wht.width, 0))
        
        # 세련된 텍스트 헤더 라벨 오버레이 추가
        left_label = f"WHT Solver (ANDES+CS-DSG): Mode {mode} ({wht_freq:.2f} Hz)"
        right_label = f"CalculiX (S4 Shell Model): Mode {mode} ({ccx_freq:.2f} Hz)"
        
        final_img = draw_header_overlay(combined_img, left_label, right_label)
        
        # 최종 결과 이미지 저장
        output_image_path = image_dir / f"mode_{mode:02d}.png"
        final_img.save(output_image_path, "PNG")
        print(f"   -> Successfully saved Side-by-Side image: {output_image_path.name}")
        
    # 임시 파일 삭제 정리
    for temp_file in [temp_wht, temp_ccx]:
        if temp_file.exists():
            temp_file.unlink()

    print("\n" + "="*80)
    print("   Modal Batch Capture Completed Successfully!")
    print(f"   Results saved in: {image_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
