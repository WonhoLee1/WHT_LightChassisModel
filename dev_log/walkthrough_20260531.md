# Walkthrough: WHTVisualizer Query Tools & Bead Height Warping

본 문서는 `WHTVisualizer`에 새롭게 탑재된 실시간 데이터 조회(Query) 인터랙션 기능과 `Bead Height` 3D Warping(입체적 돌출) 연동 기능의 구현 내역 및 테스트 검증 결과를 정리한 보고서입니다.

## 1. 구현된 주요 기능 요약

### 🔍 A. 대화형 데이터 조회 (Query Tools)
- **체크박스 제어**: `Enable Query` 체크박스가 활성화되었을 때만 Ray-casting 및 마우스 picking 로직이 동작하여 기본 뷰 조작 성능에 불필요한 연산 오버헤드를 발생시키지 않습니다.
- **Node / Element 대상 선택**: QRadioButton을 통해 현재 호버링 또는 더블클릭할 대상(Node 혹은 Element)을 직관적으로 변경할 수 있습니다.
- **실시간 Hover 쿼리**: 마우스를 메쉬 위로 이동하면 `vtkCellPicker` 레이캐스터가 3D 타겟을 실시간으로 추적하여 전역 ID(`vtkOriginalPointIds`/`vtkOriginalCellIds`로 역추적) 및 scalar 값을 QMainWindow의 **하단 상태바(StatusBar)**에 깔끔하게 표출합니다.
- **3D 텍스트 라벨 (더블클릭)**: 메쉬 표면을 더블클릭할 때 충돌 지점 좌표 위치에 `plotter.add_point_labels`를 활용하여 **8pt 크기의 텍스트 값 라벨**(`N[ID]: [Value]` 혹은 `E[ID]: [Value]`)을 3D 상에 입체적으로 고정 생성하고 actor를 캐싱합니다.
- **라벨 일괄 소거**: `Clear Labels` 버튼 클릭 시 화면 상에 생성된 모든 3D 텍스트 라벨을 `plotter.remove_actor(name)`로 흔적 없이 일괄 소거하고 캐시를 초기화합니다.

### 🌊 B. Bead Height 3D Warping 지원
- **스칼라 필드 예외 노출**: `Bead_Height` 필드가 감지되면(point_data 또는 cell_data) 스칼라 필드임에도 Deform Vector 콤보박스(`combo_warp_vec`)의 변형 물리량 후보군에 강제 등록되도록 필터를 우회하였습니다.
- **물리적 Z축 음수 Warping**: Warping이 활성화되고 `Bead_Height` 필드가 지정되면, `Bead_Height * scale` 만큼 draw-dir 방향인 Z축 음의 방향 `[0, 0, -1]`으로 가상의 3D 변위 벡터 `disp = [0, 0, -Bead_Height]`를 실시간 생성하여 `_warp_pts`에 대입합니다. (cell_data에만 존재할 경우 `cell_data_to_point_data`를 수행해 절점값으로 보간하여 안전하게 처리합니다.)

---

## 2. 자동화 테스트 검증 결과 (Regression Test)

WHT Solver의 평판 정적 굽힘, 뒤틀림, 막인장 및 다자유도 고유주파수 고속 연산에 대한 **22개 패치 테스트 및 검증 정합성**이 무결하게 통과(`Passed: 22 | Failed: 0`)함을 최종 검증하였습니다.

```
==============================================================================================================
                                      [WHT] Solver Verification Results                                      
==============================================================================================================
Test Case                 | Element  | Quantity             |       Theory |          FEM |   Error% | Result  
--------------------------------------------------------------------------------------------------------------
3-pt Bending              | QUAD4    | Max Deflection       |        1.488 |        1.486 |     0.17% | PASS    
3-pt Bending              | QUAD4    | Max Stress (Sx)      |          375 |        358.2 |     4.48% | PASS    
4-pt Bending              | QUAD4    | Max Deflection       |        10.14 |        9.924 |     2.14% | PASS    
Plate Twisting            | QUAD4    | Corner Deflection    |        37.14 |        37.19 |     0.14% | PASS    
Natural Frequency         | QUAD4    | Mode 1 (1,1) [Hz]    |        49.17 |        49.04 |     0.27% | PASS    
Natural Frequency         | QUAD4    | Mode 2 (1,2) [Hz]    |        122.9 |        122.7 |     0.18% | PASS    
Natural Frequency         | QUAD4    | Mode 3 (2,1) [Hz]    |        122.9 |        122.7 |     0.18% | PASS    
Natural Frequency         | QUAD4    | Mode 4 (2,2) [Hz]    |        196.7 |        195.2 |     0.78% | PASS    
Natural Frequency         | QUAD4    | Mode 5 (1,3) [Hz]    |        245.9 |        246.4 |     0.22% | PASS    
Membrane Tension          | QUAD4    | Max Displacement X   |      0.04762 |      0.04762 |     0.00% | PASS    
Membrane Tension          | QUAD4    | Avg Stress Sx        |          100 |          100 |     0.00% | PASS    
3-pt Bending              | TRIA3    | Max Deflection       |        1.488 |        1.486 |     0.13% | PASS    
3-pt Bending              | TRIA3    | Max Stress (Sx)      |          375 |        358.4 |     4.41% | PASS    
4-pt Bending              | TRIA3    | Max Deflection       |        10.14 |        9.924 |     2.14% | PASS    
Plate Twisting            | TRIA3    | Corner Deflection    |        37.14 |        37.96 |     2.21% | PASS    
Natural Frequency         | TRIA3    | Mode 1 (1,1) [Hz]    |        49.17 |        49.28 |     0.22% | PASS    
Natural Frequency         | TRIA3    | Mode 2 (1,2) [Hz]    |        122.9 |        123.9 |     0.77% | PASS    
Natural Frequency         | TRIA3    | Mode 3 (2,1) [Hz]    |        122.9 |        123.9 |     0.77% | PASS    
Natural Frequency         | TRIA3    | Mode 4 (2,2) [Hz]    |        196.7 |        200.1 |     1.72% | PASS    
Natural Frequency         | TRIA3    | Mode 5 (1,3) [Hz]    |        245.9 |        249.1 |     1.33% | PASS    
Membrane Tension          | TRIA3    | Max Displacement X   |      0.04762 |      0.04814 |     1.10% | PASS    
Membrane Tension          | TRIA3    | Avg Stress Sx        |          100 |        100.2 |     0.24% | PASS    
--------------------------------------------------------------------------------------------------------------
 Total: 22 | Passed: 22 | Failed: 0
==============================================================================================================
```

---

## 3. 사용 안내 및 상세 가이드

1. **Bead Height Warping 연동 방법**:
   - `WHTInspector` (오른쪽 패널) -> `Properties` 탭 상단 `Deform` 그룹의 콤보박스에서 `Bead_Height` 필드를 선택합니다.
   - `Use Deform` 체크박스를 체크한 후 우측의 수치 스피너나 슬라이더를 통해 변형 배율(`scale`)을 올리면, 성형 비드가 Z축 음의 방향 `[0, 0, -Bead_Height]`으로 리얼하고 정밀하게 돌출(부풀어오름)되어 시각화됩니다.

2. **데이터 Query 쿼리 활용법**:
   - `Properties` 탭 내에 마련된 `Query Tools` 그룹에서 `Enable Query` 체크박스를 활성화합니다.
   - 조회 대상을 절점(`Node Value`) 또는 요소(`Element Value`)로 선택합니다.
   - 3D 뷰어 화면의 임의의 부분 위로 마우스를 올려 가져다 대면, 하단 **상태바(StatusBar)**에 `[Part: ...] Node: [Global ID] | Coord: (...) | Value: [값]` 형태로 실시간 조회가 완료됩니다.
   - 특정 지점의 값을 3D 뷰어 상에 영구 고정하고 싶다면 해당 위치를 **더블클릭**합니다. 화면 상에 8pt 크기의 텍스트 값 라벨(Cyan 색상 및 Magenta 포인트)이 입체 생성됩니다.
   - 라벨을 모두 소거하고 싶을 때는 `Clear Labels` QPushButton을 클릭하여 일괄적으로 지울 수 있습니다.
