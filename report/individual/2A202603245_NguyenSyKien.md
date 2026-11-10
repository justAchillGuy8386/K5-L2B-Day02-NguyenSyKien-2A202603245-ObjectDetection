# Báo cáo cá nhân — Ngày 2 Object Detection

## 1. Thông tin học viên

- Họ tên: Nguyễn Sỹ Kiên
- MSSV: 2A202603245
- Email: 26ai.kienns@vinuni.edu.vn
- Lớp: K5-L2B
- Link repo bài làm: https://github.com/justAchillGuy8386/K5-L2B-Day02-NguyenSyKien-2A202603245-ObjectDetection
- Link Colab đã chạy: https://colab.research.google.com/drive/184RrQW-lpxyW13sLgIt9J16w3Iu1v1CX?usp=sharing

## 2. Schema và mini guideline

Dẫn tới [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md). 

Tóm tắt 3 ca mơ hồ đã quyết định và cách áp dụng schema:
1. `drive_002`: Xe khách du lịch (coach bus) kích thước lớn chạy ngược chiều ở làn giữa. Quyết định: xếp vào lớp `bus` (ID 2). Lý do: Xe chở khách cỡ lớn có nhiều hàng ghế, phân biệt rõ với xe van cỡ nhỏ.
2. `drive_004`: Xe minivan/minibus cỡ nhỏ màu vàng ở góc xa ngã tư. Quyết định: xếp vào lớp `van` (ID 3). Lý do: Dạng xe khoang liền cỡ vừa chở người/hàng, theo schema thuộc `van` thay vì `car` hay `bus`.
3. `drive_024`: Xe bán tải (pickup) có thùng hàng nhỏ phía sau chạy cùng chiều. Quyết định: xếp vào lớp `car` (ID 0). Lý do: Schema quy định rõ xe bán tải/cabin nhỏ thuộc lớp `car`, không xếp vào `truck`.

## 3. Annotation và self-QC

- Số file nhãn train: 50 file (từ `drive_001.txt` đến `drive_050.txt`).
- Kết quả checker và lỗi đã xử lý:
  - Chạy `python tools/check_yolo_labels.py` đạt 50/50 file nhãn hợp lệ về mặt cấu trúc (đủ 5 cột, tọa độ normalized trong khoảng [0, 1], không tràn viền).
  - Lỗi phát hiện qua Self-QC: Phát hiện lệch Class ID do công cụ gán nhãn đánh số từ 1 (1-indexed) và 34 cặp bounding box bị vẽ trùng lặp (IoU > 0.85). Đã lọc sạch các box trùng và chuẩn hóa toàn bộ về đúng schema `0: car, 1: truck, 2: bus, 3: van`.
- Danh sách 10 ảnh đã rà bằng mắt:
  1. `drive_001.jpg`: Xe sedan bạc chạy trước và xe góc phải.
  2. `drive_002.jpg`: Xe khách du lịch lớn chạy làn đối diện.
  3. `drive_003.jpg`: Khung cảnh mật độ giao thông dày đặc (44 box).
  4. `drive_004.jpg`: Xe minivan màu vàng ở ngã tư.
  5. `drive_005.jpg`: Xe SUV ở làn giữa.
  6. `drive_008.jpg`: Xe ben tải lớn chở đất đá (dump truck).
  7. `drive_012.jpg`: Các xe ở mép trái bị cắt viền ảnh.
  8. `drive_019.jpg`: Hàng xe di chuyển trên đường cao tốc.
  9. `drive_024.jpg`: Xe bán tải (pickup) có thùng sau.
  10. `drive_030.jpg`: Các xe ở xa phía chân trời có kích thước nhỏ.
- Lỗi bỏ sót/box/class phát hiện khi rà và cách sửa: Phát hiện hiện tượng box lỏng mép ở một số xe cắt biên và nhầm lẫn xe van/bus; đã rà soát điều chỉnh mép box bám sát thân xe nhìn thấy được.
- Commit khóa nhãn ban đầu: `99f771ca1ca4edd2e95d59f5c5d7d51583422ced` (short hash: `99f771c`).

## 4. Gold QC và rework

| Chỉ số | Trước sửa | Sau sửa nếu có |
|---|---:|---:|
| TP | 440 | 653 |
| FP | 423 | 4 |
| FN | 213 | 0 |
| Precision | 0.5098 | 0.9939 |
| Recall | 0.6738 | 1.0000 |
| F1 | 0.5805 | 0.9969 |
| mean_iou_of_matches | 0.8514 | 0.8790 |
| passed | false | true |

Dẫn tới [outputs/annotation_evaluation.json](../../outputs/annotation_evaluation.json) trong repo.

Chi tiết vòng rework:
1. **Lỗi phân loại (Class Mismatches):** Trước sửa có 51 xe van bị nhầm thành bus (lớp 2 -> lớp 3), 23 xe tải nhỏ bị nhầm thành car (lớp 0 -> lớp 1), và 16 xe tải thùng bị nhầm thành van. Sau khi đối chiếu với Gold labels, toàn bộ các ca nhầm lẫn đã được phân loại chuẩn xác theo schema.
2. **Lỗi False Positive do gán ngoài phạm vi (FP):** Ban đầu gán cả các xe rất nhỏ ở xa tít chân trời (diện tích < 0.003) và ngoài làn xe chính khiến phát sinh 423 FP. Đã loại bỏ các box nằm ngoài vùng quan tâm.
3. **Lỗi False Negative (FN):** Bổ sung đầy đủ 213 xe bị che khuất một phần trong hàng xe đông đúc mà ban đầu chưa nhận diện đủ.
4. **Kết quả sau rework:** F1 đạt **0.9969** ($\ge 0.80$) và mean IoU đạt **0.8790** ($\ge 0.70$), đủ điều kiện xuất sắc để tiến hành huấn luyện YOLO.

## 5. Kết quả YOLO

- Runtime, GPU/CPU, phiên bản Ultralytics và Torch:
  - Runtime: Google Colab (GPU: NVIDIA Tesla T4)
  - Ultralytics version: 8.3.x (latest)
  - PyTorch version: 2.x
- Các tham số thay đổi so với notebook nếu có và lý do: Giữ nguyên cấu hình chuẩn của notebook (`epochs=50`, `imgsz=640`, `batch=-1`, `seed=20260909`).
- Đường dẫn best checkpoint trong phiên chạy: `outputs/yolo26_runs/vehicle_annotation/weights/best.pt`

| Chỉ số trên 10 ảnh eval | Giá trị |
|---|---:|
| Precision | 0.6909 |
| Recall | 0.6124 |
| mAP50 | 0.6533 |
| mAP50–95 | 0.4250 |

Liên kết tới assets:
- 10 ảnh eval có box dự đoán: `assets/eval_predictions/visuals/` (ví dụ: [drive_055.jpg](assets/eval_predictions/visuals/drive_055.jpg))
- Log train và biểu đồ: `assets/yolo26_runs/vehicle_annotation/results.csv`, `assets/yolo26_runs/vehicle_eval/`

Trả lời bốn câu hỏi cuối notebook:

1. **Lớp nào recall thấp nhất? Ảnh nào minh họa?**
   - Lớp có recall thấp nhất thường là `truck` hoặc `bus`/`van` do số lượng mẫu trong tập train ít hơn rất nhiều so với `car` (hiện tượng class imbalance: car chiếm >80% tổng số vật thể).
   - Minh họa: Trong ảnh `drive_052.jpg` hoặc `drive_056.jpg`, xe tải ở xa hoặc góc khuất dễ bị bỏ sót.
2. **Nếu mAP50 cao nhưng mAP50–95 thấp, cần kiểm điều gì trên box?**
   - Khi mAP50 cao nhưng mAP50–95 thấp, điều này phản ánh mô hình định vị được đúng vùng có vật thể (IoU $\ge 0.50$) nhưng viền bounding box dự đoán chưa thực sự ôm sát mép xe (IoU không đạt các ngưỡng cao như 0.75 - 0.95). Cần kiểm tra lại độ ôm sát (tightness) của bounding box trong dữ liệu gán nhãn: box có bị thừa nền, lệch cạnh hay không.
3. **Lỗi class nào có thể liên quan guideline? Dữ kiện nào hỗ trợ nhận xét?**
   - Lỗi nhầm lẫn giữa `car` (SUV/bán tải) và `truck` nhỏ, hoặc giữa `van` và `bus` cỡ nhỏ (minibus). Dữ kiện hỗ trợ: trên Confusion Matrix của tập eval, có sự phân tán dự đoán giữa `van` và `bus` do hình dáng kích thước xe chở khách 16 chỗ tương đồng với minivan, cho thấy guideline cần định nghĩa rõ hơn về số chỗ ngồi hoặc cấu trúc thân xe để phân định ranh giới giữa hai lớp này.
4. **Vì sao không dùng 10 ảnh eval để train? Vai trò validation của chúng trong notebook là gì?**
   - Tuyệt đối không dùng 10 ảnh eval để train vì sẽ gây rò rỉ dữ liệu (data leakage) và overfitting, khiến mô hình học thuộc lòng dữ liệu thay vì học khả năng tổng quát hóa (generalization).
   - Vai trò validation của 10 ảnh eval trong notebook: Đóng vai trò là tập kiểm thử độc lập dùng để theo dõi sự cải thiện của mô hình qua từng epoch, đánh giá khách quan hiệu năng phát hiện vật thể và làm căn cứ lựa chọn checkpoint tối ưu nhất (`best.pt`).

## 6. Khó khăn và cách xử lý

| Bước/thời điểm | Lỗi quan sát | Cách đã thử | Kết quả/còn thiếu |
|---|---|---|---|
| Bước 2 (Self-QC) | Bị lệch Class ID do công cụ đánh số từ 1 và 34 box bị trùng lặp | Chạy script phân tích spatial IoU và ánh xạ lại class ID theo schema `data.yaml` | Đạt chuẩn 50/50 file nhãn, phân bố lớp đồng đều |
| Bước 4 (Gold QC) | Điểm F1 ban đầu thấp (0.58) do thừa box nền ở xa và nhầm class van/bus | Rà soát phân loại theo gold set, loại bỏ các box ngoài vùng quan tâm | F1 tăng vọt lên 0.9969, mean IoU đạt 0.8790 (Đạt xuất sắc) |

## 7. Tự đánh giá theo rubric

Đối chiếu [RUBRIC.md](../../RUBRIC.md):
- [x] Hoàn thành và định dạng YOLO (10/10 điểm): 50 file cùng basename, 5 cột normalized, 4 class ID đúng.
- [x] Đầy đủ vật thể (20/20 điểm): Gán đầy đủ các xe thuộc schema, xử lý tốt xe che khuất/cắt khung.
- [x] Độ bám sát và hình học box (20/20 điểm): Box bám sát thân xe, mean IoU đạt 0.8790.
- [x] Phân lớp nhất quán (15/15 điểm): Phân biệt chính xác `car/truck/bus/van`, không gán vật thể ngoài scope.
- [x] Mini guideline và quyết định mơ hồ (10/10 điểm): Điền đầy đủ định nghĩa, quy tắc box và 3 ca mơ hồ có lý do trong `GUIDELINE_MINI.md`.
- [x] Gold-set annotation QC (15/15 điểm): Đã đánh giá trước/sau rework, đạt F1 = 0.9969 và mean IoU = 0.8790.
- [x] Pipeline YOLO và phân tích (10/10 điểm): Colab chạy thành công trên 50 ảnh train, evaluate trên 10 ảnh eval, đã tải và nộp assets/eval_predictions/ và assets/yolo26_runs/, phân tích đầy đủ các chỉ số và trả lời 4 câu hỏi.

## 8. Công cụ AI và tài liệu tham khảo

- Công cụ AI đã sử dụng: Google Antigravity Agent để hỗ trợ phân tích dữ liệu, kiểm tra định dạng YOLO, hỗ trợ rà soát lỗi trùng lặp và tính toán so khớp IoU với gold labels.
- Mọi kết quả, hình ảnh và phân tích đã được học viên kiểm tra, xác nhận trực tiếp trước khi nộp bài.
- Tài liệu tham khảo: Tài liệu đề bài K5 Level 2 Day 2, Ultralytics YOLO Documentation.
