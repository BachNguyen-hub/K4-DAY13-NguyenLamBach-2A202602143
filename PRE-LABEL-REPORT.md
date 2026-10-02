# Báo cáo thực hành PointPillars — Day 13 (cá nhân)

Bài thực hành đã chuyển sang hình thức cá nhân theo yêu cầu hiện tại. Báo cáo dùng kết quả inference thật của gói Student KITTI và thông tin đọc được trên portal/CVAT. Lưu riêng để gửi LC, không đưa bản có danh tính lên repository public.

## 1. Người thực hiện và provenance

- Học viên: **Nguyễn Lâm Bách — 2A202602143**; tài khoản portal/CVAT đối chiếu: `2A202602143`.
- Ca trên portal: `k4-day13-async`.
- Trạng thái: **đã chạy inference thật trên máy cá nhân**, có output A/B/C và ba ca QC để đối chiếu; không phải chỉ đọc `provided-results`.
- Lần chạy dùng cho báo cáo: **02/10/2026, 16:27:25–16:28:26 (UTC+7)**, chuyển từ mốc UTC trong `smoke.json`. Lần chạy trước cùng ngày cũng thành công và được giữ để đối chiếu.
- Môi trường: Windows, Python **3.13.3**, Docker Desktop chạy **Linux containers, amd64/x86_64**. Docker Engine có 8 CPU và khoảng 8,18 GB RAM; runner giới hạn mỗi container **4 CPU/4 GB**, không dùng mạng, mount input/code chỉ đọc.
- Thư mục gói: `D:\Vin AI 2026\Day 13`.
- Thư mục bằng chứng chính: `D:\Vin AI 2026\ket-qua-Day13-20261002-02` (gọi là **OUT**). [smoke.json](../ket-qua-Day13-20261002-02/smoke.json) có **`status: passed`**, đủ A/B/C và ba ca QC.
- [Repository tài liệu](https://github.com/BachNguyen-hub/K4-DAY13-NguyenLamBach-2A202602143), commit bản clone đã đối chiếu với GitHub: `e226b934c656f23c0da70b1e12cbf365fb78dd82`.
- Code **trong gói chạy**: revision `0831856d921609312d42c7582c366e5a311bb7b1`, `working_tree_dirty: true` theo manifest. Đây là provenance khác commit repository tài liệu; không mô tả gói là clean build của commit mới nhất.
- Image tag nguồn: `day13-pointpillars:lc-20261001-amd64`.
- Image ID: `sha256:e03983bd922ec29890bf547db8de408402efd82583680b62e671c20da2fd2c82`.
- Checkpoint pretrained KITTI: `/opt/PointPillars/pretrained/epoch_160.pth`; SHA256 `482dfcf63b932cc5ccf012b4bbdad52aa51aa33becf87d0a39d61c39b377b5b1`.
- Đầu vào: `input/demo.pcd`, `frame_id=demo`, **17.238 điểm**, từ mẫu KITTI/MMDetection3D `000008`; SHA256 `3b5ea3da13e2b19149cab6a8d521c2ca55f2df93f026b5a3f8c273ce70645d60`. Dùng mẫu Student cho thí nghiệm học thuật phi thương mại theo CC BY-NC-SA 3.0, giữ [ghi nguồn](ATTRIBUTION.md) và giấy phép.
- Chuyển đổi dữ liệu: x/y giữ nguyên, `z_pcd = z_kitti + 1.73 m`; reflectance gốc bị bỏ, RGB=0 là placeholder. Adapter dùng kênh hằng **0,0 cho vehicles** và **0,7 cho pedestrian/two-wheels**, không phải intensity phục hồi.
- Ground ước lượng từ PCD: **`z_ground=0.075 m`** ở cả A/B/C, không phải mặt đường cục bộ chính xác tại mọi đối tượng.
- Phạm vi: một pass `identity`, front-window trong hệ model **x ∈ [0; 69,12] m, y ∈ [−39,68; 39,68] m, z ∈ [−3; 1] m**; không bật `--full-scene`. Score threshold **0,3**, cùng checkpoint cho ba lượt.

Lệnh thực thi trước khi điền báo cáo:

```powershell
py -3 student-bundle.py run --bundle . --out 'D:\Vin AI 2026\ket-qua-Day13-20261002-02'
```

Lần thử lại đầu tiên gặp Docker Engine chưa hoạt động; sau khi khởi động Docker Desktop, runner hoàn tất. Các bước kiểm manifest, nạp image, A/B/C và QC đều qua kiểm tra của runner. Cảnh báo PyTorch về `meshgrid` không làm lần chạy thất bại.

## 2. Ba lượt inference thật

`n_boxes` và `mean_z` lấy trực tiếp từ `summary.csv`, không suy ra từ ảnh. `mean_z` là trung bình cao độ tâm hộp trong hệ PCD nguồn, không phải chỉ số chất lượng.

| Lượt | delta (m) | Pillar XY (m) | Số hộp | mean_z (m) | File JSON/Side/CSV (trong OUT) | Quan sát có bằng chứng |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A | 0 | 0.16 | 1 | 0.330 | `run-A/boxes-demo-delta-0-voxel-0.16.json`; `run-A/side-demo-delta-0-voxel-0.16.png`; `run-A/summary.csv` | Chỉ 1 `vehicles`, tâm x≈13,154 m, y≈−0,451 m, z≈0,330 m. Side có một hộp x≈11–15 m; đáy xuống z≈−0,399 m. Chưa đủ cơ sở duyệt hộp chỉ từ hình chiếu. |
| B | 1.73 | 0.16 | 13 | 1.034 | `run-B/boxes-demo-delta-1.73-voxel-0.16.json`; `run-B/side-demo-delta-1.73-voxel-0.16.png`; `run-B/summary.csv` | 10 `vehicles`, 2 `pedestrian`, 1 `two-wheels`. Side có hộp từ vùng gần x≈3,7 m tới xe ở x≈55,58 m; nhiều hộp chồng hình chiếu tại x≈6–11 m. |
| C | 1.73 | 0.32 | 6 | 1.091 | `run-C/boxes-demo-delta-1.73-voxel-0.32.json`; `run-C/side-demo-delta-1.73-voxel-0.32.png`; `run-C/summary.csv` | Cả 6 hộp đều `pedestrian`; không còn prediction `vehicles`/`two-wheels`. Side có hộp hẹp tại x≈9–19 m và x≈33,53 m, không còn các hộp xe dài ở x≈41 và 56 m như B. |

### A/B: dịch input trước model khác dịch output như thế nào?

Giữ pillar 0,16 m, đổi delta từ 0 sang 1,73 m làm số hộp đổi **1 → 13**, đồng thời đổi phân bố class và vị trí. Chỉ dịch output một hằng số không thể tự sinh thêm 12 hộp hoặc thay đổi class/x/y. A/B là hai lần inference trên input đã biến đổi, không phải cùng một bộ hộp được nâng/hạ 1,73 m.

Pipeline dùng `z_model = z_source - z_ground - delta`; sau đó điểm còn được lọc bằng ROI trong hệ model rồi đưa vào mạng. Đổi delta có thể đổi cả điểm được giữ và biểu diễn đầu vào. Khi xuất prediction, script cộng ngược phép dịch. Chênh lệch `mean_z` **1,034 − 0,330 = 0,704 m** không cần bằng 1,73 m vì hai tập hộp khác nhau.

Trên Side, A chỉ có hộp tại x≈13 m; B có nhiều hộp tại các vùng x≈4–56 m. Chưa chắc các hộp thêm của B có đúng đối tượng hay A bỏ sót bao nhiêu: gói không có ground truth/camera để đo accuracy.

### B/C: ảnh hưởng của pillar

Giữ delta 1,73 m, tăng cạnh pillar **0,16 → 0,32 m** làm diện tích ô XY tăng bốn lần. Số hộp giảm **13 → 6**, class chuyển từ ba loại sang chỉ `pedestrian`; `mean_z` tăng 0,057 m. B có xe tại x≈40,98 m và 55,58 m, còn JSON/Side của C không có hộp xe ở hai vùng này. Đây là thay đổi prediction khi thay biểu diễn gom điểm.

C vẫn dùng checkpoint ban đầu, không phải model được train lại cho pillar 0,32 m. Chưa đủ bằng chứng kết luận B chính xác hơn C hoặc ô nhỏ luôn tốt hơn ô lớn. Cần nhãn chuẩn hoặc review đa góc có cơ sở để phân biệt miss, false positive và sai class.

### Giới hạn ROI, góc Side và việc import

- ROI chỉ phủ cửa sổ phía trước. Không có hộp ngoài ROI không chứng minh toàn frame không có đối tượng. Model cũng không dự đoán `Animal`/`Obstacle`.
- Side chiếu **x-z**, mất thông tin y. Nhiều hộp B tại x≈6–11 m có y khác nhau nhưng chồng nhau trên ảnh; không kết luận trùng hộp chỉ vì cùng vùng x. Side không đủ xác nhận yaw/hướng đầu xe; cần Top, Front, góc xoay và camera cùng frame khi có.
- Đường z=0 trên plot là tham chiếu, không chứng minh mọi đáy hộp phải nằm ở đó. Cần mặt đường cục bộ quanh đối tượng.
- **Không import JSON A/B/C hoặc `case-*.json` vào job Robotaxi.** A/B/C thuộc KITTI frame `demo`; các ca QC có `training_only=true`. B và `case-correct` cũng chưa phải nhãn đúng. Với Robotaxi, chỉ nạp prediction đúng frame/schema/job qua portal, rồi kiểm class, tọa độ, kích thước, yaw, hộp thiếu/thừa và Save trong CVAT.

## 3. Ca QC có kiểm soát — không import CVAT

Helper dùng prediction thật của **B**, không inference thêm và không sửa JSON B. Manifest QC trỏ source SHA256 `2ffb4e85d1a8746b1f290f6704204bf57e5521ae3df98ea81473a087a066fcfc`; mọi ca có `training_only=true`.

Lượng dịch có chủ đích: **`delta + z_ground = 1.73 + 0.075 = 1.805 m`**. Số hộp lệch trong bảng là so với B, không phải số lỗi đo được bằng ground truth.

| Ca | Số hộp lệch z / tổng hộp | Lượng lệch so với B | Class/x/y/yaw có đổi? | Quyết định | Bằng chứng (trong OUT) |
| --- | --- | --- | --- | --- | --- |
| case-correct | 0/13 | 0 m | Không; mọi trường hộp giữ nguyên | Giữ phép chuyển nguồn; vẫn kiểm từng hộp bằng hình học, chưa chứng nhận đúng nhãn | `qc-cases/case-correct.json`, `qc-cases/side-correct.png`; 13 tâm z trùng B |
| case-batch-z | 13/13 | −1,805 m cho mọi hộp | Không; kích thước/score cũng giữ nguyên | **Dừng sửa tay cả batch**, kiểm transform/pipeline và yêu cầu tạo lại prediction đúng | `qc-cases/case-batch-z.json`, `qc-cases/side-batch-z.png`; cả dãy hộp bị hạ cùng lượng |
| case-one-box-z | 1/13 | −1,805 m ở hộp đầu tiên; 12 hộp khác 0 m | Không; kích thước/score cũng giữ nguyên | Kiểm riêng đối tượng qua nhiều view; không quy lỗi toàn pipeline từ một hộp | `qc-cases/case-one-box-z.json`, `qc-cases/side-one-box-z.png`; chỉ hộp đầu tiên bị hạ |

Đối chiếu từng phần tử JSON xác nhận mọi trường ngoài z đều bất biến. Hộp đầu tiên là `vehicles`, x≈8,094 m, y≈1,208 m: z của B/ca giữ chuyển đổi ≈**0,921498 m**, z trong hai ca lỗi ở hộp này ≈**−0,883502 m**. Trên Side batch-z, cả dãy hộp hạ xuống; trên Side one-box-z, chỉ hộp gần x≈8 m chìm xuống, các hộp khác giữ nguyên.

Tên `case-correct` chỉ nói giữ phép chuyển tọa độ, không phải reference chất lượng cuboid. Khi chưa biết biến đổi trong tình huống thực tế, cần đối chiếu dữ liệu/log trước khi kết luận lỗi pipeline; không tự cộng 1,805 m cho mọi frame/model.

## 4. Nhận xét cá nhân và liên hệ CVAT/QC

### Vai trò và bài học từ thí nghiệm

Bài cá nhân gồm vận hành runner, đọc cấu hình/JSON/CSV, đối chiếu Side và ghi báo cáo. Quan sát nổi bật là A/B đổi cả số hộp, còn B/C đổi mạnh cơ cấu class dù cùng delta. Nhiều hộp hơn hoặc confidence cao hơn không tự là đúng hơn; không dùng `mean_z` chọn cấu hình tốt nhất.

Phép thuận/ngược:

```text
z_model  = z_source - z_ground - delta
z_source = z_model  + z_ground + delta
```

Với B/C, phép dịch tọa độ là 1,805 m. JSON xuất đã về hệ nguồn, z là tâm hộp; không cộng ngược thêm lần nữa. Script còn xử lý bottom-z sang center-z và yaw theo checkpoint. Nếu cả batch lệch có hệ thống, dừng và kiểm chuyển hệ tọa độ thay vì sửa hàng loạt bằng tay. Nếu một hộp bất thường, kiểm đối tượng qua nhiều góc và ghi điều chưa chắc.

### Thông tin đã đối chiếu trên portal/CVAT

Đọc trực tiếp [portal của ca](https://precious-opera-rarely-essays.trycloudflare.com/lab13-program/) ngày 02/10/2026, tài khoản `2A202602143`:

| Nội dung | Trạng thái quan sát được | Ý nghĩa và giới hạn |
| --- | --- | --- |
| Phiên cá nhân | Portal báo đã hết phiên | Lần chạy lại KITTI phục vụ hoàn thiện báo cáo; không tự nhận đã chạy trong phiên 240 phút |
| 30 job nguồn, task 2991 | Job **14017–14028: 12 `ready`**; **14029–14046: 18 `draft`** | `ready` cho biết bản nộp sẵn sàng chờ QC, không đồng nghĩa hoàn tất v2/nhãn đúng. `draft` chưa nộp qua portal, không đủ cơ sở nói chưa hề chỉnh trong CVAT |
| Job 14017, frame `1772259100-099741459` | [CVAT nguồn](https://cvat.note.transformerlabs.ai/tasks/2991/jobs/14017) mở đúng tài khoản; sidebar có **25 cuboid** | Có annotation hiển thị; không suy ra tất cả đúng hoặc tất cả do học viên thêm/sửa |
| Snapshot v1 job 14017 | [Viewer chỉ đọc](https://precious-opera-rarely-essays.trycloudflare.com/lab13-program/viewer/14017) báo **HTTP 503** cho point cloud và camera | Chưa kiểm chứng được hình học/camera snapshot ở lần truy cập này; không suy kết quả KITTI thành đánh giá Robotaxi |
| QC frame `1772259101-099735260` | Một lượt **`reviewed`**, phạm vi **`partial`**, lỗi **`extra_object`**, ID ghi **`2151718`** | Có feedback đã lưu; không tự nhận đã rà toàn frame hay hoàn tất toàn bộ QC được giao |

Feedback đã lưu nêu hộp nằm trên vỉa hè nên không thể là xe và đề xuất xóa. Bài học là **vị trí trên vỉa hè một mình chưa đủ chứng minh không có xe**; cần dấu vết PCD/camera cùng frame, kiểm Top/Side/Front rồi mới đề nghị xóa hộp thừa. Chưa xem được snapshot QC tương ứng trong lần này nên chỉ ghi lại nội dung đã lưu, không xác nhận kết luận xóa đúng. Nhận xét tốt hơn cần ID/vùng, góc nhìn, dấu vết quan sát, điều chưa chắc và hành động có điều kiện. Đây là tự phản tỉnh từ feedback hiện có, không phải feedback mới đã nộp.

### Điều còn chưa chắc và việc tiếp theo

KITTI demo thiếu intensity thật, camera và nhãn chuẩn; Side chỉ có một hình chiếu, vùng xa thưa điểm. Chưa đo được precision/recall, chưa chốt yaw hay cấu hình tốt hơn. Trạng thái portal không chứng nhận nhãn đúng. Snapshot lỗi 503 cần LC/operator kiểm tra trước khi đối chiếu hình học tiếp.

Với job đã nộp, chờ feedback và sửa tại job nguồn theo hạn portal, Save rồi nộp v2/phản hồi khi có bằng chứng. Với 18 job `draft` và phiên đã hết, cần LC xác nhận cách tiếp tục theo quy định hiện tại; không ghi chúng đã hoàn tất. Không dùng KITTI lấp prediction Robotaxi, không sửa snapshot QC.

## 5. Tự kiểm hoàn thành và ghi nhận LC

- Đã chạy thật A/B/C cùng input/checkpoint/threshold/ROI, đổi đúng một biến theo từng cặp; đủ JSON/Side/CSV và `smoke.json` passed.
- Đã đối chiếu QC với B: 0/13, 13/13 và 1/13 hộp đổi z; lượng dịch 1,805 m, các trường khác bất biến.
- Đã ghi quyết định dừng batch/kiểm từng hộp và giới hạn kết luận; báo cáo dùng hình thức cá nhân, không cần danh sách hoặc đổi vai nhóm.
- Giữ output lần `-01` và `-02`; quá trình lập báo cáo không import KITTI/ca lỗi, sửa annotation hay gửi feedback mới.
- **LC chưa xác nhận trong báo cáo này**: tiếp nhận báo cáo, kỹ năng thực hành học viên và quyền tiếp tục job ngoài phiên cần LC ghi nhận riêng. Không tự điền đồng ý/chấm điểm thay LC.

### Lưu ý tái chạy gói sau khi điền báo cáo

Runner kiểm hash cả mẫu `PRE-LABEL-REPORT.md`. Báo cáo đã điền nên hash khác mẫu trong manifest; **manifest không được sửa để bỏ qua kiểm tra**. Bản mẫu gốc giữ tại [report-evidence/PRE-LABEL-REPORT.original.md](report-evidence/PRE-LABEL-REPORT.original.md), kèm manifest gốc. Khi chạy lại, giải nén ZIP ban đầu sang thư mục mới, chạy gói nguyên vẹn đó và chọn output mới bên ngoài; không ghi đè báo cáo này. Kết quả sử dụng ở đây được tạo trước khi sửa mẫu, lúc kiểm hash gói vẫn thành công.
