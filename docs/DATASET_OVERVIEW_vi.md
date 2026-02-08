# OpenNeuro ds003029 (v1.0.7) — Tổng quan cho seizure detection với ECoG/iEEG

Nguồn: https://openneuro.org/datasets/ds003029/versions/1.0.7  
Tên dataset trên OpenNeuro: **Epilepsy-iEEG-Multicenter-Dataset**  
Modality: **iEEG** (bao gồm **ECoG** và **SEEG**)  
Task: **ictal**  
License: **CC0**  
Quy mô trên OpenNeuro (v1.0.7): **35 participants**, **1 session**, ~**10.32GB**, **679 files**.

---

## 1) Dataset này là gì? (bức tranh lớn)

Theo phần **README** trên trang dataset:

- Đây là bộ dữ liệu iEEG/EEG đa trung tâm (multi-center) phục vụ nghiên cứu epilepsy.
- “Acquisitions include **ECoG** and **SEEG**”.
- Mỗi **run** thường là một “snapshot” tín hiệu quanh một sự kiện seizure; trong các session có seizure, mỗi run có thể tương ứng với một seizure khác nhau (thường “one seizure per EDF file” trong mô tả).
- Dataset được tổ chức theo **BIDS** (cụ thể là **iEEG-BIDS**). Tác giả cho biết đã chuyển đổi raw **EDF** sang **BrainVision format** bằng **mne_bids**.

Điểm quan trọng cho seizure detection:
- Có **event markers/annotations** do bác sĩ/clinician gắn trong file (đặc biệt liên quan **seizure onset/offset**, có thể gồm cả clinical onset/offset và electrographic onset/offset).
- Tên marker không hoàn toàn đồng nhất → thường cần rule/regex để chuẩn hóa (README cũng cảnh báo điều này).

---

## 2) Những “thành phần file” chính bạn sẽ thấy

Từ danh sách file/folder hiển thị ở OpenNeuro, root của dataset có các file chuẩn BIDS sau:

- `README`: mô tả dataset, các lưu ý về seizure markers, dữ liệu chia sẻ/không chia sẻ.
- `CHANGES`: lịch sử thay đổi theo phiên bản.
- `dataset_description.json`: metadata chuẩn BIDS cho dataset.
- `participants.tsv` + `participants.json`: thông tin participants (và mô tả cột).
- `.bidsignore`: cấu hình bỏ qua một số file khi BIDS validation.
- Nhiều thư mục dạng `sub-.../` (ví dụ `sub-jh101`, `sub-pt01`, `sub-ummc001`, ...).

Ngoài ra OpenNeuro UI có hiển thị một folder `sourcedata/`, nhưng README cũng nói rằng “dataset uploaded … does not contain the sourcedata” do có bước anonymization thêm. Điều này thường nghĩa là:
- Có thể có `sourcedata/` tối thiểu (trống/metadata) hoặc không đủ raw gốc như EDF nguyên bản.

### 2.1) Cấu trúc BIDS điển hình trong từng subject
Vì dataset theo iEEG-BIDS, mỗi subject thường có dạng:

```
sub-<label>/
  ses-<label>/              (session subfolders)
    ieeg/
      sub-<label>_ses-<label>_task-ictal_run-<index>_ieeg.vhdr
      sub-<label>_ses-<label>_task-ictal_run-<index>_ieeg.eeg
      sub-<label>_ses-<label>_task-ictal_run-<index>_ieeg.vmrk
      sub-<...>_..._ieeg.json           (sidecar: sampling rate, line frequency, v.v.)
      sub-<...>_..._channels.tsv        (channel lists + channel type + status good/bad)
      sub-<...>_..._events.tsv          (events: onset/offset markers, trial_type, ...)
      sub-<...>_..._events.json         (feature description in events.tsv)
      sub-<...>_..._electrodes.tsv      (appear in iEEG-BIDS; depends on dataset)
      sub-<...>_..._coordsystem.json    (electrode coordinatations)
```

Lưu ý quan trọng: README của dataset nêu rõ **không có T1/CT để ước lượng electrode XYZ chính xác**, nên bạn có thể gặp:
- `electrodes.tsv`/`coordsystem.json` không có tọa độ 3D “chuẩn” (hoặc rất hạn chế),
- thay vào đó có thể có “approximate brain regions / hypothesized epileptic regions” trong metadata.

---

## 3) Với mục tiêu seizure detection, bạn nên dùng file nào?

### 3.1) Dữ liệu tín hiệu (signal) — file quan trọng nhất
Trong mỗi run:
- **BrainVision** signal thường nằm ở bộ ba file:
  - `*_ieeg.vhdr` (header)
  - `*_ieeg.eeg` (binary signal)
  - `*_ieeg.vmrk` (marker file)

Để đọc tín hiệu, thường bạn sẽ trỏ vào `*.vhdr` (nhiều tool sẽ tự tìm `*.eeg` + `*.vmrk`).

### 3.2) Nhãn/điểm thời gian seizure (labels) — để train/test
Tùy theo bạn muốn label kiểu gì:
- Nếu bạn làm **event-based detection** (phát hiện onset/offset):
  - ưu tiên đọc `*_events.tsv` (nếu có) và/hoặc markers trong `*.vmrk`.
- Nếu bạn làm **window-based classification** (preictal vs ictal vs interictal):
  - bạn sẽ cần suy ra window labels từ seizure onset/offset (trong events/annotations).

Theo README:
- Bạn sẽ gặp các marker như `eeg onset`, `clinical onset`, `eeg offset`, `clinical offset`.
- Có thể có các marker lâm sàng khác (ví dụ liên quan “Marker/Mark On/Off” cho ICTAL SPECT trong một số data).
- Tên marker có thể “messy” → nên chuẩn hóa bằng regex (ví dụ tìm các chuỗi chứa `onset`/`offset` và/hoặc `SZ`).

### 3.3) Chọn kênh (channel selection / cleaning)
Đối với seizure detection, `*_channels.tsv` rất quan trọng vì:
- Chứa danh sách kênh và có thể có `status=bad` cho các kênh không nên dùng.
- README nêu ví dụ các kênh thuộc `WM`, `VENTRICLE`, `CSF`, `OUT` đã bị loại và đánh dấu `status=bad`.

Khuyến nghị:
- Luôn filter bỏ `status==bad` trước khi extract features.
- Kiểm tra các kênh không phải iEEG (ví dụ `ECG/EKG`) nếu bạn không muốn chúng làm nhiễu mô hình.

### 3.4) Thông tin participant/session và cross-center
- `participants.tsv` để:
  - lọc theo site/center (nếu có cột liên quan),
  - phân tích thống kê theo subject,
  - split train/val/test theo subject (tránh leakage).

### 3.5) SOZ / clinical hypothesis (nếu bạn cần)
README nói có “additional clinical metadata … clinical Excel table in the publication”.

Trong phần thảo luận (comments) trên OpenNeuro, tác giả có nhắc tới file kiểu `clinical_data_summary.xlsx` (cột `soz_contacts`) để xem thông tin SOZ contacts.
- Nếu file Excel này có trong bản download của bạn: nó sẽ hữu ích cho bài toán liên quan **SOZ localization**.
- Nếu bạn chỉ làm seizure detection “ictal vs non-ictal”, bạn có thể không cần file này.

---

## 4) Những lưu ý/pitfalls cho bài toán seizure detection

- **Label noise / inconsistency**: marker naming không đồng nhất giữa centers → cần mapping/regex + kiểm tra bằng tay một vài case.
- **One seizure per run (thường)**: README nói “generally there is only one seizure per EDF file”, nhưng không nên assume tuyệt đối; hãy kiểm tra events.
- **Không có electrode XYZ chính xác**: khó làm mô hình dựa trên vị trí 3D; tập trung vào time-series features là hợp lý.
- **Data availability**: README nói một số center (ví dụ Cleveland Clinic) có yêu cầu DUA → dữ liệu bạn tải từ OpenNeuro có thể đã loại phần đó.

---

## 5) Cách lấy dataset (gợi ý thực tế)

OpenNeuro cung cấp:
- Link download trực tiếp trên trang dataset.
- DataLad/Git URL (được hiển thị trên trang): `https://github.com/OpenNeuroDatasets/ds003029.git`.

Nếu bạn dùng DataLad/Git để clone, bạn có thể dễ kiểm soát version và chỉ lấy một phần file cần thiết.

---

## 6) “Checklist” tối thiểu để bắt đầu research

1) Chọn 1–3 subject `sub-...` để thử pipeline.
2) Đọc run `*_ieeg.vhdr` để lấy signal.
3) Đọc `*_channels.tsv` và loại `status=bad` + loại kênh không mong muốn.
4) Đọc `*_events.tsv` (và/hoặc `*.vmrk`) để lấy seizure onset/offset.
5) Tạo labels theo window (ví dụ 1–5s) và train baseline model.

Nếu bạn muốn, mình có thể viết luôn một notebook nhỏ (MNE / mne-bids) để:
- đọc BrainVision theo BIDS,
- parse events (onset/offset) thành đoạn ictal,
- cắt window và tạo label cho seizure detection.
