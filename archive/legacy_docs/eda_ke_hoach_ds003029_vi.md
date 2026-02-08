# Kế hoạch EDA cho ds003029 (Epilepsy-iEEG-Multicenter-Dataset) — không cần tải full 10.3GB ngay

Link dataset (v1.0.7): https://openneuro.org/datasets/ds003029/versions/1.0.7

Mục tiêu domain: **seizure detection** với **ECoG/iEEG**.

Điểm mấu chốt: EDA hiệu quả nhất là làm theo **2 tầng**:
1) **EDA metadata (toàn bộ dataset)**: không cần tải signal nặng → trả lời “dataset có gì / label có ổn không / có bao nhiêu run usable”.
2) **EDA signal (subset có chủ đích)**: chỉ tải vài subjects/runs đại diện → kiểm tra chất lượng tín hiệu và độ khớp nhãn.

---

## 1) Bạn đang thấy gì trên OpenNeuro? (giải thích cấu trúc)

Ở mức top-level bạn thấy:
- `participants.tsv`, `participants.json` → bảng participant + mô tả cột
- `dataset_description.json`, `README`, `CHANGES` → mô tả dataset, ghi chú
- `sub-.../` → từng patient/subject
- `sourcedata/` → thường là dữ liệu gốc hoặc tài liệu phụ; README của dataset có lưu ý rằng bản upload có thể **không chứa đầy đủ sourcedata** do anonymization

Trong mỗi `sub-...` bạn nói thấy:
- một file kiểu `sub-..._ses-presurgery_scan.tsv`
- một folder con `ieeg/`

Đây là pattern BIDS khá thường gặp:
- `*_scan.tsv` mô tả các “scan/acquisition” hoặc listing file theo session (ở đây session label là `presurgery`).
- `ieeg/` chứa data iEEG theo iEEG-BIDS: signal (BrainVision) + sidecar metadata (`*_channels.tsv`, `*_events.tsv`, `*_ieeg.json`, ...).

---

## 2) Tầng 1 — EDA metadata (không cần tải signal)

### 2.1 Participant-level EDA (trên web hoặc tải rất nhẹ)
Mục tiêu: hiểu population + kiểm soát confound.

Checklist:
- Số subjects (OpenNeuro hiển thị 35 participants) và có bao nhiêu `sub-...`.
- Đọc `participants.tsv` để xem các cột có sẵn: age/sex/center/modality (tùy dataset).
- Nếu có thông tin center/site: thống kê số subject theo center (vì đây là multi-center và center thường tạo khác biệt lớn về montage, sampling rate, noise).
- Lập kế hoạch split: **train/val/test theo subject** (không split theo window) để tránh leakage.

Output mong muốn:
- Bảng tóm tắt: counts theo center, phân bố age, missingness.

### 2.2 Run-level inventory (không cần `.eeg`)
Mục tiêu: biết mỗi subject có bao nhiêu run, run nào có seizure markers, run nào thiếu label.

Bạn có thể làm hoàn toàn từ các file nhỏ (sidecar):
- `*_scan.tsv`: liệt kê scan/run; thường có cột filename/row cho từng acquisition.
- `ieeg/*_channels.tsv`: danh sách kênh + `status` (good/bad), loại kênh (ECoG/SEEG/ECG/EKG)
- `ieeg/*_events.tsv`: các event markers; dùng để tìm onset/offset
- `ieeg/*_ieeg.json`: metadata như `SamplingFrequency`, `PowerLineFrequency`, `iEEGReference`, ...
- `ieeg/*.vhdr` (BrainVision header): nhẹ, đôi khi giúp đọc nhanh channel names/sfreq mà không cần `.eeg`

Checklist thống kê:
- Mỗi run: n_channels, % bad channels, có hay không event chứa `onset`/`offset`.
- Từ events: ước lượng seizure duration (offset − onset) nếu đủ cặp.
- Tạo “vocabulary” của `trial_type`/description trong events để biết naming messy tới mức nào.

Output mong muốn:
- `run_summary.csv`: mỗi dòng là 1 run (subject/run/sfreq/n_channels/has_onset/has_offset/...)
- `event_vocab.csv`: tần suất các tên marker

### 2.3 EDA chất lượng label (cực quan trọng)
README của dataset nói rõ marker naming không đồng nhất giữa centers.

EDA bạn cần làm trước khi train model:
- Liệt kê top marker strings xuất hiện nhiều nhất.
- Định nghĩa rule tối thiểu để bắt seizure onset/offset, ví dụ regex:
  - onset candidates: `onset|sz|seizure` (case-insensitive)
  - offset candidates: `offset|end`
- Đếm % runs bắt được onset/offset rõ ràng.

Kết luận sau tầng 1:
- Bạn sẽ biết: *bao nhiêu run usable cho seizure detection* và *label có đáng tin không*.
- Lúc này bạn mới quyết định tải signal bao nhiêu.

---

## 3) Tầng 2 — EDA signal (chỉ tải subset)

### 3.1 Chọn subset download (khuyến nghị)
Không tải full 10.3GB. Chọn subset “đủ đại diện”:
- Mỗi center/site: 2–3 subjects (nếu dataset có nhiều center).
- Mỗi subject: 1–3 runs có seizure onset/offset rõ (theo EDA tầng 1).
- Tổng ban đầu: 10–30 runs thường đủ cho EDA chất lượng.

### 3.2 Signal QC EDA (mỗi run)
Mục tiêu: kiểm tra các thứ sẽ phá seizure detection.

Các kiểm tra nhanh, thực dụng:
- Biên độ theo kênh: median/IQR/outlier → phát hiện saturate/flatline.
- PSD / bandpower: kiểm tra line noise (50/60Hz) + high-frequency noise.
- Missing/constant segments.
- Tương quan giữa kênh (phát hiện kênh trùng/copy/reference issues).
- So sánh pre-onset vs ictal windows: variance, line length, bandpower (delta→gamma), spectral entropy.

### 3.3 Sanity-check nhãn trên tín hiệu
- Overlay seizure onset/offset lên waveform/energy của 1–2 kênh để xem marker có hợp lý.
- Nếu có cả `clinical onset` và `eeg onset`: đo chênh lệch thời gian và quyết định bạn dùng cái nào làm label chính.

Output mong muốn:
- Một vài plot QC + bảng thống kê chất lượng run (drop reasons).
- Quyết định label strategy: 
  - window-based (ictal vs non-ictal) hay
  - onset detection (positive windows gần onset).

---

## 4) Bạn nên tải cái gì (tối thiểu) để làm được từng tầng?

### 4.1 Để làm Tầng 1 (metadata-only)
Tải các file nhỏ:
- `participants.tsv`, `participants.json`
- `sub-*/**/*_scan.tsv`
- `sub-*/**/ieeg/*_channels.tsv`
- `sub-*/**/ieeg/*_events.tsv`
- `sub-*/**/ieeg/*_ieeg.json`
- (optional) `sub-*/**/ieeg/*.vhdr` và `*.vmrk`

Không cần:
- `*.eeg` (file binary lớn)

### 4.2 Để làm Tầng 2 (signal)
Chỉ tải thêm các run bạn chọn:
- `sub-*/**/ieeg/*_ieeg.eeg`

---

## 5) Cách download “theo phần” (không cần tải full)

OpenNeuro có nhiều cách tải. Cách phù hợp nhất để tải theo phần là:

### Option A) DataLad (khuyến nghị nếu bạn muốn tải chọn lọc)
Trên trang dataset có DataLad/Git URL:
- `https://github.com/OpenNeuroDatasets/ds003029.git`

Ý tưởng:
- clone/install dataset
- `datalad get` theo pattern để chỉ lấy `.tsv/.json/.vhdr` (tầng 1)
- sau đó `datalad get` thêm `.eeg` của vài run (tầng 2)

### Option B) Download qua OpenNeuro UI
- Hợp lý khi bạn chỉ lấy vài file lẻ.
- Nhưng sẽ khó lặp lại và khó tự động hóa khi bạn cần nhiều runs.

---

## 6) Roadmap EDA 1 ngày (thực chiến)

1) (30–60 phút) Tải/tạo bảng từ `participants.tsv` + thống kê theo center/age.
2) (1–2 giờ) Crawl danh sách `sub-*/.../ieeg/` để tạo `run_summary.csv` và `event_vocab.csv`.
3) (30–60 phút) Chọn subset 10–30 runs “label rõ”.
4) (2–3 giờ) Tải `.eeg` cho subset và chạy QC: biên độ/PSD/bad channels + sanity-check onset/offset.
5) (30 phút) Chốt: label definition + chuẩn hóa marker naming rule.

---

## 7) Nếu bạn muốn mình hỗ trợ tiếp

Bạn có thể yêu cầu mình tạo:
- 1 notebook EDA (Python) gồm 2 phần:
  - Part A: đọc `participants.tsv` + `*_events.tsv` + `*_channels.tsv` để tạo `run_summary.csv`
  - Part B: đọc một subset BrainVision runs để QC signal + overlay seizure onset/offset

Để làm được điều này trong workspace của bạn, bạn chỉ cần:
- tải về ít nhất các file sidecar (tầng 1), hoặc clone dataset bằng DataLad.
