# Workflow — ds003029 EDA (refactored)

## Big picture
EDA được chia làm 2 lớp:
- **Metadata-only (full coverage)**: quét BIDS sidecars (`*_events.tsv`, `*_channels.tsv`, `*_ieeg.json`, …) để tạo các bảng tổng hợp trong `eda_outputs/` mà **không cần tải `*.eeg`**.
- **Signal-level (subset)**: chỉ load một vài run có content thật (`*.vhdr/*.vmrk/*.eeg`) để QC tín hiệu, sanity-check markers, và demo windowing/features.

## The pipeline notebooks (run in order)
1) [notebooks/01_metadata_run_summary_ds003029.ipynb](../notebooks/01_metadata_run_summary_ds003029.ipynb)
   - Tạo `ds003029_run_summary.csv` + `ds003029_event_vocab.csv`
   - Mục tiêu: inventory toàn dataset, xác định run nào có marker / có đủ metadata / có khả năng load signal

2) [notebooks/02_marker_qc_intervals_ds003029.ipynb](../notebooks/02_marker_qc_intervals_ds003029.ipynb)
   - Tạo bảng QC marker + bảng intervals (onset/offset) an toàn cho multi-seizure
   - Các file chính:
     - `ds003029_marker_qc_by_run.csv`
     - `ds003029_seizure_intervals_by_run.csv`
     - `ds003029_trial_type_*_vocab.csv`

3) [notebooks/03_signal_eda_windows_features_ds003029.ipynb](../notebooks/03_signal_eda_windows_features_ds003029.ipynb)
   - Load 1 run (ưu tiên run có intervals), crop quanh seizure (nếu có), plot waveform + PSD, rồi windowing + features demo
   - Xuất:
     - `ds003029_window_features_demo.csv`
     - `ds003029_windowing_demo_info.csv`

## Reusable code (thin notebooks)
- Logic dùng chung nằm ở [src/ds003029_eda](../src/ds003029_eda):
  - `paths.py`: chuẩn hoá đường dẫn workspace/dataset/outputs
  - `run_summary.py`: build/export `run_summary` + `event_vocab`
  - `markers.py`: regex & parsing onset/offset + pairing intervals
  - `marker_qc.py`: build/export QC + interval/vocab tables

## Validation helper
- Script đối chiếu nhanh labels: [tools/validate_labels_ds003029.py](../tools/validate_labels_ds003029.py)
  - Chạy sau khi có `ds003029_run_summary.csv`.

## Where legacy work went
- Mục tiêu là thay thế phần “core pipeline” bằng 01→02→03.
- Nếu bạn cần tham khảo lại notebook/docs cũ (trước refactor), chúng sẽ được đưa vào `archive/`.
