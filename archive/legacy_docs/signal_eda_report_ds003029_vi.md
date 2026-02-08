# Báo cáo EDA tín hiệu ds003029 (iEEG/ECoG/SEEG) — 4 characteristics + tình trạng data

Tài liệu này tổng hợp từ notebook **eda_signal_characteristics_ds003029.ipynb** và các outputs bạn đã chạy trong **eda_rigorous_ds003029_eda_report.ipynb**:
- Phân tích 4 nhóm “characteristics” quan trọng của time series cho bài toán seizure detection.
- Giải thích **từng cell đang làm gì** (theo thứ tự cell trong notebook) và **output** hiện có.
- Trả lời các câu hỏi EDA quan trọng ở thời điểm hiện tại và kết luận **data đã sẵn sàng đưa vào model chưa**.

## 1) Tóm tắt nhanh (Executive summary)

- Dataset metadata có **106 recordings/runs** trong run_summary và **35 subjects** (theo thống kê trong notebook).
- Bạn hiện có **16/106 runs đã có nội dung EEG** (`eeg_content_present=1`) → **15.1%**.
- Trong 106 runs: có **73 runs có seizure onset+offset** → **68.9%**.
- Với trạng thái download hiện tại: **16/106 runs vừa có onset+offset vừa có EEG content** → **15.1%**.
- Notebook đã chạy EDA sâu trên **3 runs ưu tiên cao** (đều thuộc `sub-jh101`, `acq=ecog`, run 1/2/3):
  - Plot RMS/PTP theo kênh (Characteristic #1)
  - PSD + bandpower (Characteristic #2)
  - Hjorth parameters (Characteristic #3)
  - RMS trajectory theo thời gian có overlay onset/offset (Characteristic #4)
- Notebook đã demo windowing và feature extraction, tạo bảng dataset cho modeling:
  - `X_all shape = (619, 18)` và `y_all shape = (619,)`
  - Tỉ lệ positive (ictal) trong demo: ~0.415

Kết luận ngắn: **đủ để làm baseline model dạng window-feature** (proof-of-concept), nhưng **chưa “ổn” để kết luận mô hình tổng quát** vì:
- Số run có tín hiệu thực sự còn ít (16) và demo hiện mới lấy 3 run (cùng 1 subject).
- **Có khả năng leakage** nếu split theo window thay vì theo run/subject.
- **Cần chuẩn hoá sampling rate** / **xử lý line-noise** / **chuẩn hóa amplitude** trước khi huấn luyện nghiêm túc.

Ghi chú dữ liệu: trong `run_summary` hiện tại, notebook báo **thiếu cột `duration_s`** nên chưa thể mô tả phân bố “recording duration” một cách nhất quán từ bảng tổng hợp.

## 2) Domain IO (input/output) ở thời điểm hiện tại

### 2.1 Input của domain (cho seizure detection)

Notebook theo hướng **window classification**:
- Raw-window input (chưa xuất ra file trong notebook, nhưng logic có):
  - Mỗi window có shape $(C, T)$
  - $C$ = số kênh (giới hạn `MAX_CHANNELS=16` kênh good ECOG/SEEG)
  - $T$ = số mẫu trong window, với $T = window\_sec \times f_s$.
- Feature-table input (đã tạo):
  - Mỗi window → 1 vector feature
  - Dataset là bảng: $(N, F)$

### 2.2 Output/label của domain

- Nhãn `y` cho mỗi window là **binary**:
  - `1` (ictal) nếu window overlap khoảng [onset, offset]
  - `0` (non-ictal) nếu không overlap
- Onset/offset được lấy từ `run_summary` (đã tổng hợp từ BIDS events.tsv trước đó).

Ghi chú quan trọng: các notebook EDA theo hướng windowing (**eda_signal_characteristics_ds003029.ipynb** và **eda_rigorous_ds003029_eda_report.ipynb**) **không parse trực tiếp `events.tsv` khi gán nhãn**; chúng dựa vào cột `seizure_onset_s` / `seizure_offset_s` trong `ds003029_run_summary.csv`.

Vì vậy **“độ đúng của label” phụ thuộc vào pipeline tạo run_summary**. Pipeline này nằm trong notebook **eda_ds003029.ipynb** (mục “Tạo `run_summary`”), nơi `events.tsv` được parse bằng regex trên `trial_type` và lấy **onset đầu tiên** + **offset đầu tiên sau onset** (nên có thể sai nếu marker không khớp regex, có nhiều seizure trong 1 run, hoặc offset không theo đúng thứ tự).

Kiểm định consistency đã chạy (script `tools/validate_labels_ds003029.py`):
- `Runs with readable events.tsv: 106/106`
- Onset comparable: `103` (các run có onset theo regex) → match@0.5s: `103/103` (100%)
- Offset comparable: `73` (các run có onset+offset theo regex) → match@0.5s: `73/73` (100%)

Kiểm định “marker mapping + multi-seizure” đã chạy (script `tools/analyze_event_markers_ds003029.py`, đã sửa để:
- không coi `eeg sz end` là onset,
- nhận `eeg sz end` là offset,
- nhận `Z ELECTROGRAPHIC ONS` là onset,
- xuất thêm bảng intervals chi tiết):
- Runs with events.tsv readable: `106/106`
- Runs with >=1 onset markers: `103`
- Runs with >=1 offset markers: `78`
- Runs with paired intervals >=1: `78`
- Multi-seizure candidates: `16`
- Has unpaired onset: `34`
- Has orphan offset: `1`

Các file CSV phục vụ QA/ground-truth đã được tạo trong `eda_outputs/`:
- `ds003029_marker_qc_by_run.csv`: QC theo run (đếm onset/offset/paired + flags).
- `ds003029_seizure_intervals_by_run.csv`: **(khuyến nghị dùng cho labeling)** danh sách interval (onset_s/offset_s) theo run, kèm `onset_trial_type` / `offset_trial_type`.
- `ds003029_trial_type_onset_vocab.csv`, `ds003029_trial_type_offset_vocab.csv`, `ds003029_trial_type_vocab.csv`: vocab thống kê trial_type để kiểm tra regex có đang match đúng marker.

Quan sát từ vocab hiện tại:
- Onset: các string phổ biến gồm `sz onset`, `SZ EVENT # (PB SZ)`, `onset`, `clinical onset`, `eeg sz start`, `Z ELECTROGRAPHIC ONS`.
- Offset: các string phổ biến gồm `offset`, `sz offset`, `Z ELECTROGRAPHIC END`, `eeg sz end`.

Diễn giải đúng mức: kết quả này cho thấy **CSV `run_summary` đang nhất quán với `events.tsv` theo đúng heuristic/regex hiện tại** (không có “lệch pipeline”). Tuy nhiên, nó **chưa chứng minh** onset/offset đó là **ground truth lâm sàng**; nếu regex/định nghĩa marker chưa đúng với dataset thì label vẫn có thể sai một cách hệ thống.

## 3) 4 characteristics của time series: phân tích dựa trên output hiện có

Notebook đã tổ chức EDA theo 4 nhóm characteristic thực dụng cho seizure detection.

### (1) Scale & distribution / artifacts (RMS, peak-to-peak, kênh lỗi)

Notebook vẽ:
- Bar plot RMS theo kênh
- Bar plot peak-to-peak (PTP) theo kênh

Nhận xét từ các plot (3 runs của sub-jh101):
- **Độ lớn tín hiệu khác nhau đáng kể giữa các kênh**: một số kênh có RMS/PTP cao hơn rõ rệt.
- Đây có thể là:
  - Hiện tượng sinh lý thật (kênh gần vùng hoạt động hơn)
  - Hoặc **artifact/poor contact** (kênh “hot” bất thường)

Cụ thể theo log QC trong notebook (dùng robust z-score theo kênh trên tập kênh đã pick tối đa 16; ngưỡng minh hoạ `robust z > 5`):
- `sub-jh101 run-01`: nghi ngờ kênh picked-index **[10, 11]**
- `sub-jh101 run-02`: nghi ngờ kênh picked-index **[11]**
- `sub-jh101 run-03`: **không** có kênh vượt ngưỡng

Diễn giải: ngay cả khi đã lọc theo `channels.tsv` (`status=good`, `type in {ECOG,SEEG}`), vẫn có thể còn kênh “hot”/artifact cần QC bổ sung (hoặc chuẩn hoá robust theo run/kênh).
- Ý nghĩa cho modeling:
  - Cần kiểm tra các kênh có PTP cực lớn/cực nhỏ → **có thể loại bỏ hoặc chuẩn hoá**.
  - Nên **chuẩn hoá feature theo run hoặc theo kênh** (ví dụ z-score theo run) để **tránh model học “mức amplitude”** thay vì pattern seizure.

### (2) Spectral content (PSD, bandpower, line noise)

Notebook vẽ:
- PSD (Welch) của 3 kênh đầu, log-scale power, x-axis 0–200 Hz.

Nhận xét từ các PSD:
- Dạng phổ nhìn chung có **1/f**: năng lượng mạnh ở tần số thấp và giảm dần.
- Có **đỉnh hẹp** quanh ~60 Hz và harmonic (~180 Hz) → dấu hiệu **line noise**.
- Notebook cũng tính bandpower theo các dải:
  - delta (1–4), theta (4–8), alpha (8–13), beta (13–30), gamma (30–80), hfo (80–150)

Ý nghĩa cho modeling:
- **Nếu không xử lý line-noise**, các feature phổ (bandpower) **có thể bị bias**.
- Nên cân nhắc:
  - **Notch filter 50/60 Hz** (+ harmonics), hoặc
  - Dùng bandpower đã loại vùng line-noise, hoặc
  - Dùng relative bandpower (chuẩn hoá theo tổng power) để robust hơn.

### (3) Temporal dependence / complexity (Hjorth, entropy)

Notebook tính:
- Hjorth activity/mobility/complexity (tính theo kênh rồi lấy mean)
- Spectral entropy (tính theo PSD của mỗi kênh trong excerpt rồi lấy mean; trong window-features dùng entropy của kênh 0)

Diễn giải:
- **Hjorth activity** liên quan năng lượng/variance.
- **Mobility** liên quan độ “nhanh”/tần số hiệu dụng.
- **Complexity** liên quan mức biến đổi của mobility.
- Entropy cao → phổ “phẳng”/phân tán hơn; entropy thấp → phổ tập trung.

Ý nghĩa cho seizure detection:
- Seizure thường đi kèm thay đổi về regularity và energy distribution → Hjorth và entropy có thể bắt được khác biệt ictal vs non-ictal.
- Tuy nhiên: nếu chỉ dùng mean qua kênh và entropy của 1 kênh, thông tin có thể mất nhiều; baseline OK nhưng cần mở rộng nếu muốn tốt.

### (4) Non-stationarity & seizure transition (feature trajectory theo thời gian)

Notebook vẽ:
- RMS-over-time trên toàn segment (2s windows, step 1s), overlay onset (đỏ) và offset (đen) nếu có.

Nhận xét:
- RMS trajectory cho thấy tín hiệu **không stationarity**: có những đoạn RMS tăng/giảm theo thời gian.
- Trên các run có onset/offset, nhìn thấy thay đổi quanh vùng seizure (mức độ rõ tuỳ run).

Lưu ý khi diễn giải (đúng theo guidance trong notebook): non-stationarity là kỳ vọng ở iEEG, nhưng **không nên khẳng định “signature seizure” chỉ từ RMS** vì RMS rất nhạy với artifact/outlier và vì hiện mới quan sát sâu trên 3 run cùng 1 subject.

Ý nghĩa cho modeling:
- Cách gán nhãn overlap window với [onset, offset] là hợp lý cho baseline.
- Nên kiểm tra thêm các feature trajectories khác (line length, bandpower) để xem có “transition signature” rõ hơn RMS.

### (5) Trends / Seasonality / Cyclical / Spike-Outlier (bổ sung theo yêu cầu)

Phần này làm rõ cách hiểu “trend/seasonality/cyclical/spike” cho **tín hiệu iEEG** (khác với seasonality kiểu lịch). Với dữ liệu hiện tại (chủ yếu là segment quanh seizure), trọng tâm phù hợp nhất là:

**A) Trend (xu hướng theo thời gian trong 1 run/segment)**
- Trong iEEG, “trend” thường là **baseline drift** (đặc biệt ở tần số rất thấp), hoặc **thay đổi dần** của năng lượng/độ biến thiên khi tiến gần seizure.
- Trong notebook, RMS-over-time đã cho thấy dấu hiệu **không stationarity** (RMS tăng/giảm theo thời gian). Đây là dạng trend “feature-level” hữu ích cho seizure detection.
- Điểm cần chú ý: **trend do artifact** (dịch cực, thay đổi impedance, movement) có thể giống “seizure signature” nếu không QC.

**B) Seasonality (tính lặp lại theo chu kỳ cố định)**
- Ở mức “calendar seasonality” (theo ngày/tuần/tháng) thì dataset này **không phù hợp**: `acq_time` trong BIDS có thể đã date-shift/anonymized và số phiên thu theo thời gian rất thưa.
- Ở mức “signal seasonality” (lặp lại đều đặn theo chu kỳ ngắn) thì có thể xuất hiện như:
  - Nhịp sinh lý (alpha/beta/theta) nhưng thường biến thiên theo trạng thái (ngủ/thức) và theo vị trí điện cực.
  - “Pseudo-seasonality” do nhiễu hệ thống (xem mục cyclical bên dưới).
- Vì notebook đang crop quanh seizure (±120s), seasonality dài hạn (phút–giờ) khó đánh giá; nên ưu tiên QC theo **feature trajectories** và phổ tần.

**C) Cyclical (chu kỳ nhưng không nhất thiết “mùa” theo lịch)**
- Trong bối cảnh này, “cyclical” nên hiểu là **thành phần dao động lặp lại** trong tín hiệu:
  - Dao động sinh lý (delta/theta/alpha/beta/gamma): thể hiện qua PSD và bandpower.
  - **Nhiễu điện lưới (line noise 50/60 Hz) và harmonics**: đã thấy rõ ở PSD (~60 Hz, ~180 Hz). Đây là cyclical “không mong muốn”.
- Hệ quả cho model: **nếu không notch/khử line noise**, model có thể học “đặc trưng nhiễu” thay vì hoạt động thần kinh.

**D) Spike / outlier (đột biến, ngoại lệ theo thời gian hoặc theo kênh)**
- “Spike” theo nghĩa QC thường là:
  - Transient biên độ rất lớn (electrode pop, movement artifact)
  - Saturation/clipping
  - Flatline hoặc step-change do mất tiếp xúc
- Trong notebook, RMS/PTP per channel là cách nhanh để thấy **kênh outlier** (hot channel/cold channel). RMS trajectory cũng giúp thấy **đoạn outlier theo thời gian**.

Trong notebook rigorous, outlier ở mức window còn được lượng hoá bằng:
- `outlier_score = max(|robust z|)` trên 3 đặc trưng: **(RMS, PTP, line length)** theo thời gian (WINDOW=2s, STEP=1s).
- Ngưỡng minh hoạ: `outlier_score >= 6`.

Kết quả (3 run deep của `sub-jh101`):
- Run-01: outlier windows rate **2.84%**
- Run-02: outlier windows rate **2.97%**
- Run-03: outlier windows rate **2.13%**

Diễn giải: tỉ lệ outlier không quá lớn nhưng đủ để làm nhiễu feature distributions; nên coi “bad-window handling” là bước bắt buộc trước khi kết luận gì về seizure signature.
- Gợi ý QC thực dụng trước khi train:
  - **Rule-based bad channel**: loại kênh có RMS/PTP vượt ngưỡng theo MAD/z-score robust trong từng run.
  - **Rule-based bad window**: loại window có (ptp hoặc rms) cực trị, hoặc line_length quá lớn (artifact burst).
  - Nếu dùng PSD/bandpower: kiểm tra “spike” phổ (đỉnh cực hẹp) và cân nhắc notch.

Tóm lại: trong ds003029 hiện tại, “trend/cyclical/spike-outlier” đáng giá nhất là **trên feature trajectories** (RMS/line_length/bandpower theo thời gian) và **trên PSD** (đặc biệt line-noise), thay vì seasonality kiểu lịch.

## 4) Cell-by-cell: từng cell làm gì và output hiện tại

Ghi chú quan trọng về đối chiếu:
- Mục “Cell-by-cell” bên dưới mô tả **eda_signal_characteristics_ds003029.ipynb** (notebook 4-characteristics).
- Các thống kê coverage 35 subjects, cảnh báo thiếu `duration_s`, và bảng value-counts của `sfreq` trên toàn bộ runs là từ **eda_rigorous_ds003029_eda_report.ipynb**.

Ghi chú: Notebook có 19 cells (bao gồm markdown). Dưới đây đánh số theo thứ tự hiển thị trong notebook.

### Cell 1 (Markdown) — Mục tiêu notebook
- Nội dung: mô tả mục tiêu EDA signal-level, 4 characteristics, và domain IO.
- Output: không có (markdown).

### Cell 2 (Code) — Import + cấu hình hiển thị
- Làm gì:
  - Import numpy/pandas/matplotlib.
  - Import mne và welch (scipy) nếu có.
  - Set seed và option hiển thị.
- Output: không có output đáng kể (trừ khi thiếu package).

### Cell 3 (Markdown) — Config
- Nội dung: giải thích các tham số cần chỉnh.

### Cell 4 (Code) — Kiểm tra đường dẫn + tham số chính
- Làm gì:
  - Set `DATASET_ROOT = EEG/ds003029`
  - Require file `eda_outputs/ds003029_run_summary.csv`
  - Set: `N_RUNS_METADATA=15`, `N_RUNS_HEAVY=3`, `WINDOW_SEC=2`, `STEP_SEC=1`, `MAX_CHANNELS=16`, `PRE_SEC=POST_SEC=120`.
- Output (stdout):
  - In ra đường dẫn dataset root và run summary.

### Cell 5 (Markdown) — Domain IO
- Nội dung:
  - Giải thích 2 formulation: (A) window classification (khuyến nghị), (B) sequence labeling.

### Cell 6 (Code) — Load run_summary + sanity check cột
- Làm gì:
  - Đọc `ds003029_run_summary.csv` → `run_summary`.
  - Print shape và show head.
  - Check các cột kỳ vọng: `base, subject, acq, has_onset, has_offset, seizure_onset_s, seizure_offset_s`.
  - Xác định cột đánh dấu EEG đã download: `eeg_content_present`.
- Output:
  - `run_summary shape: (106, 28)`
  - Bảng head được hiển thị
  - `Missing expected cols: []`
  - `Using eeg present column: eeg_content_present`

Ghi chú: việc **thiếu `duration_s` trong `run_summary`** là quan sát từ notebook rigorous; notebook signal-characteristics không yêu cầu cột này nhưng nó vẫn đang **không có sẵn** trong file CSV hiện tại.

### Cell 7 (Code) — Chọn runs ưu tiên cho metadata và heavy EDA
- Làm gì:
  - Tạo priority dựa trên:
    - Có EEG content
    - Có onset+offset
    - acq thuộc ecog/seeg
    - ưu tiên n_channels cao hơn (nhẹ)
  - Chọn:
    - `meta_runs = top 15`
    - `heavy_runs = top 3`
  - Display một số cột của heavy_runs.
- Output:
  - `Selected for metadata: 15`
  - `Selected for heavy EDA: 3`
  - Bảng heavy_runs được hiển thị.

### Cell 8 (Markdown) — Helpers
- Nội dung: lưu ý có thể có vhdr lỗi; không preload full file.

### Cell 9 (Code) — Helper functions (load/parse/chọn kênh/feature basics)
- Làm gì:
  - `is_valid_brainvision_vhdr`
  - `paths_from_base` (map base → vhdr/vmrk/eeg/events/channels)
  - `read_events_tsv`, `read_channels_tsv`
  - `pick_good_ieeg_channels` (status=good, type in ECOG/SEEG, limit 16)
  - `compute_bandpower`, `spectral_entropy`, `hjorth_parameters`
- Output: không có.

### Cell 10 (Markdown) — Metadata QC sau annex get
- Nội dung: mục tiêu thống kê số run có EEG content, phân bố sfreq/n_channels/duration.

### Cell 11 (Code) — QC số lượng và histogram
- Làm gì:
  - In ra:
    - số run có EEG content
    - số run có onset+offset
    - số run có onset+offset+EEG
  - Vẽ histogram:
    - sfreq
    - n_channels
    - seizure_duration_s (nếu có)
    - (Recording duration `duration_s` hiện thiếu trong run_summary nên plot duration bị bỏ trống/không đáng tin)
- Output (stdout):
  - `Runs with EEG content present: 16 / 106`
  - `Runs with onset+offset: 73`
  - `Runs with onset+offset+eeg: 16`
- Output (plots):
  - sfreq (selected runs): đa số quanh ~1000 Hz, có một cụm nhỏ ~500 Hz (vì histogram này đang vẽ trên tập runs đã chọn, không phải toàn bộ 106 runs).
  - n_channels (selected runs): dao động quanh ~100–135 kênh trong các run được chọn.
  - seizure duration: phân bố rộng (khoảng vài chục giây đến vài trăm giây trong selection). Lưu ý đây là **duration của seizure theo onset/offset** (không phải duration của toàn recording).

### Cell 12 (Markdown) — 4 characteristics
- Nội dung: liệt kê 4 nhóm characteristic.

### Cell 13 (Code) — Heavy EDA: vẽ và tính feature tóm tắt cho 3 runs
- Làm gì (cho mỗi run trong heavy_runs):
  - Load BrainVision `.vhdr` bằng MNE (preload=False).
  - Xác định segment:
    - Nếu có onset/offset: crop [onset-120s, offset+120s]
    - Nếu không: lấy 5 phút đầu
  - Chọn tối đa 16 kênh good ECOG/SEEG.
  - Tính và vẽ:
    1) RMS per channel (bar)
    1b) PTP per channel (bar)
    2) PSD cho 3 kênh đầu (10s excerpt)
    4) RMS trajectory (2s windows, 1s step) + vline onset/offset
  - Tính thêm:
    - bandpower mean theo band
    - spectral entropy mean
    - Hjorth mean
  - Gom summary vào `summary_heavy`.
- Output:
  - Log đọc dữ liệu (MNE “Reading 0 … secs …”).
  - Warning: `np.trapz` deprecated (không ảnh hưởng kết quả hiện tại).
  - `Heavy EDA summary rows: 3` và hiển thị bảng summary.
  - Nhiều plots như bạn đã thấy (RMS/PTP/PSD/RMS-over-time cho run 1/2/3).

### Cell 14 (Markdown) — Dataset shape (windowing) + feature set
- Nội dung: giải thích biến run → windows; mô tả input/output; mô tả feature set.

### Cell 15 (Code) — Định nghĩa window-feature functions
- Làm gì:
  - `line_length`
  - `extract_window_features`: tạo feature vector cho 1 window
  - `window_and_label`: trượt cửa sổ và gán nhãn theo overlap onset/offset
- Output: không có.

### Cell 16 (Code) — Build demo dataset (features) từ 3 heavy runs
- Làm gì:
  - Với mỗi run (3 runs): load segment như heavy EDA
  - Chạy windowing với `WINDOW_SEC=2`, `STEP_SEC=1`
  - Tạo `X` (DataFrame features) + `y` (0/1)
  - Thêm metadata cột: subject/acq/run/base
  - Concatenate thành `X_all`, `y_all`
- Output:
  - Demo table `demo_info` (3 dòng) với thống kê:
    - run 1: 141 windows (y_pos 23, y_neg 118)
    - run 2: 337 windows (y_pos 211, y_neg 126)
    - run 3: 141 windows (y_pos 23, y_neg 118)
    - sfreq đều 1000 Hz, n_channels demo đều 16
    - n_features=14 (không tính 4 cột metadata)
  - Kích thước dataset:
    - `X_all shape: (619, 18)`
    - `y_all shape: (619,)`
    - `y positive rate: 0.4151857`
  - Hiển thị `X_all.head()`.

Feature columns thực tế trong X_all (nhìn từ head):
- Numeric: `rms_mean, ptp_mean, line_length_mean, hj_activity_mean, hj_mobility_mean, hj_complexity_mean, spec_entropy_ch0, bp_1_4_ch0, bp_4_8_ch0, bp_8_13_ch0, bp_13_30_ch0, bp_30_80_ch0, t_start, t_end`
- Metadata: `subject, acq, run, base`

### Cell 17 (Markdown) — Export outputs
- Nội dung: mô tả xuất file.

### Cell 18 (Code) — Export CSV/PKL/Parquet (parquet optional)
- Làm gì:
  - Export:
    - `eda_outputs/ds003029_signal_eda_heavy_summary.csv`
    - `eda_outputs/ds003029_windowing_demo_info.csv`
    - `eda_outputs/ds003029_window_features_demo.csv`
    - `eda_outputs/ds003029_window_features_demo.pkl`
  - Thử ghi parquet (optional).
- Output:
  - Print các đường dẫn đã ghi.
  - Parquet bị skip do lỗi engine/arrow trong kernel hiện tại (không ảnh hưởng vì đã có CSV/PKL).

### Cell 19 (Markdown) — Report bullets
- Nội dung: gợi ý bullet copy/paste vào report.

## 5) Các câu hỏi EDA quan trọng và câu trả lời ở thời điểm hiện tại

### 5.1 Dữ liệu “có đủ không”? (coverage)
- Về metadata: có 106 runs.
- Về subjects (metadata): có 35 subjects.
- Về tín hiệu thật (đã download content): hiện có 16 runs (theo `eeg_content_present=1`).
- Để train model đáng tin: 16 runs có thể đủ cho baseline, nhưng còn mỏng cho generalization (đặc biệt nếu số subject ít).

### 5.2 Nhãn seizure có ổn không?
- Có onset+offset cho 73 runs trong metadata.
- Tuy nhiên notebook đang dựa vào onset/offset trong run_summary (đã tổng hợp sẵn), không parse trực tiếp events.tsv.
- Nếu regex/heuristic tạo run_summary chọn sai marker onset/offset → label sai.

Kết quả kiểm định consistency: `run_summary` khớp 100% với `events.tsv` theo cùng regex (103/103 onset; 73/73 offset, sai số 0.5s). Điều này **tăng độ tin cậy về mặt tái lập** (reproducibility) của label, nhưng vẫn nên spot-check vài run để đánh giá **đúng nghĩa seizure boundary**.

Bổ sung kiểm định marker/multi-seizure (từ `ds003029_marker_qc_by_run.csv`):
- 78 runs có thể tạo được **ít nhất 1 interval (paired onset→offset)**.
- Có **16 multi-seizure candidates** (có >=2 onset hoặc >=2 offset hoặc >=2 intervals).
- Có **34 runs** có onset nhưng **không pair được đủ offset** (`has_unpaired_onset=1`) và **1 run** có orphan offset.

Diễn giải thực dụng:
- Nếu bạn muốn tạo **window-level ground truth** có tính “đúng logic dữ liệu” tốt hơn run_summary, nên dựa trên `ds003029_seizure_intervals_by_run.csv` và gán nhãn ictal nếu window overlap **ANY interval** trong run (union of intervals), thay vì chỉ dùng 1 cặp onset/offset đầu tiên.
- Với các run có `has_unpaired_onset=1`, cần quyết định policy (ví dụ: bỏ run khỏi supervised training, hoặc chỉ dùng interval đã pair được, hoặc fallback sang một marker khác).

Khuyến nghị EDA/QA nhãn trước khi model:
- Random spot-check 3–5 runs: mở events.tsv gốc và so onset/offset với run_summary.
- Nếu phát hiện mismatch về “ngữ nghĩa marker” (vd onset là “SZ EVENT” nhưng offset là marker khác kỳ vọng), cần điều chỉnh regex/logic chọn onset+offset (đặc biệt với run có nhiều event/đa seizure).

### 5.3 Sampling rate khác nhau có ảnh hưởng không?
- QC histogram cho thấy có ít nhất 2 cụm sampling rate (~500 Hz và ~1000 Hz).

Thống kê cụ thể (value counts của `sfreq`, làm tròn 6 chữ số):
- 1000.000000 Hz: 72 runs
- 999.412111 Hz: 13 runs
- 249.853552 Hz: 10 runs
- 499.707104 Hz: 7 runs
- 2000.000000 Hz: 3 runs
- 1024.599795 Hz: 1 run

- Với feature PSD/bandpower, fs khác nhau vẫn chạy được (vì welch dùng fs), nhưng:
  - dải tần/độ phân giải khác nhau có thể tạo bias giữa runs.

Khuyến nghị:
- **Resample về một fs chung** (vd 1000 Hz hoặc 500 Hz) trước khi trích features, hoặc
- Dùng relative bandpower/ratio features để giảm sensitivity.

### 5.4 Line noise và lọc tín hiệu
- PSD cho thấy line noise rõ (~60 Hz, harmonic ~180 Hz).

Khuyến nghị (tối thiểu trước model):
- **Notch filter tại 60 Hz** (và có thể 120/180) hoặc loại bỏ dải hẹp khi tính bandpower.

### 5.5 Channel selection và bad channels
- Notebook chọn kênh “good” theo channels.tsv và type ECOG/SEEG; limit 16.
- Nhưng plot RMS/PTP cho thấy chênh lệch năng lượng lớn → vẫn cần EDA bad-channel sâu hơn.

Khuyến nghị:
- **Thêm QC rules**: loại kênh có RMS/PTP outlier, flatline, hoặc artifact.

Tối thiểu theo kết quả hiện tại:
- Duy trì “flag” kênh theo robust z (ví dụ z>5) để review/loại bỏ theo từng run (đã thấy run-01/02 có kênh picked-index bất thường).
- Cân nhắc chuẩn hoá robust theo run/kênh (median/MAD hoặc z-score) để giảm ảnh hưởng khác biệt amplitude.

### 5.6 Window outliers (artifact burst)
- Với windowing (2s/1s), outlier-score (max |robust z| trên RMS/PTP/line length) cho thấy tỉ lệ outlier windows khoảng **2–3%** trên 3 run deep (2.13–2.97%).
- Đây là dấu hiệu cần có bước **bad-window rejection** hoặc **robust clipping** trước khi train, đặc biệt để tránh model học artifact.

### 5.7 Class imbalance / leakage
- Demo cho thấy positive rate ~0.415 trong 3 run; không quá mất cân bằng.
- Nhưng đây chỉ là 3 run của 1 subject → không đại diện.

Rủi ro leakage:
- **Nếu split train/test theo window ngẫu nhiên**, windows cùng run/subject sẽ xuất hiện ở cả train và test → **metric ảo**.

Khuyến nghị:
- **Split theo subject** hoặc ít nhất theo run.

## 6) Data đã “ổn để đưa vào model” chưa?

### 6.1 “Ổn” cho baseline/PoC (có thể làm ngay)
Có, ở mức baseline window-feature:
- Pipeline windowing chạy được.
- Có dataset demo cụ thể (619 windows) + labels.
- Có file export CSV/PKL để train nhanh (sklearn).

### 6.2 “Chưa ổn” nếu mục tiêu là model đáng tin / báo cáo nghiêm túc
Chưa, vì cần thêm các bước EDA/cleaning tối thiểu:
- Mở rộng từ demo 3 runs → ít nhất toàn bộ 16 runs đã download.
- **Xử lý sampling rate** (resample/chuẩn hoá).
- **Notch line noise**.
- **QC bad channels** (loại outlier/flatline).
- **QC bad windows/outliers** (loại windows có outlier-score quá cao, hoặc robust clipping/winsorize).
- **Validate labels** (spot-check events.tsv vs summary).
- **Thiết kế split theo subject/run để tránh leakage**.

## 7) Artifacts/export hiện có

Notebook đã export:
- `eda_outputs/ds003029_signal_eda_heavy_summary.csv`
- `eda_outputs/ds003029_windowing_demo_info.csv`
- `eda_outputs/ds003029_window_features_demo.csv`
- `eda_outputs/ds003029_window_features_demo.pkl`

Scripts QA labels đã export thêm:
- `eda_outputs/ds003029_marker_qc_by_run.csv`
- `eda_outputs/ds003029_seizure_intervals_by_run.csv`
- `eda_outputs/ds003029_trial_type_onset_vocab.csv`
- `eda_outputs/ds003029_trial_type_offset_vocab.csv`
- `eda_outputs/ds003029_trial_type_vocab.csv`

Gợi ý dùng ngay:
- Train baseline: logistic regression / random forest / XGBoost (nếu cài) trên `ds003029_window_features_demo.csv`.
- Đánh giá bằng split theo run (không split theo window).

---

Nếu bạn muốn, bước tiếp theo hợp lý nhất là: chạy batch windowing cho **tất cả 16 runs có EEG content** và xuất một dataset lớn hơn + bảng QC “run load OK/fail, fs, n_channels, duration, y_pos/y_neg”.
