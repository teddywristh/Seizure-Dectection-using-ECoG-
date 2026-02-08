# Phân tích trang “Datasets Overview” của MNE-Python (liên quan đến EEG)

Tài liệu này là bản tóm tắt/giải thích theo từng phần của trang:
https://mne.tools/stable/documentation/datasets.html

Mục tiêu: giải thích **thuật ngữ (giữ English terms)**, **cách sử dụng**, và “dataset nào hữu ích cho EEG”, đặc biệt cho các bài toán **EEG time series (one or a few channels)**.

---

## 1) Tổng quan: dataset fetchers trong `mne.datasets`

Trang này giới thiệu danh sách các **dataset fetchers** (hàm hỗ trợ tải dữ liệu mẫu) có sẵn trong module `mne.datasets`.

### 1.1 Thuật ngữ chính

- **dataset fetcher**: hàm “tải dữ liệu về máy và trả về đường dẫn” (và thường tự kiểm tra nếu đã có thì không tải lại).
- **`data_path()`**: thường là hàm tải **full dataset** về máy và trả về path.
- **`load_data()`**: thường là hàm tải **một phần dataset** (ví dụ EEGBCI tải theo `subjects` và `runs`).
- **`fetch_data()`**: thường là hàm “fetch” một phần dữ liệu theo tham số (ví dụ sleep PhysioNet có `fetch_data()` theo nhóm).
- **default download location**: nơi mặc định MNE lưu datasets (fetcher sẽ kiểm tra ở đây trước; nếu chưa có mới tải). Trang nhấn mạnh vị trí này **configurable** (cấu hình được); chi tiết xem doc của từng `data_path()`.

### 1.2 Ý nghĩa đối với EEG

- Nếu bạn làm project EEG, dùng các fetchers là cách nhanh nhất để có **dữ liệu chuẩn** (đúng format, đúng metadata) để thử pipeline: `Raw` → `Epochs`/windowing → feature/model.
- Nhiều dataset trên trang là MEG/OPM/fNIRS/phantom… Không phải cái nào cũng phục vụ EEG trực tiếp. Trang này giống “catalog” tổng hợp.

---

## 2) Sample dataset (MEG/EEG + MRI) — `mne.datasets.sample.data_path()`

### 2.1 Dataset là gì?

- Dữ liệu được thu bằng hệ thống Neuromag Vectorview. Có **EEG 60-channel** ghi đồng thời với MEG.
- Thí nghiệm gồm stimuli: visual checkerboard (trái/phải), auditory tones (trái/phải), thỉnh thoảng có “smiley face” và người tham gia nhấn nút.
- Trang nêu rõ: dataset này dùng để **làm quen với MNE**, không dùng để đánh giá hiệu năng hệ thống.

### 2.2 Thuật ngữ liên quan EEG

- **`MEG/sample`**: thư mục chứa raw/evoked/cov… (trong đó có MEG/EEG recordings).
- **`subjects/sample`**: thư mục chứa MRI reconstructions để làm source modeling.
- **FreeSurfer**: công cụ dựng bề mặt vỏ não (surface reconstructions).
- **BEM (Boundary Element Method)**: mô hình dẫn điện đầu (sọ/da/…); quan trọng cho EEG forward model.
- **watershed algorithm**: phương pháp tạo BEM surfaces.
- **event IDs**: bảng mapping như LA/RA/LV/RV/smiley/button với các số (1,2,3,4,5,32) tương ứng cột event id.

### 2.3 Liên quan đến EEG time series

- Dùng tốt cho: ERP, time-frequency, thử preprocessing cơ bản (filter, rejection), hoặc lấy một vài EEG channels để làm time series modeling.
- Tuy nhiên dataset này đi kèm cả MRI/BEM/forward, nên phần “source imaging” là **nâng cao** (không bắt buộc nếu bạn chỉ làm 1–vài kênh).

---

## 3) UCL OPM Auditory (OPM MEG) — `mne.datasets.ucl_opm_auditory.data_path()`

- Đây là auditory evoked experiment với **OPM (optically pumped magnetometer)**.
- Về EEG: dataset này chủ yếu là MEG/OPM, không phải EEG. Hữu ích nếu bạn học về sensor time series nói chung, nhưng không phải lựa chọn ưu tiên cho “EEG-only”.

---

## 4) Brainstorm datasets (CTF MEG) — `mne.datasets.brainstorm.*.data_path()`

Trang liệt kê 3 dataset fetchers gắn với các tutorial của Brainstorm:

- **Auditory**
- **Resting state**
- **Median nerve**

Đặc điểm chung:
- Ghi trên hệ **CTF 275** và ở **native CTF format (.ds)**.
- Người dùng cần **đồng ý license terms** trước khi tải.

Liên quan EEG:
- Những dataset này là MEG (không phải EEG). Có thể tham khảo nếu bạn quan tâm resting-state/time-frequency nói chung.

---

## 5) SPM faces — `mne.datasets.spm_face.data_path()`

- Dataset chứa **EEG, MEG, và fMRI** cho bài toán face perception.
- Trang dẫn ví dụ pipeline “raw → dSPM” (artifact removal, epoch averaging, forward model, source reconstruction).

Liên quan EEG:
- Nếu bạn chỉ làm EEG time series, bạn có thể:
  - lấy EEG channels để làm ERP hoặc classification/regression theo stimulus,
  - bỏ qua phần forward/inverse (vì đó là source imaging).

---

## 6) EEGBCI motor imagery (PhysioNet) — `mne.datasets.eegbci.load_data()`

### 6.1 Dataset là gì?

- 64-channel EEG, 109 subjects, 14 runs/subject.
- Format: **EDF+**.
- Ghi bằng hệ **BCI2000**.

### 6.2 Cách dùng (ý chính theo trang)

- Gọi `eegbci.load_data(subjects, runs)` để lấy danh sách file.
- Đọc bằng `read_raw_edf(..., preload=True)` rồi `concatenate_raws(raws)` để ghép runs.
- Có bước `eegbci.standardize(raw)` để “make channel names follow standard conventions”.

### 6.3 Liên quan EEG time series

- Đây là dataset rất hợp cho học phần time series:
  - có labels theo task/run (motor imagery),
  - dễ lấy 1–vài kênh để làm feature + simple model (ví dụ bandpower alpha/beta, hoặc mô hình AR).

---

## 7) Somatosensory — `mne.datasets.somato.data_path()`

- Dataset somatosensory với **ERS/ERD (event-related synchronization/desynchronization)**.
- Trang gợi ý tutorial về time-frequency và DICS beamformer.

Liên quan EEG:
- Trang không khẳng định modality cụ thể trong đoạn trích (thường là MEG trong MNE examples), nên với mục tiêu EEG-only bạn có thể xem như dataset time-frequency tham khảo, nhưng không chắc là EEG.

---

## 8) Multimodal — `mne.datasets.multimodal.data_path()`

- Một subject với auditory/visual/somatosensory stimuli.
- Trang liên kết ví dụ: “Getting averaging info from .fif files” (liên quan Elekta DACQ).

Liên quan EEG:
- Dùng tốt để học `events`/epoching/averaging theo metadata hệ ghi (nhưng có thể nghiêng về MEG/Elekta workflow).

---

## 9) fNIRS motor — `mne.datasets.fnirs_motor.data_path()`

- Dataset **fNIRS** (không phải EEG). Có 3 conditions (left tapping, right tapping, control).

Liên quan EEG:
- Không trực tiếp cho EEG, nhưng có thể học tư duy trial-based time series.

---

## 10) High frequency SEF — `mne.datasets.hf_sef.data_path()`

- Somatosensory evoked fields (median nerve stimulation), sampling 3 kHz, Elekta TRIUX MEG.

Liên quan EEG:
- Chủ yếu MEG; không ưu tiên nếu bạn chỉ làm EEG.

---

## 11) Visual 92 object categories — `mne.datasets.visual_92_categories.data_path()`

- Dataset Neuromag vectorview 306-channel (MEG), phù hợp RSA.

Liên quan EEG:
- Chủ yếu MEG.

---

## 12) mTRF Dataset — `mne.datasets.mtrf.data_path()`

- **128 channel EEG** + natural speech stimulus features.
- Dùng để fit **continuous regression models** (liên quan mTRF).

Thuật ngữ:
- **mTRF (multivariate temporal response function)**: mô hình hồi quy theo thời gian (có trễ) để ánh xạ stimulus features → neural response.

Liên quan EEG time series:
- Rất phù hợp nếu bạn muốn bài toán “time series prediction/regression” hơn là ERP.

---

## 13) Kiloword dataset — `mne.datasets.kiloword.data_path()`

- Averaged EEG từ 75 subjects, lexical decision task, 960 words.
- Words “richly annotated” (dùng cho multiple regression).

Liên quan EEG:
- Hợp cho regression với metadata (word properties), nhưng vì là **averaged EEG** nên ít “single-trial dynamics”.

---

## 14) KIT / 4D / Kernel phantom datasets — `mne.datasets.phantom_*`

- Các dataset “phantom” chủ yếu để kiểm thử hệ MEG và tutorial inverse.

Liên quan EEG:
- Không ưu tiên cho EEG.

---

## 15) OPM dataset — `mne.datasets.opm.data_path()`

- OPM data được “pipe” vào Elekta DACQ để xuất FIF.
- Có các chi tiết sensor/coil type riêng (ví dụ `coil_type` 9999) và trigger values.

Liên quan EEG:
- Không phải EEG, nhưng có nhiều chi tiết về `triggers` và đồng bộ sensor.

---

## 16) The Sleep PolySomnoGraphic Database (sleep PhysioNet) — `mne.datasets.sleep_physionet.*.fetch_data()`

- 197 whole-night PSG recordings.
- Có **EEG, EOG, chin EMG**, event markers; một số records có respiration và body temperature.
- Có **hypnograms** được chấm thủ công theo manual Rechtschaffen and Kales.

Thuật ngữ:
- **PSG (polysomnography)**: ghi nhiều tín hiệu sinh lý khi ngủ.
- **hypnogram**: nhãn stage theo thời gian (Sleep staging).

Liên quan EEG time series:
- Rất mạnh cho bài toán time series classification (sleep staging) hoặc feature extraction theo cửa sổ.

---

## 17) Reference channel noise MEG data set — `mne.datasets.refmeg_noise.data_path()`

- Dataset MEG với bursts external magnetic noise, để demo noise removal.

Liên quan EEG:
- Không ưu tiên.

---

## 18) Miscellaneous Datasets

Trang ghi rõ nhóm này thường dùng cho mục đích đặc biệt trong documentation và “generally are not useful for separate analyses”, nhưng một số mục vẫn liên quan EEG.

### 18.1 fsaverage — `mne.datasets.fetch_fsaverage()`

- Tải subject template **fsaverage**.
- Trang liên kết tutorial “EEG forward operator with a template MRI”.

Liên quan EEG:
- Hữu ích khi bạn muốn làm EEG forward/inverse mà **không có MRI cá nhân**.
- Với project time series (1–vài kênh) thường **không cần**.

### 18.2 Infant template MRIs — `mne.datasets.fetch_infant_template()`

- Tải infant template MRI + MNE-specific files.

Liên quan EEG:
- Dùng cho infant EEG source modeling; không phải ưu tiên cho time series cơ bản.

### 18.3 ECoG / sEEG (trong `mne.datasets.misc.data_path()`)

- **ECoG**: electrocorticography (grid/shaft electrodes).
- **sEEG**: stereo-electroencephalography.

Liên quan EEG:
- Đây là iEEG (intracranial), khác scalp EEG nhưng vẫn là time series neural signal.

### 18.4 LIMO — `mne.datasets.limo.load_data()`

- Task face discrimination; manipulation noise level; có ví dụ single-trial linear regression.

Liên quan EEG:
- Hợp với hồi quy/GLM trên single-trial EEG.

### 18.5 ERP CORE — `mne.datasets.erp_core.data_path()`

- Dataset gốc có 6 experiments và 7 ERP components; trong MNE fetcher hiện chỉ cung cấp data của 1 participant trong Flankers paradigm.
- Data không phải “original raw”, mà là phiên bản chỉnh sửa để demo **Epochs metadata**: đã set references/montage, events được lưu như **Annotations**, format **FIFF**.

Liên quan EEG:
- Rất hợp cho ERP/time series quanh event, đặc biệt nếu bạn muốn dùng metadata.

---

## 19) SSVEP — `mne.datasets.ssvep.data_path()`

- Frequency-tagged visual stimulation: stimuli 12.0 Hz hoặc 15.0 Hz.
- N=2 participants, 10 trials, mỗi trial 20 s.
- **32 channels wet EEG**.
- Format BrainVision (.eeg/.vhdr/.vmrk) theo chuẩn **BIDS**.

Thuật ngữ:
- **SSVEP**: steady-state visually evoked potential, tín hiệu EEG “khóa” theo tần số kích thích.
- **frequency tagging**: thiết kế stimulus để tạo peak phổ tại tần số biết trước.

Liên quan EEG time series:
- Rất hợp cho project time series: bạn có thể dùng PSD/bandpower hoặc mô hình phân loại dựa trên peak 12 vs 15 Hz.

---

## 20) EYELINK

Trang liệt kê 2 dataset eye-tracking nhỏ:

### 20.1 EEG-Eyetracking — `mne.datasets.eyelink.data_path()` (subfolder `/eeg-et/`)

- Chứa cả EEG (EGI) và eye-tracking (ASCII) trong thí nghiệm pupillary light reflex.
- Event onsets được ghi bằng photodiode và gửi cho cả EEG lẫn eye-tracking.

Liên quan EEG:
- Hợp để học đồng bộ `events` và artifact liên quan mắt (blink/saccade).

### 20.2 Freeviewing — `mne.datasets.eyelink.data_path()` (subfolder `/freeviewing/`)

- Eye-tracking only.

---

## 21) Gợi ý chọn dataset cho project “EEG time series (one or a few channels)”

Nếu bạn cần dataset phù hợp nhất từ danh sách này:

- **SSVEP**: đơn giản, cực hợp cho PSD + classification theo tần số.
- **EEGBCI motor imagery**: đa subjects/runs, hợp cho classification và feature engineering.
- **sleep PhysioNet (PSG)**: hợp cho window-based modeling (sleep staging) với nhãn hypnogram.
- **mTRF**: hợp cho continuous regression/time-lag modeling.
- **ERP CORE**: hợp cho event-locked ERP + metadata.

---

## 22) Notes về license/citation

- Một số fetchers yêu cầu **đồng ý license terms** trước khi tải (ví dụ Brainstorm).
- Một số datasets nhắc rõ “please cite” các tài liệu liên quan (ví dụ sleep PhysioNet).

