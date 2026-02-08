# Review EDA notebook: trends / seasonality / issues (ds003029)

Notebook được review: [eda_trends_seasonality_issues_ds003029.ipynb](eda_trends_seasonality_issues_ds003029.ipynb)

## 1) Tổng quan: notebook đang làm gì?
Notebook triển khai checklist EDA time series (trend/seasonality/issues) bằng cách biến ds003029 thành một “time series” ở mức **run-level timeline**:
- Nguồn thời gian: `acq_time` trong các file `sub-*_ses-*_scans.tsv`.
- Biến quan sát $y$: **số lượng run** theo giờ/ngày sau khi resample.

Điều này hợp lý để “demo pipeline EDA time series” trên ds003029, nhưng cần lưu ý: đây là **lịch ghi nhận (recording schedule)** chứ không phải time series của một đại lượng sinh lý được đo đều theo thời gian.

---

## 2) Output của từng phần và diễn giải “đang xảy ra như nào”

### 2.1 Chuẩn bị dữ liệu (Cell 2–4)
Kết quả chính:
- Tìm thấy `scans.tsv files: 35`.
- Tạo `runs_timeline shape: (106, 6)` và lọc `_ieeg.vhdr` còn `iEEG rows: 106`.

Đánh giá:
- OK: dữ liệu đủ để làm timeline ở mức run.
- 106 run trên toàn dataset là **rất ít** so với độ dài time span → mọi phép kiểm tra seasonality dựa trên resample (hourly/daily) sẽ bị **sparse** (đa số điểm = 0).

### 2.2 Mô tả dữ liệu (Cell 6)
Kết quả chính:
- Total runs: **106**
- Missing `acq_time`: **0 (0.0%)**
- Time range: **1902-01-01 07:31:38 → 1924-08-24 16:59:28**
- Phân bố khoảng cách giữa các run (delta theo giờ):
  - Median (50%): **1410.48 giờ** (~58.8 ngày)
  - Mean: **1890.60 giờ** (~78.8 ngày)
  - Min: **17.85 giờ**
  - Max: **8267.22 giờ** (~344 ngày)

Diễn giải:
- Timeline cực kỳ **không đều** (irregular sampling) và thưa.
- Time range (1902–1924) rất bất thường cho dữ liệu y khoa hiện đại → đây gần như chắc chắn là **date-shift/anonymization**.

Kỳ vọng vs thực tế:
- Với checklist “time range, frequency, number of points”: phần này trả lời tốt.

### 2.3 Data quality check (Cell 8)
Kết quả chính:
- Missing `acq_time` by subject: 0% (những subject hiển thị).
- Duplicates:
  - Duplicate (subject,session,filename): 0
  - Duplicate (acq_time,filename): 0
- Year min/max: **1902 → 1924** (plot year distribution)

Diễn giải:
- Về kỹ thuật: timestamp có đầy đủ, parse ổn, không trùng.
- Về ngữ nghĩa: year bị dịch mạnh → **không nên** kết luận monthly/yearly seasonality từ date.

Kỳ vọng vs thực tế:
- Phần “missing/duplicates/time validity” đạt kỳ vọng.

### 2.4 Resampling & baseline view (Cell 10)
Kết quả chính:
- `runs_per_hour`: **198,514** điểm (1902→1924) — gần như toàn 0.
- `runs_per_day`: **8,272** điểm — gần như toàn 0.
- Plot `Runs per day (raw)`: các spike rời rạc (0 hoặc 1).
- Rolling mean/std (7 ngày): vẫn gần 0.

Diễn giải:
- Vì dữ liệu có 106 run trải trên 8,272 ngày, chuỗi daily là gần như nhị phân (0/1).
- Resample theo giờ tạo chuỗi cực dài và thưa → các kỹ thuật như boxplot theo hour-of-day, ACF, STL sẽ bị “dìm” bởi lượng 0 khổng lồ.

Kỳ vọng vs thực tế:
- Checklist yêu cầu “plot raw series + aggregated granularity”: có, nhưng **series không phải** dạng time series liên tục nên interpret phải thận trọng.

### 2.5 Trends (Cell 12)
Kết quả chính:
- Monthly table được tạo (nhưng phụ thuộc year/date-shift nên chỉ mang tính kỹ thuật).
- Linear trend slope: **4.10e-07** (xấp xỉ 0).
- Plot trend line gần như phẳng.

Diễn giải:
- Không thấy trend rõ rệt trong **run-count per day**.
- Tuy nhiên: “trend” ở đây phản ánh lịch ghi nhận run theo thời gian ẩn danh, không phản ánh trend sinh lý.

Kỳ vọng vs thực tế:
- Checklist yêu cầu: line plot + rolling + trend định lượng (linear): notebook đã làm.
- Nhưng kết luận hợp lý nhất là: **không có trend rõ rệt** (và không nên over-interpret).

### 2.6 Seasonality (Cells 14–17)

#### Seasonal plots (Cell 14)
Kết quả quan sát từ plot:
- Bar chart hour-of-day cho thấy một vài giờ có count cao hơn.
- Boxplot “Hourly runs — distribution by hour-of-day” gần như một đường phẳng quanh 0.
- Day-of-week bar chart có vẻ lệch (Sat/Sun cao hơn), nhưng có cảnh báo date-shift.

Diễn giải:
- Bar chart hour-of-day **có thể** phản ánh thói quen/lịch thu EEG (giờ làm việc/lịch trực), nhưng chưa chứng minh seasonality 24h trong “hiện tượng” (ví dụ seizure).
- Boxplot theo hour-of-day bị “hỏng ý nghĩa” vì:
  - resample hourly tạo ra cực nhiều giờ 0 → mỗi nhóm hour có gần như toàn 0,
  - boxplot vì thế không thể hiện pattern.

Kết luận: seasonal plot hiện tại **chưa đạt** kỳ vọng “thấy pattern rõ” trong bối cảnh dữ liệu sparse.

#### ACF/PACF (Cell 16)
Kết quả quan sát từ plot:
- ACF/PACF gần như nằm sát 0, không thấy peak rõ tại lag=24.

Diễn giải:
- Với series thưa (mostly zeros) và chỉ 106 spike, ACF rất khó hiện peak seasonality.
- Kết quả này phù hợp với giả thuyết: **không có seasonality mạnh** (hoặc không đủ signal để phát hiện).

#### STL decomposition (Cell 17)
Kết quả quan sát:
- STL plot cho thấy “Trend/Season” gần như toàn spike giống raw; resid rất nhỏ (thang 1e-18).

Diễn giải:
- STL trên chuỗi nhị phân thưa và cực dài thường không cho decomposition có ý nghĩa.
- Seasonal component không tách được pattern ổn định 24h.

Kỳ vọng vs thực tế (đối chiếu checklist seasonality):
- Đã có: seasonal plots + ACF/PACF + STL.
- Nhưng “evidence seasonality period” hiện tại nghiêng về: **không chứng minh được** seasonality đáng tin cậy trong run-count.

### 2.7 Issues (Cell 19)
Kết quả chính:
- Spike days (|z|>3): **85**
- Plot đánh dấu rất nhiều spike.

Diễn giải:
- Đây là false positive “theo định nghĩa”: daily series chủ yếu 0, thỉnh thoảng 1.
- Rolling mean/std với dữ liệu gần nhị phân làm $z$-score trở nên không ổn định → nhiều ngày có run=1 sẽ bị gọi là spike.

Kỳ vọng vs thực tế:
- Checklist muốn phát hiện outliers/spikes trong $y$.
- Với $y$ là run-count (0/1), spike detection kiểu z-score **không phù hợp**. Output hiện tại chưa trả lời tốt phần “issues” theo nghĩa outlier thật.

---

## 3) EDA có đúng expectation chưa?

### Những phần đạt expectation
- Mô tả dữ liệu: time range, số điểm, missing `acq_time`, phân bố gap.
- Data quality: duplicates = 0; parse timestamp OK; nhận diện date-shift qua year.
- Baseline plots: có raw + rolling.

### Những phần chưa đạt expectation (hoặc cần diễn giải lại)
- Seasonality:
  - Bar chart hour-of-day có trực giác, nhưng boxplot/ACF/STL bị ảnh hưởng nặng bởi **series quá sparse**.
  - Kết luận hợp lý: “không thấy evidence seasonality mạnh trong run-count; hour-of-day pattern có thể do lịch thu dữ liệu”.
- Issues:
  - Spike detection đang báo quá nhiều spike do lựa chọn $y$ (0/1) và cách chuẩn hoá.

---

## 4) Output hiện tại có trả lời được câu hỏi của thầy giáo không?
Nếu thầy đang chấm theo checklist time series chuẩn (hourly/daily measurement), notebook **đáp ứng về mặt hình thức**: có phần mô tả dữ liệu, trend plots, seasonal plots, decomposition, ACF/PACF, và issues.

Nhưng về mặt “nội dung đúng domain”:
- Bạn phải nói rõ trong report: đây là **run-level timeline** (lịch ghi nhận), không phải giá trị sinh lý đo liên tục.
- Vì timeline quá thưa, các kết luận seasonality/trend theo kiểu forecasting dataset sẽ yếu; điều này không phải lỗi code mà là **giới hạn của proxy**.

Tóm lại:
- Có thể dùng để trả lời thầy nếu yêu cầu là “thực hiện đầy đủ checklist EDA”.
- Để trả lời thầy theo kiểu “seasonality period là bao nhiêu, có stable không”, câu trả lời thuyết phục nhất từ output hiện tại là: **không phát hiện seasonality rõ** trong run-count; mọi pattern theo giờ nếu có có khả năng là artefact của lịch thu.

---

## 5) Khuyến nghị chỉnh để report chặt hơn (không bắt buộc, nhưng rất nên)

### 5.1 Sửa cách làm seasonal plots để không bị ‘chìm’ bởi zeros
- Thay vì boxplot trên `runs_per_hour` (gồm cả 198k giờ 0), hãy phân tích trực tiếp trên **106 run timestamps**:
  - Ví dụ: đếm số run theo hour-of-day (đã có bar chart) và thêm chuẩn hoá theo subject.
  - Làm “per-subject hour-of-day profile”, rồi boxplot across subjects (mỗi subject 24 giá trị).

### 5.2 ACF/STL: dùng chuỗi ít sparse hơn hoặc kỹ thuật cho irregular event times
- Chọn chuỗi theo **ngày có hoạt động** (lọc days with runs>0) để phân tích pattern theo khoảng cách giữa events.
- Hoặc dùng Lomb–Scargle periodogram (tốt cho irregular sampling) để tìm peak 24h/7d.

### 5.3 Issues: thay spike detection theo z-score bằng thống kê phù hợp cho count sparse
- Với daily runs 0/1, “spike” không nên định nghĩa bằng z-score.
- Thay vào đó:
  - report “sparsity rate” (% ngày có run),
  - tìm “bursts” (nhiều run trong cùng ngày/giờ, nếu có),
  - hoặc tập trung issues vào chất lượng timestamp/date-shift.

### 5.4 Nhỏ nhưng nên sửa
- `resample('H')` có warning deprecate → dùng `resample('h')`.
- `plt.boxplot(..., labels=...)` warning → dùng `tick_labels=`.
- Cell cài package nên dùng `%pip install statsmodels` thay vì `pip install statsmodels` để chắc chắn đúng kernel.

---

## 6) Bullet gợi ý để bạn đưa thẳng vào report (đã điền số)

### Data description
- Run-level time range (from BIDS `acq_time`): 1902-01-01 07:31:38 → 1924-08-24 16:59:28; total runs: 106; missing `acq_time`: 0%.
- Timeline irregularity: median gap 1410.48 hours (~58.8 days), p90 4423.25 hours (~184 days) → irregular & sparse; cần resample hoặc kỹ thuật cho event times.
- Note: Year distribution (1902–1924) indicates anonymization/date-shift, so month/year interpretations are not meaningful.

### Trend
- Daily run-count series is sparse (mostly zeros with occasional spikes to 1). Linear trend slope ≈ 4.1e-07 (near zero) → no strong long-term trend detected in run-count.

### Seasonality
- Primary seasonality tested: 24h (hour-of-day). Hour-of-day counts show non-uniform distribution, but this likely reflects recording schedule rather than physiological seasonality.
- ACF/PACF on hourly resampled counts shows no clear peak at lag=24 → no strong evidence of stable 24h seasonality in run-count proxy.
- STL (period=24) does not produce a meaningful separated seasonal component due to extreme sparsity.

### Issues
- Main issue is not missing timestamps but **sparsity + anonymized calendar**: month/year (and possibly weekday) are unreliable for seasonality claims.
- Z-score spike detection on daily binary counts flags many days as spikes (85 days), indicating the method is not appropriate for this proxy; interpret with caution.
