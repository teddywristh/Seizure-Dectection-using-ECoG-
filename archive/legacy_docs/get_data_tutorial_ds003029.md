# Hướng dẫn lấy dữ liệu cần thiết (git-annex) — ds003029

Mục tiêu: tải **tối thiểu** dữ liệu để chạy được phần **Visualize 1 case signal** trong notebook [eda_ds003029.ipynb](eda_ds003029.ipynb).

> Dataset này là DataLad/git-annex. File `.eeg/.vhdr/.vmrk` thường chỉ là placeholder nếu bạn mới clone. Bạn phải `git annex get` thì MNE mới đọc được.

## 0) Yêu cầu
- Đã có repo dataset tại: `EEG/ds003029`
- Có `git` và `git-annex` dùng được trong terminal

## 1) Di chuyển vào dataset root
PowerShell:

```powershell
Set-Location "C:\Users\LENOVO\Downloads\eeg\EEG\ds003029"
```

(hoặc dùng đường dẫn của bạn tới `...\eeg\EEG\ds003029`).

## 2) Chọn 1 run để tải
Bạn cần basename (không extension) dạng:

`sub-xxx/ses-yyy/ieeg/sub-xxx_ses-yyy_task-..._acq-..._run-01_ieeg`

Ví dụ (run bạn đã dùng trước đây):

```text
sub-jh101/ses-presurgery/ieeg/sub-jh101_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg
```

## 3) Tải tối thiểu 3 file BrainVision
Trong PowerShell, đặt biến `$run` rồi get 3 file:

```powershell
$run = "sub-jh101/ses-presurgery/ieeg/sub-jh101_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg"

git annex get -- "$run.eeg"
git annex get -- "$run.vhdr"
git annex get -- "$run.vmrk"
```

> Lưu ý: chỉ `get .eeg` là **chưa đủ**. `.vhdr` và `.vmrk` cũng có thể là placeholder.

## 4) Kiểm tra content đã có trong máy chưa
```powershell
git annex find -i here -- "$run.eeg"
git annex find -i here -- "$run.vhdr"
git annex find -i here -- "$run.vmrk"
```

Nếu mỗi lệnh in ra đúng đường dẫn file, nghĩa là **content present**.

## 5) Chạy notebook
- Mở [eda_ds003029.ipynb](eda_ds003029.ipynb)
- Chạy lần lượt từ trên xuống
- Để visualize 1 case: chạy đến **Phần 3** (signal)

## 6) (Tuỳ chọn) Giải phóng dung lượng sau khi xem xong
```powershell
git annex drop -- "$run.eeg"
```

Bạn thường có thể giữ `.vhdr/.vmrk` (rất nhỏ), nhưng cũng có thể drop nếu muốn:

```powershell
git annex drop -- "$run.vhdr"
git annex drop -- "$run.vmrk"
```

## 7) Troubleshooting nhanh
- Nếu `git annex get` báo lỗi mạng/DNS (ví dụ `gai_strerror ... 11001`): kiểm tra mạng, proxy/VPN, DNS.
- Nếu notebook báo không load được BrainVision: đảm bảo bạn đã `get` đủ `.eeg/.vhdr/.vmrk` và verify bằng `git annex find -i here`.
