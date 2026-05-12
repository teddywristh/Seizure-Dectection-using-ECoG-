# Hướng dẫn cài môi trường DataLad trong Conda + cài Git-Annex trên Windows (VI)

Tài liệu này giúp bạn:
1) Tạo môi trường Conda để chạy `datalad`
2) Cài và cấu hình `git-annex` trên Windows
3) Kiểm tra mọi thứ hoạt động để có thể clone/get/drop dữ liệu kiểu OpenNeuro (DataLad/git-annex)

> Gợi ý: Nên dùng **Miniconda/Anaconda Prompt** hoặc **PowerShell**. Nếu bạn dùng PowerShell, nhớ chạy `conda init powershell` một lần.

---

## 0) Yêu cầu tối thiểu
- Windows 10/11
- Conda (Miniconda/Anaconda)
- Kết nối mạng (một số remote là S3)

Khuyến nghị:
- Git for Windows (để có `git` ổn định)

---

## 1) Tạo môi trường Conda cho DataLad

### 1.1. Tạo env
Trong Anaconda Prompt:

```bat
conda create -n datalad python=3.11 -y
conda activate datalad
```

### 1.2. Cài DataLad từ conda-forge

```bat
conda install -c conda-forge datalad -y
```

Kiểm tra:

```bat
datalad --version
python -c "import datalad; print(datalad.__version__)"
```

---

## 2) Cài Git và cấu hình cơ bản

### 2.1. Git
Nếu máy bạn đã có Git: bỏ qua. Nếu chưa, cài **Git for Windows** (khuyến nghị).

Kiểm tra:

```bat
git --version
```

### 2.2. Cấu hình identity (bắt buộc để commit metadata)

```bat
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 2.3. Windows path dài (khuyến nghị)
Một số dataset có đường dẫn dài. Bật long paths:

```bat
git config --global core.longpaths true
```

(Nếu vẫn lỗi, cần bật Long Paths ở Windows policy/registry, nhưng thường `core.longpaths` đã đủ cho nhiều trường hợp.)

---

## 3) Cài git-annex trên Windows (2 cách)

DataLad phụ thuộc vào `git-annex` để quản lý nội dung (get/drop). Trên Windows, cách ổn định nhất thường là cài `git-annex` bản standalone.

### Cách A — Cài git-annex bằng conda-forge (nếu có)
Trong env `datalad`:

```bat
conda install -c conda-forge git-annex -y
```

Kiểm tra:

```bat
git annex version
```

Nếu không cài được / lỗi dependency: dùng Cách B.

### Cách B — Cài git-annex bản standalone (khuyến nghị trên Windows)

1) Tải bản phát hành git-annex cho Windows (thường là `.exe` installer hoặc `.zip`).
2) Cài đặt theo installer **hoặc** giải nén `.zip`.
3) Đảm bảo thư mục chứa `git-annex.exe` có trong `PATH`.

Kiểm tra trong terminal mới:

```bat
git annex version
where git-annex
```

> Lưu ý: Bạn có thể cài git-annex độc lập nhưng vẫn chạy trong env conda; chỉ cần `git-annex` nằm trong PATH.

---

## 4) Kiểm tra DataLad nhận được git-annex

Trong env `datalad`:

```bat
conda activate datalad

datalad --version
git --version
git annex version
```

Test nhanh: tạo repo thử và bật annex:

```bat
mkdir datalad_smoketest
cd datalad_smoketest

datalad create -c text2git demo_ds
cd demo_ds

echo hello> hello.txt
# file nhỏ thường commit vào git luôn; annex chủ yếu cho file lớn

git status
```

---

## 5) Workflow cơ bản với dataset kiểu OpenNeuro (DataLad/git-annex)

### 5.1. Clone dataset (chỉ metadata, chưa tải content)
Ví dụ (URL minh họa):

```bat
datalad clone <DATASET_URL> my_ds
cd my_ds
```

### 5.2. Xem file nào là placeholder
- Nếu repo dùng git-annex, file có thể “tồn tại” nhưng content chưa có.

Bạn có thể kiểm tra content present:

```bat
git annex find -i here
```

### 5.3. Tải 1 file (get) và bỏ (drop) để tiết kiệm dung lượng

```bat
git annex get -- path/to/file.eeg
# ... dùng xong

git annex drop -- path/to/file.eeg
```

---

## 6) Cấu hình hữu ích (khuyến nghị)

### 6.1. DataLad: tắt hỏi/interactive (đỡ bị block)

```bat
datalad configuration --scope global set datalad.runtime.raiseonerror=1
```

### 6.2. Proxy/VPN (nếu mạng công ty)
Nếu `git annex get` lỗi DNS/SSL, bạn có thể cần cấu hình proxy cho Git:

```bat
git config --global http.proxy http://user:pass@proxy:port
git config --global https.proxy http://user:pass@proxy:port
```

Hoặc dùng VPN / đổi DNS (tùy môi trường).

---

## 7) Troubleshooting nhanh

### 7.1. `datalad` không nhận ra trong terminal
- Bạn chưa `conda activate datalad`
- Hoặc PowerShell chưa init conda:

```powershell
conda init powershell
# mở terminal mới
```

### 7.2. `git annex` báo “Invalid argument …”
- Kiểm tra version git-annex. Một số subcommand khác nhau giữa phiên bản.
- Dùng lệnh phổ biến, ổn định: `git annex find -i here`, `git annex get`, `git annex drop`.

### 7.3. `git annex get` lỗi kiểu DNS (`gai_strerror ... 11001`)
- Đây thường là vấn đề mạng/DNS/proxy. Thử:
  - Đổi DNS (1.1.1.1 / 8.8.8.8)
  - Dùng mạng khác / VPN
  - Cấu hình proxy cho Git (mục 6.2)

### 7.4. Lỗi đường dẫn dài
- Đã bật `git config --global core.longpaths true`
- Nếu vẫn lỗi: bật Long Paths trong Windows settings/policy.

---

## 8) Checklist cuối
Bạn đã OK khi các lệnh sau chạy được:

```bat
conda activate datalad

datalad --version
git --version
git annex version
```

Và `git annex find -i here` chạy được trong một dataset git-annex.
