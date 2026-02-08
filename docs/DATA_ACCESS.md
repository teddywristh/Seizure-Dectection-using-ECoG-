# Data access (git-annex/DataLad) — minimal

Signal-level EDA (notebook 03) chỉ chạy được khi bạn có **content thật** của BrainVision triple:
- `*.eeg`
- `*.vhdr`
- `*.vmrk`

## Minimal: download a single run
Trong PowerShell (từ dataset root `EEG/ds003029`):

```powershell
Set-Location "EEG\ds003029"

$run = "sub-jh101/ses-presurgery/ieeg/sub-jh101_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg"

git annex get -- "$run.eeg"
git annex get -- "$run.vhdr"
git annex get -- "$run.vmrk"

git annex find -i here -- "$run.eeg"
git annex find -i here -- "$run.vhdr"
git annex find -i here -- "$run.vmrk"
```

Nếu mỗi lệnh `find -i here` in ra đúng path, nghĩa là content đã có và MNE có thể load.

## Drop to save disk (optional)
```powershell
git annex drop -- "$run.eeg"
```

## Full setup notes
Nếu cần hướng dẫn setup DataLad/git-annex chi tiết trên Windows, xem bản legacy (sẽ được đưa vào `archive/`).
