# Cấu trúc thư mục dự án Chat-Bot HAUI

```
chat-botHAUI/
├── app_chainlit.py              # Điểm vào chính của ứng dụng Chainlit
├── admin_dashboard.py           # Giao diện quản trị dashboard
├── telegram_bot.py              # Bot Telegram
├── audit_md_files.py            # Kiểm tra/audit các file markdown
├── diagnose.py                  # Công cụ chẩn đoán hệ thống
├── run_chatbot_tests.py         # Chạy bộ test chatbot
├── chainlit.md                  # Nội dung trang chào mừng Chainlit
├── requirements.txt             # Các thư viện Python cần thiết
├── .env                         # Biến môi trường (API keys, cấu hình)
│
├── public/                      # Tài nguyên tĩnh (ảnh, icon)
│   ├── favicon.png
│   ├── logo_dark.png
│   └── logo_light.png
│
├── data/
│   ├── processed/               # Dữ liệu đã xử lý sẵn sàng để index
│   │   ├── gioi_thieu_truong.md
│   │   ├── hoc_bong.md
│   │   ├── ky_tuc_xa.md
│   │   ├── van_bang.md
│   │   ├── quy_mo_dao_tao.md
│   │   ├── phuong_thuc_tuyen_sinh_2025.md
│   │   ├── lich_tuyen_sinh_2026.md
│   │   ├── chinh_sach_uu_tien.md
│   │   ├── faq_dang_ky_xet_tuyen.md
│   │   ├── huong_dan_dang_ky_du_tuyen_2025.md
│   │   ├── huong_dan_nhap_hoc_2025_2026.md
│   │   ├── cach_tinh_hoc_phi_2025_2026.md
│   │   ├── diem_chuan_2023_2024_2025.json
│   │   ├── chi_tieu_to_hop_2025.json
│   │   ├── chi_tieu_tuyen_sinh_2026.json
│   │   ├── diem_quy_doi.json
│   │   ├── diem_uu_tien.json
│   │   ├── muc_thu_hoc_phi.json
│   │   ├── to_hop_mon_thi.json
│   │   └── nganh/               # Thông tin chi tiết từng ngành học
│   │       ├── truong_cntt_va_truyen_thong/
│   │       │   ├── nganh_an_toan_thong_tin.md
│   │       │   ├── nganh_cong_nghe_da_phuong_tien.md
│   │       │   ├── nganh_cong_nghe_ky_thuat_may_tinh.md
│   │       │   ├── nganh_cong_nghe_thong_tin.md
│   │       │   ├── nganh_he_thong_thong_tin.md
│   │       │   ├── nganh_khoa_hoc_may_tinh.md
│   │       │   ├── nganh_ky_thuat_phan_mem.md
│   │       │   └── nganh_mang_may_tinh_va_truyen_thong_du_lieu.md
│   │       ├── truong_co_khi_o_to/
│   │       │   ├── nganh_cong_nghe_ky_thuat_co_dien_tu.md
│   │       │   ├── nganh_cong_nghe_ky_thuat_co_dien_tu_o_to.md
│   │       │   ├── nganh_cong_nghe_ky_thuat_co_khi.md
│   │       │   ├── nganh_cong_nghe_ky_thuat_khuon_mau.md
│   │       │   ├── nganh_cong_nghe_ky_thuat_o_to.md
│   │       │   ├── nganh_ky_thuat_co_khi_dong_luc.md
│   │       │   ├── nganh_ky_thuat_he_thong_cong_nghiep.md
│   │       │   ├── nganh_robot_va_tri_tue_nhan_tao.md
│   │       │   └── nganh_thiet_ke_co_khi_va_kieu_dang_cong_nghiep.md
│   │       ├── truong_dien_dien_tu/
│   │       │   ├── nganh_cong_nghe_ky_thuat_dien_dien_tu.md
│   │       │   ├── nganh_cong_nghe_ky_thuat_dien_tu_vien_thong.md
│   │       │   ├── nganh_cong_nghe_ky_thuat_dien_tu_y_sinh.md
│   │       │   ├── nganh_cong_nghe_ky_thuat_dieu_khien_va_tu_dong_hoa.md
│   │       │   ├── nganh_cong_nghe_ky_thuat_nhiet.md
│   │       │   ├── nganh_ky_thuat_san_xuat_thong_minh.md
│   │       │   └── nganh_nang_luong_tai_tao.md
│   │       ├── truong_kinh_te/
│   │       │   ├── nganh_ke_toan.md
│   │       │   ├── nganh_kiem_toan.md
│   │       │   ├── nganh_kinh_te_dau_tu.md
│   │       │   ├── nganh_logistics_va_quan_ly_chuoi_cung_ung.md
│   │       │   ├── nganh_marketing.md
│   │       │   ├── nganh_phan_tich_du_lieu_kinh_doanh.md
│   │       │   ├── nganh_quan_tri_kinh_doanh.md
│   │       │   ├── nganh_quan_tri_nhan_luc.md
│   │       │   ├── nganh_quan_tri_van_phong.md
│   │       │   └── nganh_tai_chinh_ngan_hang.md
│   │       ├── truong_ngoai_ngu_du_lich/
│   │       │   ├── nganh_du_lich.md
│   │       │   ├── nganh_ngon_ngu_anh.md
│   │       │   ├── nganh_ngon_ngu_han_quoc.md
│   │       │   ├── nganh_ngon_ngu_hoc.md
│   │       │   ├── nganh_ngon_ngu_nhat.md
│   │       │   ├── nganh_ngon_ngu_trung_quoc.md
│   │       │   ├── nganh_ngon_ngu_trung_quoc_lien_ket_2_2.md
│   │       │   ├── nganh_quan_tri_dich_vu_du_lich_va_lu_hanh.md
│   │       │   ├── nganh_quan_tri_khach_san.md
│   │       │   ├── nganh_quan_tri_nha_hang_va_dich_vu_an_uong.md
│   │       │   └── nganh_trung_quoc_hoc.md
│   │       └── khoa_cong_nghe_hoa/
│   │           ├── cong_nghe_ky_thuat_hoa_hoc.md
│   │           ├── cong_nghe_ky_thuat_moi_truong.md
│   │           ├── cong_nghe_thuc_pham.md
│   │           └── hoa_duoc.md
│   │       └── khoa_cong_nghe_may_va_thiet_ke_thoi_trang/
│   │           ├── cong_nghe_det_may.md
│   │           ├── cong_nghe_vat_lieu_det_may.md
│   │           └── thiet_ke_thoi_trang.md
│   │
│   ├── vectorstore/             # Vector database (ChromaDB)
│   │   └── chroma_db/           # Dữ liệu embedding đã index
│   │
│   └── chat_logs.db             # Cơ sở dữ liệu SQLite lưu lịch sử chat
│
├── src/                         # Mã nguồn chính
│   ├── __init__.py
│   ├── indexing/                # Module xây dựng vector index
│   │   ├── __init__.py
│   │   ├── build_index.py       # Script tạo/cập nhật ChromaDB index
│   │   ├── chunker.py           # Chia nhỏ văn bản thành các chunk
│   │   └── embedder.py          # Tạo embedding vectors
│   │
│   ├── pipeline/                # Pipeline xử lý hội thoại
│   │   ├── __init__.py
│   │   ├── chatbot.py           # Logic chatbot chính (RAG pipeline)
│   │   ├── chatbot_patch.py     # Patch/fix bổ sung cho chatbot
│   │   ├── router.py            # Định tuyến câu hỏi đến handler phù hợp
│   │   ├── profiler.py          # Phân tích hồ sơ người dùng
│   │   ├── entity_extractor.py  # Trích xuất thực thể từ câu hỏi
│   │   ├── smart_context.py     # Quản lý context thông minh
│   │   └── logger.py            # Ghi log hội thoại
│   │
│   ├── query_json/              # Module truy vấn dữ liệu JSON có cấu trúc
│   │   ├── __init__.py
│   │   ├── _loader.py           # Load dữ liệu JSON
│   │   ├── _utils.py            # Tiện ích xử lý chuỗi/dữ liệu
│   │   ├── diem_chuan.py        # Truy vấn điểm chuẩn tuyển sinh
│   │   ├── diem_xet_tuyen.py    # Truy vấn điểm xét tuyển
│   │   ├── hoc_phi.py           # Truy vấn học phí
│   │   ├── nganh.py             # Truy vấn thông tin ngành học
│   │   └── formatter.py         # Định dạng kết quả trả về
│   │
│   └── retrieval/               # Module truy xuất tài liệu (RAG)
│       ├── __init__.py
│       └── retriever.py         # Truy xuất tài liệu từ ChromaDB
│
└── tests/                       # Kiểm thử và đánh giá
    └── results/                 # Kết quả đánh giá (CSV, JSON)
```

## Mô tả tổng quan

Dự án **Chat-Bot HAUI** là một chatbot hỗ trợ tư vấn tuyển sinh cho Trường Đại học Công nghiệp Hà Nội (HAUI), sử dụng kiến trúc RAG (Retrieval-Augmented Generation).

### Luồng xử lý chính

1. **Người dùng** gửi câu hỏi qua giao diện Chainlit (`app_chainlit.py`) hoặc Telegram (`telegram_bot.py`)
2. **Router** (`src/pipeline/router.py`) phân loại câu hỏi: truy vấn JSON có cấu trúc hay tìm kiếm văn bản tự do
3. **Query JSON** (`src/query_json/`) xử lý các câu hỏi về điểm chuẩn, học phí, chỉ tiêu
4. **Retriever** (`src/retrieval/retriever.py`) tìm kiếm tài liệu liên quan từ ChromaDB
5. **Chatbot** (`src/pipeline/chatbot.py`) tổng hợp câu trả lời bằng LLM (Claude/Gemini)
6. **Logger** ghi lại lịch sử hội thoại vào `data/chat_logs.db`
