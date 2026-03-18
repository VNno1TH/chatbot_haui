"""
evaluate_rag.py  (v3 — 90 câu, ground_truth chính xác 100% từ dữ liệu thực tế)

Cải tiến so v2:
  - Sửa DT01/DT04/DT06 (đậu/trượt tính sai)
  - Sửa TT09 (25đ KV1: 0.50 không phải 0.5)
  - Sửa H07/H08 (out of scope → thay câu hỏi phù hợp)
  - Sửa TH09 (thêm tên PT1-PT5 vào keywords)
  - Sửa FAQ04 (thêm giá phòng CLC chi tiết)
  - Loại bỏ câu ngoài scope (tiến sĩ, thạc sĩ, liên thông)
  - Keywords flex: dùng _kw_hit() normalize số tiền

Chạy:
    python tests/evaluate_rag.py              # 90 câu
    python tests/evaluate_rag.py --quick      # 20 câu
    python tests/evaluate_rag.py --save       # lưu JSON
    python tests/evaluate_rag.py --cat dau_truot
"""
import sys, json, re, time, argparse
from pathlib import Path
from dataclasses import dataclass, asdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import anthropic

EVAL_DATASET = [

    # ════════════════════════════════════════════════════════
    # THÔNG TIN TRƯỜNG — 10 câu
    # ════════════════════════════════════════════════════════
    {"id":"T01","category":"truong",
     "question":"HaUI có bao nhiêu cơ sở đào tạo và ở đâu?",
     "expected":"HaUI có 3 cơ sở: Cơ sở 1 tại Minh Khai (Bắc Từ Liêm), Cơ sở 2 tại Tây Tựu (Bắc Từ Liêm), Cơ sở 3 tại Phủ Lý (Hà Nam), tổng diện tích gần 50 ha",
     "keywords":["3","Minh Khai","Tây Tựu","Phủ Lý"]},

    {"id":"T02","category":"truong",
     "question":"Địa chỉ chính của trường Đại học Công nghiệp Hà Nội?",
     "expected":"Số 298 Đường Cầu Diễn, Phường Minh Khai, Quận Bắc Từ Liêm, Hà Nội",
     "keywords":["298","Cầu Diễn","Minh Khai","Bắc Từ Liêm"]},

    {"id":"T03","category":"truong",
     "question":"HaUI trực thuộc bộ ngành nào?",
     "expected":"HaUI trực thuộc Bộ Công Thương",
     "keywords":["Bộ Công Thương"]},

    {"id":"T04","category":"truong",
     "question":"Mã trường của HaUI là gì khi đăng ký nguyện vọng?",
     "expected":"Mã trường HaUI là DCN",
     "keywords":["DCN"]},

    {"id":"T05","category":"truong",
     "question":"Trường HaUI thành lập năm nào? Tên tiền thân là gì?",
     "expected":"Tiền thân là Trường Chuyên nghiệp Hà Nội thành lập năm 1898, nâng cấp thành Đại học năm 2005, chuyển thành Đại học Công nghiệp Hà Nội ngày 20/11/2025",
     "keywords":["1898","2005"]},

    {"id":"T06","category":"truong",
     "question":"haui mấy campus vậy",
     "expected":"HaUI có 3 cơ sở đào tạo",
     "keywords":["3","cơ sở"]},

    {"id":"T07","category":"truong",
     "question":"Tỷ lệ sinh viên HaUI có việc làm sau tốt nghiệp là bao nhiêu?",
     "expected":"Tỷ lệ sinh viên có việc làm sau tốt nghiệp 1 năm đạt trên 95%",
     "keywords":["95%","việc làm"]},

    {"id":"T08","category":"truong",
     "question":"HaUI có bao nhiêu giảng viên? Tỷ lệ tiến sĩ là bao nhiêu?",
     "expected":"HaUI có gần 1.100 giảng viên, tỷ lệ có trình độ Phó Giáo sư/Tiến sĩ trên 41%",
     "keywords":["1.100","41%"]},

    {"id":"T09","category":"truong",
     "question":"trường haui ở đâu vậy",
     "expected":"HaUI ở số 298 Cầu Diễn, Minh Khai, Bắc Từ Liêm, Hà Nội",
     "keywords":["Cầu Diễn","Bắc Từ Liêm"]},

    {"id":"T10","category":"truong",
     "question":"HaUI có những thành tích nổi bật nào?",
     "expected":"HaUI đạt Huân chương Hồ Chí Minh (2003), Danh hiệu Anh hùng Lao động (2008), 5 chương trình đào tạo đạt chuẩn ABET (Hoa Kỳ), vô địch Robocon 2008 và 2023, tỷ lệ có việc làm trên 95%",
     "keywords":["ABET","Robocon","95%"]},

    # ════════════════════════════════════════════════════════
    # ĐIỂM CHUẨN — 10 câu (từ diem_chuan_2023_2024_2025.json)
    # ════════════════════════════════════════════════════════
    {"id":"D01","category":"diem_chuan",
     "question":"Điểm chuẩn ngành CNTT năm 2024 thi THPT là bao nhiêu?",
     "expected":"Điểm chuẩn ngành Công nghệ thông tin năm 2024 theo PT3 (thi THPT) là 25.22 điểm thang 30",
     "keywords":["25.22","2024","PT3"]},

    {"id":"D02","category":"diem_chuan",
     "question":"Điểm chuẩn ngành Kế toán 2024 xét bằng HSA?",
     "expected":"Điểm chuẩn ngành Kế toán năm 2024 theo PT4 (HSA - ĐHQG HN) là 27.69 điểm",
     "keywords":["27.69","Kế toán"]},

    {"id":"D03","category":"diem_chuan",
     "question":"So sánh điểm chuẩn CNTT qua 3 năm 2023 2024 2025",
     "expected":"PT3 thi THPT: 2023 là 25.19, 2024 là 25.22, 2025 là 23.09 (điểm sàn chung). Xu hướng 2025 giảm đáng kể so 2024.",
     "keywords":["25.19","25.22","23.09"]},

    {"id":"D04","category":"diem_chuan",
     "question":"Điểm chuẩn ngành Kỹ thuật phần mềm năm 2024 thi THPT?",
     "expected":"Điểm chuẩn ngành Kỹ thuật phần mềm năm 2024 theo PT3 là 24.68 điểm",
     "keywords":["24.68","Kỹ thuật phần mềm"]},

    {"id":"D05","category":"diem_chuan",
     "question":"Điểm chuẩn ngành Robot và AI năm 2024 tất cả phương thức?",
     "expected":"Robot và TTNT 2024: PT2 (học bạ) 25.77, PT3 (THPT) 25.01, PT4 (HSA) 28.44, PT5 (TSA) 16.5",
     "keywords":["25.01","28.44","16.5"]},

    {"id":"D06","category":"diem_chuan",
     "question":"Ngành Marketing điểm chuẩn 2024 thi THPT bao nhiêu?",
     "expected":"Điểm chuẩn ngành Marketing năm 2024 theo PT3 là 25.33 điểm",
     "keywords":["25.33","Marketing"]},

    {"id":"D07","category":"diem_chuan",
     "question":"Điểm chuẩn Logistics 2024 theo phương thức thi THPT?",
     "expected":"Điểm chuẩn ngành Logistics và quản lý chuỗi cung ứng năm 2024 theo PT3 là 25.89 điểm",
     "keywords":["25.89","Logistics"]},

    {"id":"D08","category":"diem_chuan",
     "question":"điểm chuẩn ktpm 2023 thi thpt",
     "expected":"Điểm chuẩn Kỹ thuật phần mềm năm 2023 theo PT3 là 24.54 điểm",
     "keywords":["24.54","2023"]},

    {"id":"D09","category":"diem_chuan",
     "question":"Ngành điện tử viễn thông HaUI điểm chuẩn 2024 là bao nhiêu?",
     "expected":"Điểm chuẩn ngành Công nghệ kỹ thuật điện tử - viễn thông năm 2024 theo PT3 là 24.4 điểm",
     "keywords":["24.4","điện tử"]},

    {"id":"D10","category":"diem_chuan",
     "question":"Điểm chuẩn năm 2025 ngành CNTT bao nhiêu? Có khác 2024 không?",
     "expected":"Năm 2025 ngành CNTT điểm sàn chung là 23.09 (thấp hơn PT3-2024 là 25.22 khoảng 2.13 điểm). Năm 2025 áp dụng điểm sàn chung cho nhiều phương thức.",
     "keywords":["23.09","25.22","2025"]},

    # ════════════════════════════════════════════════════════
    # HỌC PHÍ — 10 câu (từ muc_thu_hoc_phi.json)
    # Chỉ hỏi về đại học chính quy + kỹ sư — trong scope chatbot
    # ════════════════════════════════════════════════════════
    {"id":"H01","category":"hoc_phi",
     "question":"Học phí ngành đại trà K20 HaUI bao nhiêu tiền một tín chỉ?",
     "expected":"Học phí Cử nhân K20 chương trình đại trà là 700.000 đồng/tín chỉ",
     "keywords":["700.000","K20","đại trà"]},

    {"id":"H02","category":"hoc_phi",
     "question":"Học phí chương trình tiếng Anh K20 là bao nhiêu?",
     "expected":"Học phí Cử nhân K20 chương trình đào tạo bằng tiếng Anh là 1.000.000 đồng/tín chỉ",
     "keywords":["1.000.000","tiếng Anh"]},

    {"id":"H03","category":"hoc_phi",
     "question":"Học phí K19 tại HaUI năm 2025-2026 là bao nhiêu?",
     "expected":"Học phí Cử nhân K19 tại HaUI năm học 2025-2026 là 550.000 đồng/tín chỉ",
     "keywords":["550.000","K19"]},

    {"id":"H04","category":"hoc_phi",
     "question":"K18 trở về trước đóng học phí bao nhiêu một tín chỉ?",
     "expected":"Sinh viên Cử nhân K18 trở về trước đóng học phí 495.000 đồng/tín chỉ",
     "keywords":["495.000","K18"]},

    {"id":"H05","category":"hoc_phi",
     "question":"sinh viên K19 đóng học phí bao nhiêu một tín chỉ?",
     "expected":"Sinh viên Cử nhân K19 đóng học phí 550.000 đồng/tín chỉ",
     "keywords":["550.000","K19"]},

    {"id":"H06","category":"hoc_phi",
     "question":"Học phí kỹ sư K3 đại trà HaUI là bao nhiêu?",
     "expected":"Học phí Kỹ sư K3 chương trình đại trà là 700.000 đồng/tín chỉ",
     "keywords":["700.000","Kỹ sư K3"]},

    {"id":"H07","category":"hoc_phi",
     "question":"Học phí kỹ sư K3 chương trình tiếng Anh bao nhiêu?",
     "expected":"Học phí Kỹ sư K3 chương trình đào tạo bằng tiếng Anh là 1.000.000 đồng/tín chỉ",
     "keywords":["1.000.000","Kỹ sư K3"]},

    {"id":"H08","category":"hoc_phi",
     "question":"Kỹ sư K2 học phí một tín chỉ là bao nhiêu?",
     "expected":"Học phí Kỹ sư K2 là 550.000 đồng/tín chỉ",
     "keywords":["550.000","Kỹ sư K2"]},

    {"id":"H09","category":"hoc_phi",
     "question":"Học phí Kỹ sư K1 bao nhiêu một tín chỉ?",
     "expected":"Học phí Kỹ sư K1 là 495.000 đồng/tín chỉ",
     "keywords":["495.000","Kỹ sư K1"]},

    {"id":"H10","category":"hoc_phi",
     "question":"So sánh học phí K20 đại trà và K20 tiếng Anh khác nhau bao nhiêu?",
     "expected":"K20 đại trà 700.000 đồng/tín chỉ, K20 tiếng Anh 1.000.000 đồng/tín chỉ, chênh lệch 300.000 đồng/tín chỉ",
     "keywords":["700.000","1.000.000","300.000"]},

    # ════════════════════════════════════════════════════════
    # TỔ HỢP & CHỈ TIÊU — 10 câu (từ chi_tieu_to_hop_2025.json)
    # ════════════════════════════════════════════════════════
    {"id":"TH01","category":"to_hop",
     "question":"Ngành Công nghệ thông tin HaUI xét tổ hợp nào năm 2025?",
     "expected":"Ngành CNTT xét tuyển tổ hợp A00 (Toán-Lý-Hóa), A01 (Toán-Lý-Anh), X06 (Toán-Tin-Công nghệ), X07 (Toán-Lý-Công nghệ)",
     "keywords":["A00","A01","X06","X07"]},

    {"id":"TH02","category":"to_hop",
     "question":"Tổ hợp D01 gồm những môn gì?",
     "expected":"Tổ hợp D01 gồm Toán, Ngữ văn, Tiếng Anh",
     "keywords":["Toán","Ngữ văn","Tiếng Anh"]},

    {"id":"TH03","category":"to_hop",
     "question":"Tổ hợp X06 và X07 khác nhau như thế nào?",
     "expected":"X06 gồm Toán, Tin học, Công nghệ. X07 gồm Toán, Vật lý, Công nghệ (Công nghiệp)",
     "keywords":["Tin học","Vật lý","Công nghệ"]},

    {"id":"TH04","category":"to_hop",
     "question":"Chỉ tiêu tuyển sinh HaUI năm 2026 dự kiến bao nhiêu?",
     "expected":"Tổng chỉ tiêu 9.420 sinh viên: đại học chính quy 8.300, đại học từ xa 750, liên thông 250, kỹ sư bậc 7 120",
     "keywords":["9.420","8.300","2026"]},

    {"id":"TH05","category":"to_hop",
     "question":"Ngành Kế toán HaUI xét tổ hợp gì?",
     "expected":"Ngành Kế toán xét tuyển tổ hợp A01 (Toán-Lý-Anh), D01 (Toán-Văn-Anh), X25 (Toán-Anh-GDKT)",
     "keywords":["A01","D01","X25"]},

    {"id":"TH06","category":"to_hop",
     "question":"Ngành Du lịch HaUI xét tuyển bằng tổ hợp gì?",
     "expected":"Ngành Du lịch xét tuyển tổ hợp D01 (Toán-Văn-Anh), D14 (Văn-Sử-Anh), D15 (Văn-Địa-Anh)",
     "keywords":["D01","D14","D15"]},

    {"id":"TH07","category":"to_hop",
     "question":"Ngành Kỹ thuật phần mềm HaUI tổ hợp xét tuyển gì?",
     "expected":"Ngành Kỹ thuật phần mềm xét tuyển tổ hợp A00, A01, X06, X07 — giống CNTT",
     "keywords":["A00","A01","X06","X07"]},

    {"id":"TH08","category":"to_hop",
     "question":"HaUI tuyển sinh 2025 có bao nhiêu ngành và bao nhiêu chỉ tiêu?",
     "expected":"HaUI tuyển sinh 2025 với 62 mã ngành/chương trình đào tạo và 7.990 chỉ tiêu",
     "keywords":["62","7.990"]},

    {"id":"TH09","category":"to_hop",
     "question":"HaUI có bao nhiêu phương thức xét tuyển? Gồm những phương thức nào?",
     "expected":"HaUI có 5 phương thức: PT1 (xét thẳng), PT2 (học bạ + HSG/chứng chỉ quốc tế), PT3 (thi THPT), PT4 (ĐGNL HSA - ĐHQG HN), PT5 (ĐGTD TSA - ĐHBK HN)",
     "keywords":["PT1","PT2","PT3","PT4","PT5"]},

    {"id":"TH10","category":"to_hop",
     "question":"Ngành Ngôn ngữ Anh HaUI xét tuyển tổ hợp nào?",
     "expected":"Ngành Ngôn ngữ Anh xét tuyển tổ hợp D01 (Toán, Ngữ văn, Tiếng Anh)",
     "keywords":["D01","Ngôn ngữ Anh"]},

    # ════════════════════════════════════════════════════════
    # TÍNH TOÁN ĐIỂM — 10 câu (từ diem_quy_doi.json + diem_uu_tien.json)
    # ════════════════════════════════════════════════════════
    {"id":"TT01","category":"tinh_toan",
     "question":"TSA 80 điểm khu vực 2 nông thôn thì điểm xét tuyển là bao nhiêu?",
     "expected":"TSA 80 quy đổi 29.0 điểm thang 30. KV2-NT trên 22.5 áp dụng giảm dần: [(30-29)/7.5]×0.5=0.07. Điểm xét tuyển: 29.07",
     "keywords":["29.07","29.0","0.07"]},

    {"id":"TT02","category":"tinh_toan",
     "question":"Điểm ưu tiên khu vực 1 tối đa là bao nhiêu điểm?",
     "expected":"KV1 được cộng tối đa +0.75 điểm. Nếu tổng điểm từ 22.5 trở lên áp dụng công thức giảm dần [(30-điểm)/7.5]×0.75",
     "keywords":["0.75","KV1","22.5"]},

    {"id":"TT03","category":"tinh_toan",
     "question":"HSA 105 điểm quy đổi được bao nhiêu điểm thang 30?",
     "expected":"HSA 105 điểm quy đổi thành 25.0 điểm thang 30 theo bảng quy đổi của HaUI",
     "keywords":["25.0","105"]},

    {"id":"TT04","category":"tinh_toan",
     "question":"20 điểm KV1 thì điểm xét tuyển là bao nhiêu?",
     "expected":"20 điểm dưới ngưỡng 22.5 nên cộng thẳng +0.75. Điểm xét tuyển = 20.75",
     "keywords":["20.75","0.75"]},

    {"id":"TT05","category":"tinh_toan",
     "question":"HSA 105 điểm KV2-NT thì tổng điểm xét tuyển là bao nhiêu?",
     "expected":"HSA 105 quy đổi 25.0. KV2-NT trên 22.5: [(30-25)/7.5]×0.5=0.33. Điểm xét tuyển: 25.33",
     "keywords":["25.33","0.33"]},

    {"id":"TT06","category":"tinh_toan",
     "question":"Điểm ưu tiên KV2-NT tối đa là bao nhiêu điểm?",
     "expected":"KV2-NT được cộng tối đa +0.50 điểm. Nếu điểm từ 22.5 trở lên áp dụng công thức giảm dần",
     "keywords":["0.50","KV2-NT"]},

    {"id":"TT07","category":"tinh_toan",
     "question":"TSA 75 điểm KV2 thành thị điểm xét tuyển bao nhiêu?",
     "expected":"TSA 75 quy đổi 28.5 điểm. KV2 trên 22.5: [(30-28.5)/7.5]×0.25=0.05. Điểm xét tuyển: 28.55",
     "keywords":["28.55","28.5","0.05"]},

    {"id":"TT08","category":"tinh_toan",
     "question":"22 điểm KV1 được cộng bao nhiêu điểm ưu tiên?",
     "expected":"22 điểm dưới ngưỡng 22.5 nên cộng thẳng +0.75. Điểm xét tuyển = 22.75",
     "keywords":["22.75","0.75"]},

    {"id":"TT09","category":"tinh_toan",
     "question":"25 điểm KV1 thì điểm ưu tiên thực tế là bao nhiêu?",
     "expected":"25 điểm trên ngưỡng 22.5: [(30-25)/7.5]×0.75=0.50. Điểm ưu tiên thực là +0.50, điểm xét tuyển = 25.50",
     "keywords":["0.50","25.50"]},

    {"id":"TT10","category":"tinh_toan",
     "question":"30 điểm KV1 có được cộng điểm ưu tiên không?",
     "expected":"30 điểm đạt tối đa: [(30-30)/7.5]×0.75=0. Điểm ưu tiên thực tế bằng 0 — không được cộng thêm",
     "keywords":["0","không"]},

    # ════════════════════════════════════════════════════════
    # MÔ TẢ NGÀNH — 10 câu (từ file .md thực tế truong_cntt_va_truyen_thong)
    # ════════════════════════════════════════════════════════
    {"id":"ND01","category":"mo_ta_nganh",
     "question":"Ngành Công nghệ thông tin HaUI học những gì?",
     "expected":"CNTT (mã 7480201, 360 chỉ tiêu) học lập trình, cơ sở dữ liệu, mạng máy tính, bảo mật, AI, học máy, phát triển web và ứng dụng di động, quản lý hệ thống CNTT",
     "keywords":["lập trình","AI","mạng máy tính","bảo mật"]},

    {"id":"ND02","category":"mo_ta_nganh",
     "question":"Ngành Kỹ thuật phần mềm HaUI học gì và ra làm gì?",
     "expected":"KTPM (mã 7480103, 240 chỉ tiêu) học Agile/Scrum, DevOps, kiểm thử phần mềm, Java/C#/Python, Spring/.NET/React. Ra làm lập trình viên, kỹ sư phần mềm, DevOps, QA/QC, Solution Architect",
     "keywords":["Agile","DevOps","kiểm thử","lập trình viên"]},

    {"id":"ND03","category":"mo_ta_nganh",
     "question":"Ngành Kỹ thuật phần mềm khác gì so với CNTT?",
     "expected":"KTPM tập trung phát triển phần mềm chuyên sâu: Agile/Scrum, DevOps, kiểm thử, quản lý dự án phần mềm. CNTT rộng hơn bao gồm thêm mạng, hệ thống, AI, học máy, khoa học dữ liệu",
     "keywords":["Agile","DevOps","kiểm thử","phần mềm"]},

    {"id":"ND04","category":"mo_ta_nganh",
     "question":"Ngành An toàn thông tin HaUI học gì ra làm gì?",
     "expected":"ATTT (mã 7480202, 40 chỉ tiêu) học mã hóa, kiểm thử xâm nhập (pentesting), phân tích mã độc, an ninh mạng, bảo mật IoT/cloud. Ra làm pentester, SOC analyst, kỹ sư an ninh mạng, chuyên viên ATTT",
     "keywords":["mã hóa","pentesting","an ninh mạng","SOC"]},

    {"id":"ND05","category":"mo_ta_nganh",
     "question":"Ngành Hệ thống thông tin HaUI học và làm gì?",
     "expected":"HTTT (mã 7480104, 120 chỉ tiêu) học ERP, CRM, Business Intelligence, Data Warehouse, chuyển đổi số, Big Data, điện toán đám mây. Ra làm phân tích hệ thống, tư vấn ERP, quản trị CNTT doanh nghiệp",
     "keywords":["ERP","Business Intelligence","Big Data","chuyển đổi số"]},

    {"id":"ND06","category":"mo_ta_nganh",
     "question":"Ngành Khoa học máy tính HaUI có gì đặc biệt?",
     "expected":"KHMT (mã 7480101, 120 chỉ tiêu) học sâu về AI, học máy, học sâu, thị giác máy tính, blockchain, IoT, AR/VR, lập trình Python/Java/C++. Thiên về nghiên cứu và phát triển công nghệ tiên tiến",
     "keywords":["AI","học máy","blockchain","Python"]},

    {"id":"ND07","category":"mo_ta_nganh",
     "question":"Ngành Mạng máy tính HaUI học gì?",
     "expected":"Mạng máy tính và truyền thông dữ liệu (mã 7480102, 70 chỉ tiêu) học TCP/IP, định tuyến, chuyển mạch, bảo mật mạng, IoT, Cloud, 5G, VPN, tường lửa. Dùng công cụ Cisco Packet Tracer, Wireshark, GNS3",
     "keywords":["TCP/IP","IoT","Cloud","Cisco","5G"]},

    {"id":"ND08","category":"mo_ta_nganh",
     "question":"Ngành Công nghệ đa phương tiện HaUI học gì ra làm gì?",
     "expected":"Đa phương tiện (mã 74802012, 60 chỉ tiêu) học thiết kế đồ họa, lập trình web, phát triển game, sản xuất video, hoạt hình 2D/3D, VFX, UX/UI. Ra làm designer, web developer, game developer, VFX artist",
     "keywords":["đồ họa","game","UX/UI","video","web"]},

    {"id":"ND09","category":"mo_ta_nganh",
     "question":"Ngành Công nghệ kỹ thuật máy tính HaUI học gì?",
     "expected":"Công nghệ kỹ thuật máy tính (mã 7480108, 140 chỉ tiêu) học thiết kế phần cứng, hệ thống nhúng, vi mạch điện tử, tự động hóa, tích hợp phần cứng-phần mềm. Ra làm kỹ sư phần cứng, kỹ sư hệ thống nhúng",
     "keywords":["phần cứng","hệ thống nhúng","vi mạch","tự động hóa"]},

    {"id":"ND10","category":"mo_ta_nganh",
     "question":"Trường CNTT và Truyền thông HaUI có những ngành nào?",
     "expected":"Trường CNTT & Truyền thông HaUI có 8 ngành: Công nghệ thông tin (360 chỉ tiêu), Kỹ thuật phần mềm (240), Công nghệ kỹ thuật máy tính (140), Hệ thống thông tin (120), Khoa học máy tính (120), Mạng máy tính (70), Công nghệ đa phương tiện (60), An toàn thông tin (40)",
     "keywords":["360","240","140","8 ngành"]},

    # ════════════════════════════════════════════════════════
    # FAQ & HƯỚNG DẪN — 10 câu (từ các file .md)
    # ════════════════════════════════════════════════════════
    {"id":"FAQ01","category":"faq",
     "question":"Làm sao để đăng ký xét tuyển vào HaUI?",
     "expected":"Đăng ký tại xettuyen.haui.edu.vn từ 15/5 đến 05/7. Tạo tài khoản, nhập điểm học bạ/chứng chỉ, upload minh chứng, đăng ký nguyện vọng, nộp lệ phí 50.000 đồng qua QR ngân hàng",
     "keywords":["xettuyen.haui.edu.vn","50.000","15/5"]},

    {"id":"FAQ02","category":"faq",
     "question":"HaUI có những loại học bổng nào cho sinh viên?",
     "expected":"HaUI có 4 loại: Học bổng HaUI (đầu vào toàn khóa/năm 1/5 triệu), Học bổng KKHT (theo kết quả học kỳ), Học bổng Nguyễn Thanh Bình (hoàn cảnh khó khăn), và học bổng tài trợ từ doanh nghiệp",
     "keywords":["HaUI","KKHT","Nguyễn Thanh Bình"]},

    {"id":"FAQ03","category":"faq",
     "question":"Điều kiện duy trì học bổng HaUI hàng kỳ như thế nào?",
     "expected":"Mỗi học kỳ chính cần đạt: điểm TBC ≥ 2.5, rèn luyện loại Tốt trở lên, tổng tín chỉ xét ≥ 15 tín chỉ",
     "keywords":["2.5","15","Tốt"]},

    {"id":"FAQ04","category":"faq",
     "question":"Ký túc xá HaUI giá phòng bao nhiêu một tháng?",
     "expected":"Phòng CLC: 3 người 800.000đ, 4 người 600.000đ, 6 người 400.000đ/tháng. Phòng tiêu chuẩn CS1: 4 người 465.000đ, 6 người 310.000đ. CS2: 4 người 420.000đ, 6 người 280.000đ",
     "keywords":["800.000","600.000","280.000"]},

    {"id":"FAQ05","category":"faq",
     "question":"Thí sinh tốt nghiệp năm 2023 có đăng ký PT4 PT5 được không?",
     "expected":"Không. PT2, PT4, PT5 chỉ dành cho thí sinh tốt nghiệp THPT năm 2025, thí sinh tốt nghiệp trước 2025 không được đăng ký các phương thức này",
     "keywords":["không","2025"]},

    {"id":"FAQ06","category":"faq",
     "question":"Lệ phí đăng ký xét tuyển HaUI là bao nhiêu tiền?",
     "expected":"Lệ phí đăng ký xét tuyển là 50.000 đồng/hồ sơ, nộp qua mã QR ngân hàng BIDV hoặc Ngân hàng Lộc Phát",
     "keywords":["50.000","BIDV"]},

    {"id":"FAQ07","category":"faq",
     "question":"Học bổng toàn khóa HaUI cần điều kiện gì?",
     "expected":"Đoạt giải Nhất/Nhì/Ba HSG quốc gia/quốc tế hoặc KHKT cấp quốc gia, hoặc thủ khoa tổ hợp A00/A01/D01, hoặc thủ khoa PT2/PT4/PT5",
     "keywords":["HSG","thủ khoa","toàn khóa"]},

    {"id":"FAQ08","category":"faq",
     "question":"Trúng tuyển HaUI rồi phải làm gì để nhập học?",
     "expected":"Nhập học trực tuyến tại nhaphoc.haui.edu.vn hoặc app MyHaUI: chọn NHẬP HỌC, nhập mã hồ sơ trúng tuyển, nộp học phí kỳ 1 qua QR BIDV hoặc Lộc Phát, sau đó nộp hồ sơ bản cứng",
     "keywords":["nhaphoc.haui.edu.vn","MyHaUI"]},

    {"id":"FAQ09","category":"faq",
     "question":"haui có hỗ trợ ký túc xá không đăng ký ở đâu?",
     "expected":"HaUI có ký túc xá hơn 800 phòng khép kín tại cơ sở 1 và 2. Đăng ký tại ssc.haui.edu.vn hoặc liên hệ trực tiếp khi nhập học",
     "keywords":["800","ssc.haui.edu.vn"]},

    {"id":"FAQ10","category":"faq",
     "question":"Xét tuyển bằng học bạ vào HaUI cần điều kiện gì?",
     "expected":"PT2 (học bạ + HSG/chứng chỉ): điểm TB môn tổ hợp lớp 10-11-12 ≥ 7.0, kèm giải HSG cấp tỉnh (Nhất/Nhì/Ba) hoặc chứng chỉ quốc tế (IELTS ≥ 5.5, SAT ≥ 1000, TOEFL iBT ≥ 50...)",
     "keywords":["7.0","HSG","IELTS"]},

    # ════════════════════════════════════════════════════════
    # ĐẬU TRƯỢT — 10 câu (số liệu chính xác từ JSON)
    # ════════════════════════════════════════════════════════
    # Công thức: 24đ KV2-NT → [(30-24)/7.5]*0.5=0.40 → xét=24.40 < 25.22 → TRƯỢT
    {"id":"DT01","category":"dau_truot",
     "question":"24 điểm KV2-NT thi THPT có đậu ngành CNTT không?",
     "expected":"24đ KV2-NT: UT=0.40 → xét=24.40. CNTT PT3: 2024=25.22, 2023=25.19, 2025/chung=23.09. Điểm xét thấp hơn 2024 (thiếu 0.82) → khả năng thấp với điểm hiện tại",
     "keywords":["24.40","25.22","thấp hơn"]},

    # TSA 80 → 29.0, KV1: [(30-29)/7.5]*0.75=0.10 → xét=29.10 > 18.5 → ĐẬU
    {"id":"DT02","category":"dau_truot",
     "question":"TSA 80 điểm KV1 có đậu ngành CNTT không?",
     "expected":"TSA 80→29.0. KV1: UT=0.10 → xét=29.10. CNTT PT5: 2024=18.5, 2025/chung=23.09. Điểm xét 29.10 cao hơn cả 3 năm → khả năng rất cao",
     "keywords":["29.10","18.5","cao hơn"]},

    # 25đ KV3 → không ưu tiên → xét=25.0 > 24.68 → ĐẬU
    {"id":"DT03","category":"dau_truot",
     "question":"25 điểm THPT KV3 có đậu ngành Kỹ thuật phần mềm không?",
     "expected":"25đ KV3 không ưu tiên → xét=25.0. KTPM PT3: 2024=24.68, 2023=24.54, 2025/chung=21.75. Điểm xét cao hơn cả 3 năm → khả năng cao",
     "keywords":["25.0","24.68","cao hơn"]},

    # HSA 105 → 25.0, KV2-NT: [(30-25)/7.5]*0.5=0.33 → xét=25.33 < 27.69 → TRƯỢT
    {"id":"DT04","category":"dau_truot",
     "question":"HSA 105 KV2-NT có đậu ngành Kế toán không?",
     "expected":"HSA 105→25.0. KV2-NT: UT=0.33 → xét=25.33. Kế toán PT4/học bạ: 2024=27.69, 2023=28.0. Điểm xét thấp hơn 2024 (thiếu 2.36) → khả năng thấp với PT4. Nên thử PT3",
     "keywords":["25.33","27.69","thấp hơn"]},

    # 26đ KV3 → xét=26.0 > 25.89 → ĐẬU
    {"id":"DT05","category":"dau_truot",
     "question":"26 điểm THPT không có khu vực ưu tiên có đậu Logistics không?",
     "expected":"26đ KV3 không ưu tiên → xét=26.0. Logistics PT3: 2024=25.89, 2023=24.88, 2025/chung=23.5. Điểm xét cao hơn cả 3 năm → khả năng cao",
     "keywords":["26.0","25.89","cao hơn"]},

    # 23đ KV1: dưới 22.5 → cộng thẳng +0.75 → xét=23.75 < 25.33 → TRƯỢT
    {"id":"DT06","category":"dau_truot",
     "question":"em được 23 điểm thi THPT KV1 thì có vào được ngành Marketing không",
     "expected":"23đ KV1: dưới 22.5 → +0.75 → xét=23.75. Marketing PT3: 2024=25.33, 2023=24.28, 2025/chung=22.5. Điểm xét cao hơn 2025 nhưng thấp hơn 2024 → khả năng trung bình",
     "keywords":["23.75","25.33","thấp hơn"]},

    # TSA 85 → 30.0, KV2: [(30-30)/7.5]*0.25=0 → xét=30.0 > 16.5 → ĐẬU
    {"id":"DT07","category":"dau_truot",
     "question":"TSA 85 điểm KV2 có vào được ngành Robot AI không?",
     "expected":"TSA 85→30.0 (tối đa). KV2: UT=0 → xét=30.0. Robot PT5: 2024=16.5, 2025/chung=24.3. Điểm xét 30.0 cao hơn cả 3 năm → khả năng rất cao",
     "keywords":["30.0","16.5","cao hơn"]},

    # Học bạ 27đ vs CNTT PT2=27.0 → SÁT NÚT đúng điểm chuẩn
    {"id":"DT08","category":"dau_truot",
     "question":"Điểm học bạ xét tuyển 27 có đậu ngành CNTT 2024 không?",
     "expected":"Học bạ 27đ → xét=27.0. CNTT PT2/học bạ: 2024=27.0, 2023=28.93, 2025/chung=23.09. Bằng đúng điểm 2024 — sát nút, rủi ro cao. Nên cân nhắc ngành phụ",
     "keywords":["27","27.0","ngang bằng"]},

    # 24.5đ KV1: [(30-24.5)/7.5]*0.75=0.55 → xét=25.05 > 24.4 → ĐẬU
    {"id":"DT09","category":"dau_truot",
     "question":"24.5 điểm THPT KV1 thi vào ngành Điện tử viễn thông có đậu không?",
     "expected":"24.5đ KV1: UT=0.55 → xét=25.05. Điện tử VT PT3: 2024=24.4, 2025/chung=22.75. Điểm xét cao hơn cả 3 năm → khả năng cao",
     "keywords":["25.05","24.4","cao hơn"]},

    # 22đ KV3 → không ưu tiên → xét=22.0 < 24.01 → TRƯỢT
    {"id":"DT10","category":"dau_truot",
     "question":"em 22 điểm thi THPT không có ưu tiên muốn vào ngành kế toán được không",
     "expected":"22đ KV3 không ưu tiên → xét=22.0. Kế toán PT3: 2024=24.01, 2023=23.6, 2025/chung=20.0. Điểm xét cao hơn 2025 nhưng thấp hơn 2024 → khả năng trung bình",
     "keywords":["22.0","24.01","thấp hơn"]},


    # ════════════════════════════════════════════════════════
    # TỔNG HỢP ĐA NGUỒN — 10 câu
    # Mỗi câu yêu cầu bot kết hợp 2-4 nguồn thông tin khác nhau
    # ════════════════════════════════════════════════════════

    # --- Loại 1: Tính điểm + đậu/trượt + học phí (3 nguồn) ---
    {"id":"TH_01","category":"tong_hop",
     "question":"Em thi THPT được 24 điểm khu vực 1, muốn vào ngành CNTT. Điểm xét tuyển bao nhiêu, có đậu không, và nếu đậu thì học phí một tín chỉ là bao nhiêu?",
     "expected":"24đ KV1: UT=[(30-24)/7.5]×0.75=0.60 → điểm xét=24.60. Điểm chuẩn CNTT 2024 PT3=25.22 → TRƯỢT. Nếu đậu: học phí K20 đại trà 700.000đ/tín chỉ",
     "keywords":["24.60","25.22","thấp hơn","700.000"]},

    {"id":"TH_02","category":"tong_hop",
     "question":"Tôi thi TSA được 80 điểm KV2-NT, muốn vào Kỹ thuật phần mềm. Cho biết điểm xét tuyển, kết quả đậu trượt và học phí nếu trúng tuyển.",
     "expected":"TSA 80 → 29.0. KV2-NT: [(30-29)/7.5]×0.5=0.07 → điểm xét=29.07. Điểm chuẩn KTPM 2024 PT5=16.01 → ĐẬU. Học phí K20 đại trà 700.000đ/tín chỉ",
     "keywords":["29.07","16.01","cao hơn","700.000"]},

    # --- Loại 2: Điểm chuẩn + tổ hợp + mô tả ngành (3 nguồn) ---
    {"id":"TH_03","category":"tong_hop",
     "question":"Ngành An toàn thông tin HaUI: điểm chuẩn 2024, tổ hợp xét tuyển và ra trường làm gì?",
     "expected":"ATTT 2024 PT3=25.12 (hoặc tra điểm chuẩn thực tế). Tổ hợp A00, A01, X06, X07. Ra làm: pentester, SOC analyst, kỹ sư an ninh mạng, chuyên viên ATTT tại ngân hàng/tập đoàn",
     "keywords":["A00","A01","an ninh mạng","pentester"]},

    {"id":"TH_04","category":"tong_hop",
     "question":"Ngành Kỹ thuật phần mềm HaUI tuyển sinh năm 2024 như thế nào? Cho biết điểm chuẩn, tổ hợp và mô tả ngành học.",
     "expected":"KTPM 2024: PT3=24.68, PT5=16.01. Tổ hợp A00, A01, X06, X07. Ngành học Agile/Scrum, DevOps, kiểm thử, lập trình Java/Python/C#. 240 chỉ tiêu",
     "keywords":["24.68","A00","A01","Agile","240"]},

    # --- Loại 3: Học phí + học bổng + điều kiện (2 nguồn) ---
    {"id":"TH_05","category":"tong_hop",
     "question":"Học K20 đại trà ở HaUI tốn bao nhiêu tiền? Có học bổng nào không và điều kiện như thế nào?",
     "expected":"Học phí K20 đại trà 700.000đ/tín chỉ. Có 3 loại học bổng đầu vào: toàn khóa (thủ khoa/HSG quốc gia), năm 1 (điểm cao), 5 triệu. Duy trì: TBC≥2.5, rèn luyện Tốt, ≥15 tín chỉ/kỳ",
     "keywords":["700.000","toàn khóa","2.5","15"]},

    {"id":"TH_06","category":"tong_hop",
     "question":"Nếu tôi trúng tuyển HaUI thì một năm học tốn khoảng bao nhiêu tiền? Có hỗ trợ học bổng không?",
     "expected":"K20 đại trà 700.000đ/tín chỉ, mỗi kỳ khoảng 20-25 tín chỉ → ~14-17.5 triệu/kỳ. Có học bổng KKHT hàng kỳ nếu TBC≥2.5, rèn luyện Tốt, ≥15 tín chỉ",
     "keywords":["700.000","KKHT","2.5","học bổng"]},

    # --- Loại 4: So sánh 2 ngành cùng lúc ---
    {"id":"TH_07","category":"tong_hop",
     "question":"So sánh ngành CNTT và Kỹ thuật phần mềm ở HaUI: điểm chuẩn, chỉ tiêu, nội dung học và cơ hội việc làm khác nhau thế nào?",
     "expected":"CNTT: 360 chỉ tiêu, PT3-2024=25.22, học rộng (AI/mạng/bảo mật/dữ liệu). KTPM: 240 chỉ tiêu, PT3-2024=24.68, học chuyên sâu phần mềm (Agile/DevOps/kiểm thử). Cả hai tổ hợp A00/A01/X06/X07",
     "keywords":["360","240","25.22","24.68","Agile"]},

    {"id":"TH_08","category":"tong_hop",
     "question":"Khoa học máy tính và CNTT ở HaUI khác nhau gì? Nên chọn ngành nào?",
     "expected":"KHMT (120 chỉ tiêu) thiên nghiên cứu: AI sâu, học máy, blockchain, thuật toán. CNTT (360 chỉ tiêu) thiên thực hành: phát triển phần mềm, mạng, hệ thống. CNTT phù hợp muốn đi làm ngay, KHMT phù hợp muốn nghiên cứu/học cao hơn",
     "keywords":["120","360","blockchain","thuật toán","học máy"]},

    # --- Loại 5: Tư vấn toàn diện ---
    {"id":"TH_09","category":"tong_hop",
     "question":"Em được 26 điểm THPT KV3, muốn học ngành liên quan đến bảo mật máy tính. HaUI có ngành nào phù hợp, điểm chuẩn bao nhiêu, học phí và cơ hội việc làm thế nào?",
     "expected":"Ngành phù hợp: An toàn thông tin (7480202). Điểm 26 KV3 không ưu tiên, cần tra điểm chuẩn ATTT. Học phí K20 700.000đ/tín chỉ. Ra làm: pentester, SOC analyst, kỹ sư an ninh mạng",
     "keywords":["An toàn thông tin","7480202","700.000","pentester"]},

    {"id":"TH_10","category":"tong_hop",
     "question":"HSA 110 điểm KV2 muốn vào CNTT HaUI. Tư vấn đầy đủ: điểm xét tuyển, kết quả, học phí, học bổng có thể nhận được.",
     "expected":"HSA 110→25.75. KV2: [(30-25.75)/7.5]×0.25=0.14 → điểm xét=25.89. Điểm chuẩn CNTT 2024 PT4=28.89 → TRƯỢT (thiếu 3.0 điểm). Nếu thử PT3: cần thi THPT ≥25.22. Học phí K20 700.000đ/tín chỉ",
     "keywords":["25.75","25.89","28.89","thấp hơn","700.000"]},

    # ════════════════════════════════════════════════════════
    # EDGE CASES — 10 câu (ngôn ngữ tự nhiên, câu thực tế)
    # ════════════════════════════════════════════════════════
    {"id":"E01","category":"edge",
     "question":"haui thuộc bộ gì",
     "expected":"HaUI trực thuộc Bộ Công Thương",
     "keywords":["Bộ Công Thương"]},

    {"id":"E02","category":"edge",
     "question":"trường công nghiệp hà nội địa chỉ ở đâu",
     "expected":"Địa chỉ chính: 298 Cầu Diễn, Minh Khai, Bắc Từ Liêm, Hà Nội",
     "keywords":["298","Cầu Diễn"]},

    {"id":"E03","category":"edge",
     "question":"haui có ký túc xá không bao nhiêu tiền",
     "expected":"HaUI có ký túc xá hơn 800 phòng khép kín, giá từ 280.000 đến 800.000 đồng/tháng tùy loại phòng và cơ sở",
     "keywords":["800","280.000","800.000"]},

    {"id":"E04","category":"edge",
     "question":"học ở haui có việc làm không",
     "expected":"Tỷ lệ sinh viên HaUI có việc làm sau tốt nghiệp 1 năm đạt trên 95%, trường có quan hệ với nhiều doanh nghiệp đối tác",
     "keywords":["95%","việc làm"]},

    {"id":"E05","category":"edge",
     "question":"cntt hay ktpm chọn ngành nào tốt hơn",
     "expected":"Cả hai đều tốt: CNTT rộng hơn (AI, dữ liệu, mạng), KTPM tập trung phát triển phần mềm (Agile, DevOps, kiểm thử). Nên chọn theo định hướng nghề nghiệp cụ thể của bản thân",
     "keywords":["CNTT","phần mềm","Agile"]},

    {"id":"E06","category":"edge",
     "question":"mình thi tsa được 70 điểm thì quy đổi được bao nhiêu",
     "expected":"TSA 70 điểm quy đổi thành 27.75 điểm thang 30 theo bảng quy đổi của HaUI",
     "keywords":["70","27.75"]},

    {"id":"E07","category":"edge",
     "question":"haui có chương trình liên kết quốc tế không",
     "expected":"HaUI có chương trình liên kết 2+2 Ngôn ngữ Trung Quốc với Đại học Khoa học Kỹ thuật Quảng Tây (học 2 năm HaUI + 2 năm Trung Quốc, nhận 2 bằng), và các chương trình đào tạo bằng tiếng Anh",
     "keywords":["Quảng Tây","tiếng Anh","2+2"]},

    {"id":"E08","category":"edge",
     "question":"đăng ký xét tuyển haui ở đâu hạn chót khi nào",
     "expected":"Đăng ký tại xettuyen.haui.edu.vn, hạn chót 05/7 hàng năm (PT2, PT4, PT5). Lệ phí 50.000 đồng. Năm 2026 mở từ 15/5/2026.",
     "keywords":["xettuyen.haui.edu.vn","05/7","50.000"]},

    {"id":"E09","category":"edge",
     "question":"haui kiểm định chất lượng đạt chuẩn gì chưa",
     "expected":"HaUI có 5 chương trình đào tạo đạt chuẩn kiểm định ABET (Hoa Kỳ) — một trong những trường kỹ thuật hàng đầu Việt Nam đạt chuẩn quốc tế này",
     "keywords":["ABET","5","kiểm định"]},

    {"id":"E10","category":"edge",
     "question":"năm 2026 haui tuyển bao nhiêu chỉ tiêu đại học chính quy",
     "expected":"Năm 2026 HaUI dự kiến tuyển 8.300 chỉ tiêu đại học chính quy trong tổng 9.420 chỉ tiêu các hệ",
     "keywords":["8.300","9.420","2026"]},
    # ════════════════════════════════════════════════════════════════
    # PHỨC TẠP — 20 câu đòi hỏi suy luận nhiều bước
    # Bao gồm 6 dạng còn thiếu: multi-turn, KV+đối tượng, không dấu,
    # so sánh 3 ngành, profile học sinh, clarify/pending
    # ════════════════════════════════════════════════════════════════

    # ── Dạng 1: Tính toán phức tạp KV + đối tượng ưu tiên ────────────
    # KV1 (+0.75) + đối tượng 02 (+2.0) → tổng ưu tiên tối đa +2.75
    # Nhưng điểm < 22.5 → cộng thẳng
    {"id":"CX01","category":"complex",
     "question":"Em được 20 điểm thi THPT, ở khu vực 1 và là con thương binh (đối tượng 02). Tổng điểm xét tuyển của em là bao nhiêu?",
     "expected":"20 điểm < 22.5 nên cộng thẳng. UT khu vực KV1=+0.75, đối tượng 02 nhóm UT1=+2.0. Tổng ưu tiên=+2.75. Điểm xét tuyển=20+2.75=22.75",
     "keywords":["22.75","2.75","KV1","đối tượng"]},

    # KV2-NT + đối tượng 05 (+1.0), điểm ≥22.5 → giảm dần
    {"id":"CX02","category":"complex",
     "question":"HSA 100 điểm, khu vực 2 nông thôn, đối tượng 05. Điểm xét tuyển là bao nhiêu?",
     "expected":"HSA 100→24.25. KV2-NT=+0.5, đối tượng 05 nhóm UT2=+1.0. Tổng ưu tiên gốc=1.5. 24.25≥22.5 nên giảm dần: [(30-24.25)/7.5]×1.5=1.15. Điểm xét=24.25+1.15=25.40",
     "keywords":["24.25","1.15","25.40","đối tượng 05"]},

    # ── Dạng 2: Multi-turn — câu phụ thuộc ngữ cảnh ──────────────────
    # (Bot phải nhớ ngành từ câu trước)
    {"id":"CX03","category":"complex",
     "question":"Ngành CNTT HaUI điểm chuẩn 2024 bao nhiêu và học phí một tín chỉ là bao nhiêu?",
     "expected":"CNTT 2024 PT3=25.22, PT5=18.5, PT4=28.89. Học phí K20 đại trà 700.000đ/tín chỉ, K20 tiếng Anh 1.000.000đ/tín chỉ",
     "keywords":["25.22","700.000","CNTT"]},

    {"id":"CX04","category":"complex",
     "question":"Ngành đó (CNTT) xét tổ hợp nào và ra trường làm được gì?",
     "expected":"CNTT xét tổ hợp A00, A01, X06, X07. Ra làm lập trình viên, kỹ sư phần mềm, kỹ sư AI/ML, phân tích dữ liệu, bảo mật thông tin",
     "keywords":["A00","A01","X06","lập trình"]},

    # ── Dạng 3: Câu không dấu / viết tắt teen ────────────────────────
    {"id":"CX05","category":"complex",
     "question":"cho hoi diem chuan nganh cntt nam 2024 bao nhieu vay",
     "expected":"Điểm chuẩn ngành Công nghệ thông tin năm 2024: PT3 (THPT) là 25.22, PT5 (TSA) là 18.5, PT4 (HSA) là 28.89",
     "keywords":["25.22","18.5","2024"]},

    {"id":"CX06","category":"complex",
     "question":"thi dc 22 diem kv1 co vao cntt dc ko",
     "expected":"22 điểm KV1: dưới 22.5 cộng thẳng +0.75. Điểm xét=22.75. Điểm chuẩn CNTT 2024 PT3=25.22. Kết quả: TRƯỢT, thiếu 2.47 điểm",
     "keywords":["22.75","25.22","thấp hơn"]},

    {"id":"CX07","category":"complex",
     "question":"hoc phi 1 tin chi la bao nhieu vay ad",
     "expected":"Học phí tính theo chương trình: K20 đại trà 700.000đ/tín chỉ, K20 tiếng Anh 1.000.000đ/tín chỉ, K19 là 550.000đ/tín chỉ",
     "keywords":["700.000","550.000","tín chỉ"]},

    # ── Dạng 4: So sánh 3+ ngành cùng lúc ────────────────────────────
    {"id":"CX08","category":"complex",
     "question":"So sánh điểm chuẩn 2024 của 3 ngành CNTT, Kỹ thuật phần mềm và Khoa học máy tính",
     "expected":"CNTT (7480201): PT3=25.22, PT5=18.5. KTPM (7480103): PT3=24.68, PT5=16.01. KHMT (7480101): chỉ có PT3/PT5. Ba ngành cùng tổ hợp A00/A01/X06/X07",
     "keywords":["25.22","24.68","CNTT","KTPM","KHMT"]},

    {"id":"CX09","category":"complex",
     "question":"Ngành nào trong số CNTT, KTPM, An toàn thông tin có điểm chuẩn thấp nhất năm 2024?",
     "expected":"KTPM PT3=24.68, CNTT PT3=25.22, ATTT cần tra. Trong 3 ngành theo PT3: KTPM có điểm chuẩn thấp nhất (24.68)",
     "keywords":["24.68","KTPM","thấp nhất"]},

    # ── Dạng 5: Profile đầy đủ học sinh ──────────────────────────────
    {"id":"CX10","category":"complex",
     "question":"Em học ở Hà Nội (KV3), thi THPT được 26 điểm, thích lập trình và có thể học phí tầm 15-20 triệu/học kỳ. Nên đăng ký ngành nào ở HaUI?",
     "expected":"26 điểm KV3 không ưu tiên, điểm xét=26.0. Phù hợp: CNTT (chuẩn 25.22 ĐẬU), KTPM (chuẩn 24.68 ĐẬU), KHMT (chuẩn ~25 ĐẬU). Học phí K20 đại trà ~700k/tín chỉ, khoảng 14-17 triệu/kỳ — phù hợp ngân sách",
     "keywords":["26.0","cao hơn","CNTT","KTPM","700.000"]},

    {"id":"CX11","category":"complex",
     "question":"Em là học sinh dân tộc thiểu số ở vùng núi (KV1, đối tượng 01), thi THPT 18 điểm. Em có vào được HaUI không và ngành nào phù hợp?",
     "expected":"18 điểm KV1 + đối tượng 01 nhóm UT1 (+2.0): dưới 22.5 cộng thẳng 0.75+2.0=2.75. Điểm xét=20.75. Điểm chuẩn các ngành 2025 (sàn chung): CNTT=23.09, KTPM=21.75 → KTPM có thể phù hợp hơn",
     "keywords":["20.75","21.75","KTPM","đối tượng 01"]},

    {"id":"CX12","category":"complex",
     "question":"Em thi TSA 75 điểm, KV2, muốn học ngành liên quan đến bảo mật máy tính. Cho em biết điểm xét tuyển, kết quả, học phí và cơ hội việc làm.",
     "expected":"TSA 75→28.5. KV2: [(30-28.5)/7.5]×0.25=0.05. Điểm xét=28.55. Ngành An toàn thông tin PT5=cần tra. Học phí K20=700.000đ/tín chỉ. Việc làm: pentester, SOC analyst, kỹ sư an ninh mạng",
     "keywords":["28.55","28.5","An toàn thông tin","700.000"]},

    # ── Dạng 6: Bot phải hỏi lại / clarify ───────────────────────────
    {"id":"CX13","category":"complex",
     "question":"Em được 25 điểm, có vào HaUI được không?",
     "expected":"Bot cần hỏi thêm: phương thức thi (THPT/TSA/HSA), ngành muốn vào, khu vực. Với 25 điểm thi THPT KV3: CNTT chuẩn 25.22 → sát nút, KTPM chuẩn 24.68 → ĐẬU",
     "keywords":["phương thức","ngành","25.22","24.68"]},

    {"id":"CX14","category":"complex",
     "question":"Ngành kỹ thuật ở HaUI học phí bao nhiêu?",
     "expected":"HaUI có nhiều ngành kỹ thuật với học phí khác nhau. K20 đại trà 700.000đ/tín chỉ, K20 tiếng Anh 1.000.000đ/tín chỉ, Kỹ sư K3 đại trà 700.000đ/tín chỉ. Bạn muốn hỏi ngành kỹ thuật cụ thể nào?",
     "keywords":["700.000","kỹ thuật","ngành"]},

    # ── Dạng 7: Suy luận ngược — từ điểm gợi ý ngành ─────────────────
    {"id":"CX15","category":"complex",
     "question":"Em có 23 điểm thi THPT KV2-NT, không có đối tượng ưu tiên. Em có thể đăng ký những ngành nào tại HaUI mà khả năng đậu cao?",
     "expected":"23 điểm KV2-NT: 23≥22.5 nên giảm dần: [(30-23)/7.5]×0.5=0.47. Điểm xét=23.47. Có thể xét: KTPM (chuẩn 2025: 21.75 ĐẬU), Kế toán (chuẩn 2025: 20.0 ĐẬU), Marketing (22.5 SÁT NÚT), CNTT (23.09 SÁT NÚT)",
     "keywords":["23.47","21.75","20.0","KTPM","Kế toán"]},

    {"id":"CX16","category":"complex",
     "question":"HSA 95 điểm KV1, muốn thi vào ngành có điểm chuẩn thấp để chắc đậu. Gợi ý cho em ngành nào?",
     "expected":"HSA 95→23.5. KV1: [(30-23.5)/7.5]×0.75=0.65. Điểm xét=24.15. Các ngành PT4-2024 dưới 24.15: Robot (28.44 TRƯỢT), Kế toán (27.69 TRƯỢT)... Nên thử PT3 với điểm thi THPT hoặc thi lại HSA để cải thiện",
     "keywords":["23.5","24.15","KV1","điểm xét"]},

    # ── Dạng 8: Tổng hợp 3+ nguồn phức tạp ──────────────────────────
    {"id":"CX17","category":"complex",
     "question":"Em 26 điểm THPT KV2-NT muốn vào CNTT. Tính điểm xét tuyển, cho biết có đậu không, học phí, học bổng có thể nhận và điều kiện duy trì.",
     "expected":"26 điểm KV2-NT: [(30-26)/7.5]×0.5=0.27. Điểm xét=26.27. CNTT PT3=25.22 → ĐẬU. Học phí K20=700.000đ/tín chỉ. Học bổng: nếu top điểm → năm 1 hoặc 5tr. Duy trì: TBC≥2.5, rèn luyện Tốt, ≥15 tín chỉ/kỳ",
     "keywords":["26.27","cao hơn","700.000","2.5","15"]},

    {"id":"CX18","category":"complex",
     "question":"So sánh ngành CNTT và An toàn thông tin: điểm chuẩn, tổ hợp, nội dung học và cơ hội việc làm khác nhau thế nào?",
     "expected":"CNTT: 360 chỉ tiêu, PT3=25.22, tổ hợp A00/A01/X06/X07, học rộng (AI/mạng/CSDL). ATTT: 40 chỉ tiêu, tổ hợp A00/A01/X06/X07, học chuyên sâu bảo mật (pentesting/mã hóa/SOC). ATTT hẹp và chuyên hơn, phù hợp muốn làm bảo mật",
     "keywords":["360","40","25.22","pentesting","bảo mật"]},

    {"id":"CX19","category":"complex",
     "question":"Tôi học xong THPT năm 2023, có bằng IELTS 6.5, muốn vào CNTT HaUI năm 2026. Tôi đăng ký được phương thức nào và cần chuẩn bị gì?",
     "expected":"Tốt nghiệp 2023 KHÔNG được đăng ký PT2/PT4/PT5 (chỉ dành cho tốt nghiệp năm hiện tại). Chỉ còn PT3 (thi THPT 2026). IELTS 6.5 không thay thế điểm thi THPT. Cần đăng ký thi lại môn trong tổ hợp xét tuyển cùng HS lớp 12 năm 2026",
     "keywords":["không","PT2","PT3","IELTS","2026"]},

    {"id":"CX20","category":"complex",
     "question":"Em thi HSA được 120 điểm, KV2-NT, đối tượng 06. Cho em biết điểm xét tuyển ngành KTPM theo PT5 và khả năng như thế nào? Học phí năm 1 bao nhiêu?",
     "expected":"HSA 120→27.05. KV2-NT=+0.5, đối tượng 06=+1.0 → UT=0.59 (giảm dần). Điểm xét=27.64. KTPM PT5: 2024=16.01, 2025/chung=21.75. Điểm xét 27.64 cao hơn cả 3 năm → khả năng rất cao. Học phí K20 đại trà 700.000đ/tín chỉ, khoảng 14 triệu/học kỳ",
     "keywords":["27.05","27.64","16.01","700.000","14"]},
]


@dataclass
class EvalResult:
    id: str; category: str; question: str; answer: str; expected: str
    keywords_hit: list; keywords_miss: list
    score: float; feedback: str; latency_ms: int; intent: str


def judge_answer(client, question, answer, expected):
    ans_safe = answer.replace('"',"'").replace('\n',' ')[:500]
    exp_safe = expected.replace('"',"'")
    prompt = f"""Chấm điểm câu trả lời chatbot tuyển sinh HaUI (thang 0-1):
Q: {question}
A: {ans_safe}
Expected: {exp_safe}

Thang điểm:
1.0 = Chính xác, đủ số liệu, rõ ràng
0.8 = Đúng nhưng thiếu 1-2 chi tiết nhỏ
0.6 = Đúng hướng nhưng thiếu số liệu cụ thể
0.4 = Sai một phần đáng kể (sai số liệu hoặc kết luận)
0.2 = Sai số liệu chính hoặc kết luận ngược
0.0 = Từ chối trả lời dù có dữ liệu

Trả về JSON 1 dòng: {{"score":0.8,"feedback":"nhận xét ngắn"}}"""
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=120,
        messages=[{"role":"user","content":prompt}])
    raw = re.sub(r'```json|```','',resp.content[0].text).strip()
    try:
        d = json.loads(raw)
        return float(d["score"]), str(d.get("feedback",""))
    except:
        ms = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        mf = re.search(r'"feedback"\s*:\s*"([^"]{0,150})"', raw)
        return float(ms.group(1)) if ms else 0.5, mf.group(1) if mf else "parse error"


def _kw_hit(answer: str, keywords: list):
    """
    Keywords linh hoạt:
    - Bỏ qua hoa/thường
    - Số tiền: 700.000 ≈ 700,000 ≈ 700000
    - Số điểm: 25.22 ≈ 25,22
    """
    al = answer.lower()
    hits, misses = [], []
    for kw in keywords:
        kw_lower = kw.lower()
        found = kw_lower in al
        if not found:
            # normalize: bỏ dấu chấm và phẩy trong số
            kw_norm = re.sub(r'[.,]', '', kw_lower)
            al_norm  = re.sub(r'[.,]', '', al)
            found = len(kw_norm) >= 2 and kw_norm in al_norm
        hits.append(kw) if found else misses.append(kw)
    return hits, misses


def run_evaluation(quick=False, save=False, category=""):
    from datetime import datetime
    print("="*70)
    print(f"  RAG Evaluation v3 — Chatbot Tuyển sinh HaUI")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    from src.pipeline.chatbot import Chatbot
    print("\nKhởi tạo pipeline...")
    bot = Chatbot()
    client = anthropic.Anthropic()
    print("✓ Sẵn sàng\n")

    ds = EVAL_DATASET
    if category: ds = [d for d in ds if d["category"] == category]
    if quick:    ds = ds[:20]
    print(f"Tổng: {len(ds)} câu hỏi\n")

    results = []
    for i, item in enumerate(ds, 1):
        print(f"[{i:02d}/{len(ds)}] [{item['id']}] {item['question'][:58]}...")
        t0 = time.time()
        try:
            resp   = bot.chat(item["question"])
            answer = resp.answer
            intent = resp.intent_type if hasattr(resp, "intent_type") else ""
            bot.reset()
        except Exception as e:
            answer = f"ERROR: {e}"
            intent = "ERROR"
        lat = int((time.time() - t0) * 1000)

        hits, miss = _kw_hit(answer, item["keywords"])
        kw_rate = len(hits) / max(len(hits)+len(miss), 1)

        try:    score, fb = judge_answer(client, item["question"], answer, item["expected"])
        except Exception as e: score, fb = 0.5, f"judge error: {e}"

        results.append(EvalResult(
            id=item["id"], category=item["category"],
            question=item["question"], answer=answer,
            expected=item["expected"],
            keywords_hit=hits, keywords_miss=miss,
            score=score, feedback=fb,
            latency_ms=lat, intent=str(intent)))

        icon = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
        kw_icon = "✓" if kw_rate == 1.0 else "~" if kw_rate > 0 else "✗"
        print(f"  {icon} score={score:.1f} | kw={kw_icon}{len(hits)}/{len(hits)+len(miss)} | {lat}ms")
        print(f"     {fb[:70]}")
        if miss: print(f"     miss: {miss}")
        print()
        time.sleep(0.3)

    # ── Tổng kết ────────────────────────────────────────────────────────────
    print("="*70)
    print("  KẾT QUẢ ĐÁNH GIÁ")
    print("="*70)

    avg = sum(r.score for r in results) / len(results)
    by_cat: dict = {}
    for r in results: by_cat.setdefault(r.category, []).append(r.score)

    print(f"\nĐiểm TB LLM-judge : {avg:.3f}/1.0  ({avg*100:.1f}%)")
    print(f"✅ Tốt  ≥ 0.8 : {sum(1 for r in results if r.score>=0.8):3d}/{len(results)}")
    print(f"⚠️  TB  0.5–0.8: {sum(1 for r in results if 0.5<=r.score<0.8):3d}/{len(results)}")
    print(f"❌ Kém  < 0.5 : {sum(1 for r in results if r.score<0.5):3d}/{len(results)}")

    total_kw = sum(len(r.keywords_hit)+len(r.keywords_miss) for r in results)
    hit_kw   = sum(len(r.keywords_hit) for r in results)
    print(f"\nKeyword Hit Rate  : {hit_kw}/{total_kw} = {hit_kw/total_kw:.1%}")

    lats = sorted(r.latency_ms for r in results)
    print(f"Latency TB        : {sum(lats)//len(lats)}ms  (P50={lats[len(lats)//2]}ms, P90={lats[int(len(lats)*0.9)]}ms)")

    print("\nĐiểm theo danh mục:")
    for cat, sc in sorted(by_cat.items(), key=lambda x: sum(x[1])/len(x[1])):
        a = sum(sc)/len(sc)
        bar = "█"*int(a*10) + "░"*(10-int(a*10))
        icon = "✅" if a>=0.8 else "⚠️" if a>=0.5 else "❌"
        print(f"  {icon} {cat:15} {bar} {a:.2f}  (n={len(sc)})")

    worst = sorted(results, key=lambda r: r.score)[:5]
    print("\n5 câu kém nhất:")
    for r in worst:
        print(f"  [{r.id}] score={r.score:.1f} | {r.question[:55]}")
        print(f"         {r.feedback[:80]}")

    if save:
        from datetime import datetime
        out_data = {
            "timestamp": datetime.now().isoformat(),
            "n_questions": len(results),
            "avg_score": round(avg, 4),
            "keyword_rate": round(hit_kw/total_kw, 4),
            "avg_latency_ms": sum(lats)//len(lats),
            "by_category": {
                cat: round(sum(sc)/len(sc), 4)
                for cat, sc in by_cat.items()
            },
            "details": [asdict(r) for r in results]
        }
        out = ROOT / "tests" / "eval_results_v3.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Lưu: {out}")

    return avg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick",  action="store_true", help="Chạy 20 câu đầu")
    ap.add_argument("--save",   action="store_true", help="Lưu kết quả JSON")
    ap.add_argument("--cat",    default="",
        help="Chạy 1 danh mục: truong/diem_chuan/hoc_phi/to_hop/tinh_toan/mo_ta_nganh/faq/dau_truot/edge")
    args = ap.parse_args()
    score = run_evaluation(quick=args.quick, save=args.save, category=args.cat)
    print(f"\n>>> Final score: {score:.3f}/1.0")