"""
haui_ragas_dataset.py
Bộ test dataset đầy đủ cho RAGAs evaluation — Chatbot Tuyển sinh HaUI.

Xây dựng từ toàn bộ file .md và JSON trong dự án:
  - gioi_thieu_truong.md       → 10 câu (cat: truong)
  - ky_tuc_xa.md               → 8  câu (cat: ktx)
  - hoc_bong.md                → 10 câu (cat: hoc_bong)
  - phuong_thuc_tuyen_sinh_2025.md → 10 câu (cat: phuong_thuc)
  - chinh_sach_uu_tien.md      → 10 câu (cat: uu_tien)
  - faq_dang_ky_xet_tuyen.md   → 10 câu (cat: faq)
  - lich_tuyen_sinh_2026.md    → 8  câu (cat: lich)
  - cach_tinh_hoc_phi_2025_2026.md → 6 câu (cat: hoc_phi_cach_tinh)
  - diem_chuan JSON            → 12 câu (cat: diem_chuan)
  - muc_thu_hoc_phi JSON       → 8  câu (cat: hoc_phi)
  - to_hop / chi_tieu JSON     → 10 câu (cat: to_hop)
  - diem_quy_doi / diem_uu_tien JSON → 10 câu (cat: tinh_toan)
  - Các file ngành .md         → 14 câu (cat: mo_ta_nganh)
  - Câu hỏi đậu/trượt phức hợp → 10 câu (cat: dau_truot)
  - Câu khó / edge cases      → 10 câu (cat: edge)
  - van_bang.md / quy_mo.md    → 6  câu (cat: van_bang)
                                  ──────
Tổng:                            152 câu

Cấu trúc mỗi item:
  {
    "id"       : "CAT_NNN",
    "category" : str,
    "question" : str,           # câu hỏi thực tế của user
    "expected" : str,           # ground truth (câu trả lời chuẩn)
    "keywords" : list[str],     # từ khoá cần xuất hiện trong answer
    "intent"   : str,           # intent_type dự kiến
    "note"     : str,           # ghi chú thêm (để trống nếu không cần)
  }
"""

EVAL_DATASET = [

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: truong — Thông tin trường HaUI
    # Nguồn: gioi_thieu_truong.md
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TRUONG_001",
        "category": "truong",
        "question": "Trường Đại học Công nghiệp Hà Nội trực thuộc bộ nào?",
        "expected": "HaUI trực thuộc Bộ Công Thương.",
        "keywords": ["Bộ Công Thương"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "TRUONG_002",
        "category": "truong",
        "question": "Mã trường HaUI là gì?",
        "expected": "Mã trường của Đại học Công nghiệp Hà Nội là DCN.",
        "keywords": ["DCN"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "TRUONG_003",
        "category": "truong",
        "question": "Địa chỉ cơ sở 1 của HaUI ở đâu?",
        "expected": "Cơ sở 1 đặt tại số 298, đường Cầu Diễn, phường Minh Khai, quận Bắc Từ Liêm, Hà Nội.",
        "keywords": ["298", "Cầu Diễn", "Bắc Từ Liêm"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "TRUONG_004",
        "category": "truong",
        "question": "HaUI có bao nhiêu cơ sở đào tạo?",
        "expected": "HaUI có 3 cơ sở đào tạo: cơ sở 1 và 2 ở Bắc Từ Liêm (Hà Nội), cơ sở 3 ở Phủ Lý (Hà Nam).",
        "keywords": ["3", "cơ sở", "Hà Nam"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "TRUONG_005",
        "category": "truong",
        "question": "Năm nào Trường Đại học Công nghiệp Hà Nội chính thức trở thành Đại học?",
        "expected": "Ngày 20/11/2025, HaUI chính thức được chuyển thành Đại học Công nghiệp Hà Nội theo Quyết định số 2536/QĐ-TTg, trở thành Đại học thứ 12 của cả nước.",
        "keywords": ["20/11/2025", "2536", "thứ 12"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "TRUONG_006",
        "category": "truong",
        "question": "Tỷ lệ sinh viên HaUI có việc làm sau tốt nghiệp 1 năm là bao nhiêu?",
        "expected": "Tỷ lệ sinh viên có việc làm sau tốt nghiệp 1 năm của HaUI đạt trên 95%.",
        "keywords": ["95%"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "TRUONG_007",
        "category": "truong",
        "question": "HaUI có bao nhiêu giảng viên và tỷ lệ tiến sĩ là bao nhiêu?",
        "expected": "HaUI có gần 1.100 giảng viên, trong đó tỷ lệ có trình độ Phó Giáo sư, Tiến sĩ đạt trên 41%.",
        "keywords": ["1.100", "41%"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "TRUONG_008",
        "category": "truong",
        "question": "HaUI có bao nhiêu chương trình đào tạo đạt chuẩn ABET?",
        "expected": "HaUI có 5 chương trình đào tạo đạt chuẩn kiểm định chất lượng ABET của Hoa Kỳ.",
        "keywords": ["5", "ABET"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "TRUONG_009",
        "category": "truong",
        "question": "HaUI có những trường và khoa nào thuộc cơ cấu tổ chức?",
        "expected": "HaUI có 5 Trường (Ngoại ngữ - Du lịch, Kinh tế, Cơ khí - Ô tô, CNTT & Truyền thông, Điện - Điện tử) và 4 Khoa (Công nghệ Hóa, Công nghệ May & Thiết kế thời trang, Lý luận Chính trị & Pháp luật, Khoa học cơ bản).",
        "keywords": ["5 Trường", "4 Khoa", "CNTT", "Kinh tế"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "TRUONG_010",
        "category": "truong",
        "question": "Website đăng ký xét tuyển của HaUI là gì? Lệ phí bao nhiêu?",
        "expected": "Website đăng ký xét tuyển là xettuyen.haui.edu.vn, lệ phí 50.000 đồng/hồ sơ, nộp qua mã QR ngân hàng.",
        "keywords": ["xettuyen.haui.edu.vn", "50.000"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: ktx — Ký túc xá
    # Nguồn: ky_tuc_xa.md
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "KTX_001",
        "category": "ktx",
        "question": "Giá phòng ký túc xá chất lượng cao HaUI 4 người bao nhiêu?",
        "expected": "Phòng chất lượng cao 4 người tại KTX HaUI là 600.000 VNĐ/sinh viên/tháng.",
        "keywords": ["600.000", "4 người"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "KTX_002",
        "category": "ktx",
        "question": "Phòng ký túc xá tiêu chuẩn 6 người tại cơ sở 2 HaUI giá bao nhiêu?",
        "expected": "Phòng tiêu chuẩn 6 người tại cơ sở 2 là 280.000 VNĐ/sinh viên/tháng.",
        "keywords": ["280.000", "cơ sở 2", "6 người"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "KTX_003",
        "category": "ktx",
        "question": "Ký túc xá HaUI có điều hòa không?",
        "expected": "Phòng chất lượng cao có điều hòa. Phòng tiêu chuẩn không có điều hòa.",
        "keywords": ["chất lượng cao", "điều hòa", "tiêu chuẩn"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "KTX_004",
        "category": "ktx",
        "question": "Tiền điện nước ký túc xá HaUI tính như thế nào?",
        "expected": "Tiền điện và nước tính theo chỉ số đồng hồ thực tế hàng tháng, theo giá của nhà nước.",
        "keywords": ["chỉ số đồng hồ", "nhà nước"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "KTX_005",
        "category": "ktx",
        "question": "Ký túc xá HaUI có những tiện ích gì?",
        "expected": "Khu KTX HaUI có nhà ăn, nhà xe, siêu thị mini, quán cafe và sân thể thao.",
        "keywords": ["nhà ăn", "siêu thị mini", "sân thể thao"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "KTX_006",
        "category": "ktx",
        "question": "Giá ký túc xá rẻ nhất tại HaUI là bao nhiêu?",
        "expected": "Giá rẻ nhất là 280.000 VNĐ/sinh viên/tháng (phòng tiêu chuẩn 6 người tại cơ sở 2).",
        "keywords": ["280.000"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "KTX_007",
        "category": "ktx",
        "question": "Phòng ký túc xá HaUI có nhà vệ sinh riêng không?",
        "expected": "Tất cả các phòng KTX HaUI đều khép kín và có nhà vệ sinh riêng.",
        "keywords": ["khép kín", "nhà vệ sinh"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "KTX_008",
        "category": "ktx",
        "question": "Phòng 3 người chất lượng cao ký túc xá HaUI giá bao nhiêu?",
        "expected": "Phòng chất lượng cao 3 người là 800.000 VNĐ/sinh viên/tháng.",
        "keywords": ["800.000", "3 người"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: hoc_bong — Học bổng
    # Nguồn: hoc_bong.md
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "HOC_BONG_001",
        "category": "hoc_bong",
        "question": "HaUI có những loại học bổng nào?",
        "expected": "HaUI có 4 loại học bổng: Học bổng HaUI (đầu vào), Học bổng Khuyến khích học tập (KKHT) theo học kỳ, Học bổng Nguyễn Thanh Bình (hoàn cảnh khó khăn) và Học bổng tài trợ từ doanh nghiệp.",
        "keywords": ["HaUI", "KKHT", "Nguyễn Thanh Bình", "tài trợ"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "HOC_BONG_002",
        "category": "hoc_bong",
        "question": "Điều kiện để nhận học bổng toàn khóa HaUI là gì?",
        "expected": "Học bổng toàn khóa (100% học phí toàn khóa) dành cho sinh viên đoạt giải Nhất/Nhì/Ba HSG quốc gia/quốc tế, thi tay nghề ASEAN/quốc tế, hoặc là thủ khoa của nhóm tổ hợp/phương thức xét tuyển.",
        "keywords": ["HSG quốc gia", "thủ khoa", "100%"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "HOC_BONG_003",
        "category": "hoc_bong",
        "question": "Điều kiện duy trì học bổng HaUI sau mỗi học kỳ là gì?",
        "expected": "Sinh viên cần đạt điểm TBC học kỳ từ 2.5 trở lên, rèn luyện loại Tốt trở lên và đăng ký ít nhất 15 tín chỉ trong kỳ.",
        "keywords": ["2.5", "Tốt", "15 tín chỉ"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "HOC_BONG_004",
        "category": "hoc_bong",
        "question": "Học bổng KKHT loại Xuất sắc cần điểm TBC bao nhiêu?",
        "expected": "Học bổng KKHT loại Xuất sắc yêu cầu điểm TBC học kỳ từ 3.60 đến 4.0 (đào tạo tín chỉ) hoặc từ 9.0 đến 10.0 (niên chế), kết hợp rèn luyện Xuất sắc hoặc Tốt.",
        "keywords": ["3.60", "Xuất sắc"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "HOC_BONG_005",
        "category": "hoc_bong",
        "question": "Học bổng Nguyễn Thanh Bình dành cho đối tượng nào?",
        "expected": "Học bổng Nguyễn Thanh Bình dành cho sinh viên có hoàn cảnh khó khăn như mắc bệnh hiểm nghèo, khuyết tật, mồ côi, cha/mẹ mắc bệnh hiểm nghèo, thuộc hộ nghèo/cận nghèo, hoặc có thành tích nghiên cứu khoa học xuất sắc.",
        "keywords": ["hoàn cảnh khó khăn", "hộ nghèo", "khuyết tật"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "HOC_BONG_006",
        "category": "hoc_bong",
        "question": "Sinh viên vừa làm vừa học có được xét học bổng HaUI không?",
        "expected": "Không. Học bổng HaUI và KKHT không áp dụng cho sinh viên vừa làm vừa học, đào tạo từ xa và học chương trình thứ 2.",
        "keywords": ["không", "vừa làm vừa học", "từ xa"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "HOC_BONG_007",
        "category": "hoc_bong",
        "question": "Học bổng 5 triệu đồng của HaUI dành cho sinh viên như thế nào?",
        "expected": "Học bổng 5 triệu đồng dành cho sinh viên không thuộc diện học bổng toàn khóa và năm thứ nhất, xét điểm từ cao xuống thấp theo chỉ tiêu phân bổ từng phương thức.",
        "keywords": ["5 triệu", "không thuộc diện"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "HOC_BONG_008",
        "category": "hoc_bong",
        "question": "Sinh viên năm 4 muốn học thêm thạc sĩ có học bổng hỗ trợ không?",
        "expected": "Có. Sinh viên năm 4 đăng ký học trước các học phần thạc sĩ được hỗ trợ học bổng 30% học phí các học phần đó (tối đa 15 tín chỉ), với điều kiện điểm TBC tích lũy đến hết học kỳ 6 đạt từ 2.5 trở lên.",
        "keywords": ["30%", "15 tín chỉ", "năm 4"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "HOC_BONG_009",
        "category": "hoc_bong",
        "question": "Học bổng KKHT có được cộng dồn với học bổng HaUI không?",
        "expected": "Không. Sinh viên đã nhận học bổng HaUI trong cùng học kỳ sẽ không được xét học bổng KKHT.",
        "keywords": ["không", "cùng học kỳ"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "HOC_BONG_010",
        "category": "hoc_bong",
        "question": "Mức học bổng Nguyễn Thanh Bình cho sinh viên mồ côi là bao nhiêu phần trăm?",
        "expected": "Sinh viên mồ côi bố hoặc mẹ thuộc đối tượng 2.1.3, được nhận mức học bổng 120% mức học bổng cơ bản.",
        "keywords": ["120%", "mồ côi"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: phuong_thuc — Phương thức tuyển sinh
    # Nguồn: phuong_thuc_tuyen_sinh_2025.md
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "PT_001",
        "category": "phuong_thuc",
        "question": "HaUI có mấy phương thức xét tuyển năm 2025?",
        "expected": "HaUI có 5 phương thức xét tuyển năm 2025: PT1 (xét thẳng), PT2 (HSG/chứng chỉ quốc tế), PT3 (thi THPT), PT4 (ĐGNL ĐHQG HN), PT5 (ĐGTD ĐHBK HN).",
        "keywords": ["5", "PT1", "PT2", "PT3", "PT4", "PT5"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "PT_002",
        "category": "phuong_thuc",
        "question": "Phương thức 2 của HaUI yêu cầu IELTS tối thiểu bao nhiêu?",
        "expected": "PT2 yêu cầu IELTS Academic tối thiểu 5.5, kết hợp điểm học bạ trung bình từng môn trong tổ hợp xét tuyển đạt từ 7.0 trở lên.",
        "keywords": ["5.5", "7.0", "học bạ"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "PT_003",
        "category": "phuong_thuc",
        "question": "Công thức tính điểm xét tuyển theo PT2 là gì?",
        "expected": "Điểm xét tuyển PT2 = ĐKQHT × 2 + ĐQĐCC + Điểm ưu tiên, trong đó ĐKQHT là điểm quy đổi từ học bạ 3 môn tổ hợp (thang 10), ĐQĐCC là điểm quy đổi chứng chỉ/giải HSG.",
        "keywords": ["ĐKQHT × 2", "ĐQĐCC", "ưu tiên"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "PT_004",
        "category": "phuong_thuc",
        "question": "Giải HSG cấp tỉnh loại Nhì theo PT2 được quy đổi mấy điểm?",
        "expected": "Giải HSG cấp tỉnh loại Nhì được quy đổi thành 9.50 điểm (thang 10) theo bảng quy đổi của PT2.",
        "keywords": ["9.50", "Nhì"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "PT_005",
        "category": "phuong_thuc",
        "question": "Ai được xét tuyển thẳng vào HaUI theo PT1?",
        "expected": "PT1 dành cho: Anh hùng lao động/LLVT/Chiến sĩ thi đua toàn quốc; thí sinh đoạt giải Nhất/Nhì/Ba HSG quốc gia/quốc tế hoặc KHKT quốc gia; thí sinh đoạt giải tay nghề ASEAN/quốc tế; học sinh hoàn thành dự bị đại học (TB ≥ 8.0 tổ hợp xét tuyển).",
        "keywords": ["HSG quốc gia", "tay nghề ASEAN", "dự bị đại học"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "PT_006",
        "category": "phuong_thuc",
        "question": "Thí sinh tốt nghiệp THPT trước năm 2025 có dùng được PT4, PT5 không?",
        "expected": "Không. Thí sinh tốt nghiệp THPT trước năm 2025 không được đăng ký PT2, PT4, PT5. Muốn xét tuyển phải thi cùng học sinh lớp 12 năm 2025.",
        "keywords": ["không", "trước năm 2025"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "PT_007",
        "category": "phuong_thuc",
        "question": "Năm 2025 HaUI tuyển sinh bao nhiêu chỉ tiêu và mấy mã ngành?",
        "expected": "Năm 2025, HaUI tuyển sinh 62 mã ngành/chương trình đào tạo với 7.990 chỉ tiêu.",
        "keywords": ["62", "7.990"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "PT_008",
        "category": "phuong_thuc",
        "question": "TOPIK cấp mấy thì đủ điều kiện đăng ký PT2 HaUI?",
        "expected": "Cần TOPIK cấp 3 trở lên để đăng ký PT2 (điểm quy đổi 9.00). Riêng ngành Ngôn ngữ Trung Quốc LK 2+2 cần TOPIK từ cấp 4.",
        "keywords": ["cấp 3", "TOPIK", "9.00"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "PT_009",
        "category": "phuong_thuc",
        "question": "Thời gian đăng ký PT2, PT4, PT5 năm 2025 là khi nào?",
        "expected": "Đăng ký PT2, PT4, PT5 từ 15/5/2025 đến 05/7/2025 tại xettuyen.haui.edu.vn. Thời gian đăng ký nguyện vọng: 16/7/2025 đến 17h00 ngày 28/7/2025.",
        "keywords": ["15/5/2025", "05/7/2025", "28/7/2025"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "PT_010",
        "category": "phuong_thuc",
        "question": "Có thể dùng IELTS để thay thế điểm thi Tiếng Anh THPT trong PT3 không?",
        "expected": "Không. Nhà trường không quy đổi chứng chỉ ngoại ngữ thay cho môn thi Tiếng Anh trong PT3. Muốn xét tổ hợp có Tiếng Anh theo PT3 bắt buộc phải thi môn này.",
        "keywords": ["không", "PT3", "bắt buộc"],
        "intent": "RAG_FAQ",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: uu_tien — Chính sách ưu tiên
    # Nguồn: chinh_sach_uu_tien.md, diem_uu_tien.json
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "UT_001",
        "category": "uu_tien",
        "question": "Khu vực 1 được cộng mấy điểm ưu tiên?",
        "expected": "Thí sinh thuộc khu vực 1 (KV1) được cộng 0.75 điểm ưu tiên (nếu tổng điểm dưới 22.5 thì cộng thẳng).",
        "keywords": ["0.75", "KV1"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "UT_002",
        "category": "uu_tien",
        "question": "KV2-NT được cộng bao nhiêu điểm ưu tiên?",
        "expected": "Khu vực 2 nông thôn (KV2-NT) được cộng 0.50 điểm ưu tiên.",
        "keywords": ["0.50", "KV2-NT"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "UT_003",
        "category": "uu_tien",
        "question": "Đối tượng ưu tiên nhóm UT1 được cộng mấy điểm?",
        "expected": "Nhóm UT1 (đối tượng 01, 02, 03) được cộng 2.00 điểm ưu tiên.",
        "keywords": ["2.00", "UT1"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "UT_004",
        "category": "uu_tien",
        "question": "Công thức tính điểm ưu tiên khi tổng điểm đạt từ 22.5 trở lên?",
        "expected": "Khi tổng điểm ≥ 22.5, áp dụng công thức: Điểm ưu tiên = [(30 – Tổng điểm) / 7.5] × Mức điểm ưu tiên. Kết quả làm tròn đến hàng phần trăm.",
        "keywords": ["22.5", "7.5", "mức điểm ưu tiên"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "UT_005",
        "category": "uu_tien",
        "question": "Thí sinh đạt 25 điểm thuộc KV1 thì điểm ưu tiên thực tế là bao nhiêu?",
        "expected": "Điểm ưu tiên thực tế = [(30 - 25) / 7.5] × 0.75 = 0.50 điểm. Điểm xét tuyển = 25 + 0.50 = 25.50 điểm.",
        "keywords": ["0.50", "25.50"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "UT_006",
        "category": "uu_tien",
        "question": "Khu vực 3 được cộng mấy điểm ưu tiên?",
        "expected": "Khu vực 3 (KV3 — các phường của thành phố trực thuộc Trung ương) không được cộng điểm ưu tiên khu vực (0 điểm).",
        "keywords": ["0", "KV3", "không"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "UT_007",
        "category": "uu_tien",
        "question": "Con liệt sĩ thuộc đối tượng ưu tiên mấy và được cộng bao nhiêu điểm?",
        "expected": "Con liệt sĩ thuộc đối tượng 03, nhóm UT1, được cộng 2.00 điểm ưu tiên.",
        "keywords": ["đối tượng 03", "UT1", "2.00"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "UT_008",
        "category": "uu_tien",
        "question": "Thí sinh đạt 30 điểm thi THPT có được cộng điểm ưu tiên không?",
        "expected": "Khi điểm = 30 (tối đa), công thức cho ra [(30-30)/7.5] × mức = 0 điểm. Tức là điểm ưu tiên thực tế bằng 0 dù thí sinh thuộc khu vực hay đối tượng ưu tiên.",
        "keywords": ["0", "bằng 0", "30 điểm"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "UT_009",
        "category": "uu_tien",
        "question": "Khu vực tuyển sinh xác định dựa vào tiêu chí nào?",
        "expected": "Khu vực xác định theo trường THPT thí sinh học lâu nhất. Nếu thời gian bằng nhau, lấy theo trường học sau cùng. Thí sinh được hưởng ưu tiên trong năm tốt nghiệp THPT và một năm kế tiếp.",
        "keywords": ["lâu nhất", "sau cùng", "năm kế tiếp"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "UT_010",
        "category": "uu_tien",
        "question": "Người khuyết tật nặng thuộc đối tượng ưu tiên nào?",
        "expected": "Người khuyết tật nặng có giấy xác nhận của cơ quan thẩm quyền thuộc đối tượng 06, nhóm UT2, được cộng 1.00 điểm ưu tiên.",
        "keywords": ["đối tượng 06", "UT2", "1.00"],
        "intent": "RAG_FAQ",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: faq — Hỏi đáp đăng ký xét tuyển
    # Nguồn: faq_dang_ky_xet_tuyen.md, huong_dan_dang_ky_du_tuyen_2025.md
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "FAQ_001",
        "category": "faq",
        "question": "Sau khi đăng ký dự tuyển trên hệ thống HaUI có cần nộp hồ sơ bản cứng không?",
        "expected": "Không cần. Thí sinh chỉ nộp hồ sơ bản cứng khi trúng tuyển và nhập học theo Giấy báo nhập học.",
        "keywords": ["không", "trúng tuyển", "nhập học"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "FAQ_002",
        "category": "faq",
        "question": "Muốn sửa nguyện vọng sau khi đã đăng ký và khóa hồ sơ thì làm thế nào?",
        "expected": "Hệ thống tự động khóa sau khi đăng ký. Muốn thay đổi cần liên hệ trường qua điện thoại 0834560255 hoặc 0383371290. Khi sửa phải xóa hết nguyện vọng trước rồi chỉnh sửa.",
        "keywords": ["liên hệ", "0834560255", "xóa hết nguyện vọng"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "FAQ_003",
        "category": "faq",
        "question": "Lệ phí đăng ký xét tuyển HaUI là bao nhiêu và nộp bằng cách nào?",
        "expected": "Lệ phí 50.000 đồng/hồ sơ, nộp qua quét mã QR bằng ứng dụng ngân hàng.",
        "keywords": ["50.000", "QR", "ngân hàng"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "FAQ_004",
        "category": "faq",
        "question": "Có thể đăng ký nhiều phương thức xét tuyển cùng lúc không?",
        "expected": "Được. Nếu đủ điều kiện nhiều phương thức, thí sinh có thể đăng ký nhiều phương thức cùng lúc và cập nhật điểm, ảnh minh chứng tương ứng.",
        "keywords": ["được", "nhiều phương thức"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "FAQ_005",
        "category": "faq",
        "question": "Hồ sơ đăng ký dự tuyển trạng thái 'Chờ xử lý' có nghĩa là gì?",
        "expected": "Trạng thái 'Chờ xử lý' nghĩa là đăng ký và nộp lệ phí đã thành công. Hồ sơ đang chờ nhà trường duyệt để đưa vào hệ thống tính điểm xét tuyển.",
        "keywords": ["thành công", "chờ duyệt"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "FAQ_006",
        "category": "faq",
        "question": "Đăng ký tài khoản xét tuyển HaUI bị báo 'Đăng nhập không thành công' phải làm gì?",
        "expected": "Kiểm tra lại số CCCD và mật khẩu. Thử copy-paste mật khẩu từ email thay vì gõ tay. Nếu vẫn lỗi, bấm 'Quên mật khẩu' hoặc gọi hotline 0834560255.",
        "keywords": ["CCCD", "copy-paste", "Quên mật khẩu"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "FAQ_007",
        "category": "faq",
        "question": "Hồ sơ đăng ký dự tuyển HaUI cần chuẩn bị những gì?",
        "expected": "Cần chuẩn bị: ảnh chân dung 3×4, ảnh 2 mặt CCCD, ảnh học bạ lớp 10/11/12 có dấu đỏ, ảnh chứng chỉ quốc tế (nếu có), ảnh chứng nhận giải HSG (nếu có), ảnh kết quả ĐGNL/ĐGTD (nếu có).",
        "keywords": ["ảnh CCCD", "học bạ", "chứng chỉ"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "FAQ_008",
        "category": "faq",
        "question": "Kết quả thi ĐGTD nộp muộn sau 05/7 có được chấp nhận không?",
        "expected": "Không. Nhà trường chỉ nhận kết quả thi ĐGTD (PT5) trước ngày 05/7/2025. Kết quả nộp sau ngày này sẽ không được chấp nhận.",
        "keywords": ["không", "05/7/2025", "không được chấp nhận"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "FAQ_009",
        "category": "faq",
        "question": "Học bạ Tiếng Pháp có dùng đăng ký PT2 ngành Ngôn ngữ được không?",
        "expected": "Không được. Điểm Tiếng Pháp không nằm trong tổ hợp xét tuyển của HaUI nên phần mềm sẽ không hiển thị các ngành Ngôn ngữ để đăng ký.",
        "keywords": ["không", "Tiếng Pháp", "không nằm trong tổ hợp"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "FAQ_010",
        "category": "faq",
        "question": "Thí sinh đăng ký PT5, nếu thi lại ĐGTD điểm cao hơn có sửa được không?",
        "expected": "Được, nhưng phải liên hệ trường trước ngày 05/7/2025 và upload lại ảnh kết quả đợt thi mới.",
        "keywords": ["được", "05/7/2025", "upload lại"],
        "intent": "RAG_FAQ",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: lich — Lịch tuyển sinh 2026
    # Nguồn: lich_tuyen_sinh_2026.md
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "LICH_001",
        "category": "lich",
        "question": "Thời gian đăng ký dự tuyển PT2, PT4, PT5 năm 2026 là khi nào?",
        "expected": "Năm 2026, đăng ký PT2, PT4, PT5 từ 15/5/2026 đến 20/6/2026 tại xettuyen.haui.edu.vn.",
        "keywords": ["15/5/2026", "20/6/2026"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "LICH_002",
        "category": "lich",
        "question": "Khi nào HaUI thông báo trúng tuyển đợt 1 năm 2026?",
        "expected": "Thông báo trúng tuyển đợt 1 trước 17h00 ngày 13/8/2026.",
        "keywords": ["13/8/2026"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "LICH_003",
        "category": "lich",
        "question": "Học kỳ I năm 2026 của HaUI bắt đầu ngày nào?",
        "expected": "Học kỳ I năm học 2026 bắt đầu từ ngày 07/9/2026.",
        "keywords": ["07/9/2026"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "LICH_004",
        "category": "lich",
        "question": "Chương trình Kỹ sư (bậc 7) năm 2026 đăng ký từ khi nào?",
        "expected": "Chương trình Kỹ sư (bậc 7) năm 2026 đăng ký từ 15/6/2026 đến 15/7/2026.",
        "keywords": ["15/6/2026", "15/7/2026", "Kỹ sư"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "LICH_005",
        "category": "lich",
        "question": "Đại học từ xa HaUI năm 2026 có bao nhiêu đợt tuyển sinh?",
        "expected": "Đại học từ xa năm 2026 có 8 đợt tuyển sinh, khai giảng từ tháng 3/2026 đến tháng 1/2027.",
        "keywords": ["8 đợt", "từ xa"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "LICH_006",
        "category": "lich",
        "question": "Thời gian xác nhận nhập học trực tuyến đợt 1 năm 2026 là khi nào?",
        "expected": "Thí sinh xác nhận nhập học trực tuyến đợt 1 trước 17h00 ngày 21/8/2026.",
        "keywords": ["21/8/2026"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "LICH_007",
        "category": "lich",
        "question": "Nộp hồ sơ xét tuyển thẳng PT1 năm 2026 hạn cuối là ngày nào?",
        "expected": "Hạn nộp hồ sơ xét tuyển thẳng PT1 năm 2026 là trước 17h00 ngày 20/6/2026.",
        "keywords": ["20/6/2026", "PT1"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "LICH_008",
        "category": "lich",
        "question": "Nhập học trực tuyến năm 2026 diễn ra trong thời gian nào?",
        "expected": "Nhập học trực tuyến trên hệ thống HaUI từ 15/8/2026 đến 25/8/2026.",
        "keywords": ["15/8/2026", "25/8/2026"],
        "intent": "RAG_FAQ",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: hoc_phi_cach_tinh — Cách tính học phí
    # Nguồn: cach_tinh_hoc_phi_2025_2026.md
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "HP_CT_001",
        "category": "hoc_phi_cach_tinh",
        "question": "Công thức tính học phí học phần tại HaUI là gì?",
        "expected": "Học phí = N_TCHP × H_LHP × ĐG, trong đó N_TCHP là số tín chỉ học phí, H_LHP là hệ số lớp học phần (thường = 1.0), ĐG là đơn giá tín chỉ do Hiệu trưởng quyết định hàng năm.",
        "keywords": ["N_TCHP", "H_LHP", "ĐG"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "HP_CT_002",
        "category": "hoc_phi_cach_tinh",
        "question": "Hệ số tín chỉ cho học phần lý thuyết của K20 là bao nhiêu?",
        "expected": "Đối với K20 trở lên, học phần lý thuyết/tiểu luận/thực hành trong nhóm Lý luận, Ngoại ngữ... có hệ số tín chỉ 1.5. Học phần thực hành chuyên sâu có hệ số 2.5.",
        "keywords": ["1.5", "2.5", "K20"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "HP_CT_003",
        "category": "hoc_phi_cach_tinh",
        "question": "Học phần Giáo dục thể chất tính hệ số tín chỉ bằng bao nhiêu?",
        "expected": "Học phần Giáo dục thể chất và Quốc phòng an ninh tính hệ số tín chỉ = 1.0 (không nhân thêm hệ số).",
        "keywords": ["1.0", "Giáo dục thể chất", "Quốc phòng"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "HP_CT_004",
        "category": "hoc_phi_cach_tinh",
        "question": "H_LHP bằng bao nhiêu với lớp học mở bình thường theo kế hoạch đào tạo?",
        "expected": "Lớp mở bình thường theo kế hoạch đào tạo có H_LHP = 1.0.",
        "keywords": ["1.0", "bình thường"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "HP_CT_005",
        "category": "hoc_phi_cach_tinh",
        "question": "Đơn giá ĐG trong công thức học phí HaUI do ai quyết định?",
        "expected": "Đơn giá ĐG do Hiệu trưởng quyết định hàng năm và có thể thay đổi theo từng năm học.",
        "keywords": ["Hiệu trưởng", "hàng năm"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "HP_CT_006",
        "category": "hoc_phi_cach_tinh",
        "question": "Học phần Ngoại ngữ của K19 tính N_TCHP thế nào?",
        "expected": "Với K19 trở về trước, học phần Ngoại ngữ tính N_TCHP = Số tín chỉ × 1.5.",
        "keywords": ["1.5", "Ngoại ngữ", "K19"],
        "intent": "RAG_FAQ",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: diem_chuan — Điểm chuẩn
    # Nguồn: diem_chuan_2023_2024_2025.json
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "DC_001",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Công nghệ thông tin năm 2024 thi THPT là bao nhiêu?",
        "expected": "Ngành Công nghệ thông tin năm 2024, phương thức PT3 (thi THPT), điểm chuẩn là 25.22 điểm.",
        "keywords": ["25.22", "CNTT", "2024"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_002",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Kỹ thuật phần mềm năm 2025 là bao nhiêu?",
        "expected": "Năm 2025, ngành Kỹ thuật phần mềm có điểm chuẩn chung 21.75 điểm (áp dụng cho PT2, PT3, PT5).",
        "keywords": ["21.75", "Kỹ thuật phần mềm", "2025"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_003",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Logistics năm 2024 thi THPT là bao nhiêu?",
        "expected": "Ngành Logistics và quản lý chuỗi cung ứng năm 2024, PT3 (thi THPT), điểm chuẩn là 25.89 điểm.",
        "keywords": ["25.89", "Logistics", "2024"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_004",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Điều khiển và tự động hóa năm 2025 là bao nhiêu?",
        "expected": "Năm 2025, ngành Công nghệ kỹ thuật điều khiển và tự động hóa có điểm chuẩn chung 26.27 điểm.",
        "keywords": ["26.27", "tự động hóa", "2025"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_005",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Kế toán năm 2023 thi THPT là bao nhiêu?",
        "expected": "Ngành Kế toán năm 2023, PT3 (thi THPT), điểm chuẩn là 23.80 điểm.",
        "keywords": ["23.80", "23.8", "Kế toán", "2023"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_006",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Robot và Trí tuệ nhân tạo năm 2025 là bao nhiêu?",
        "expected": "Năm 2025, ngành Robot và trí tuệ nhân tạo có điểm chuẩn chung 24.30 điểm.",
        "keywords": ["24.30", "24.3", "Robot", "2025"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_007",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Marketing thi THPT các năm 2023, 2024, 2025 thế nào?",
        "expected": "Ngành Marketing: năm 2023 là 25.24, năm 2024 là 25.33 (đều PT3). Năm 2025 điểm chung 22.50 (PT2, PT3, PT4). Xu hướng giảm từ 2024 sang 2025 theo thang điểm chung.",
        "keywords": ["25.24", "25.33", "22.50", "Marketing"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_008",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Ngôn ngữ Trung Quốc năm 2024 xét học bạ là bao nhiêu?",
        "expected": "Ngành Ngôn ngữ Trung Quốc năm 2024, PT4 (xét học bạ/ĐGNL), điểm chuẩn là 27.62 điểm.",
        "keywords": ["27.62", "Ngôn ngữ Trung", "2024"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_009",
        "category": "diem_chuan",
        "question": "Ngành nào có điểm chuẩn 2025 cao nhất trong nhóm Cơ khí - Ô tô?",
        "expected": "Trong nhóm Cơ khí - Ô tô năm 2025, ngành Công nghệ kỹ thuật điều khiển và tự động hóa có điểm chuẩn chung cao nhất là 26.27 điểm.",
        "keywords": ["26.27", "tự động hóa", "cao nhất"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_010",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Hóa dược năm 2024 thi THPT là bao nhiêu?",
        "expected": "Ngành Hóa dược năm 2024, PT3 (thi THPT), điểm chuẩn là 21.55 điểm.",
        "keywords": ["21.55", "Hóa dược", "2024"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_011",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành Công nghệ kỹ thuật môi trường năm 2025 là bao nhiêu?",
        "expected": "Năm 2025, ngành Công nghệ kỹ thuật môi trường có điểm chuẩn chung 18.75 điểm — thấp nhất trong các ngành khối kỹ thuật.",
        "keywords": ["18.75", "môi trường", "2025"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },
    {
        "id": "DC_012",
        "category": "diem_chuan",
        "question": "Điểm chuẩn ngành An toàn thông tin năm 2025 là bao nhiêu?",
        "expected": "Ngành An toàn thông tin năm 2025 có điểm chuẩn chung 23.43 điểm (áp dụng PT2, PT3, PT5).",
        "keywords": ["23.43", "An toàn thông tin", "2025"],
        "intent": "JSON_DIEM_CHUAN",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: hoc_phi — Học phí theo chương trình
    # Nguồn: muc_thu_hoc_phi.json
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "HP_001",
        "category": "hoc_phi",
        "question": "Học phí K20 đại trà HaUI năm 2025-2026 là bao nhiêu?",
        "expected": "Học phí K20 chương trình đại trà năm 2025-2026 là 700.000 đồng/tín chỉ.",
        "keywords": ["700.000", "K20", "đại trà"],
        "intent": "JSON_HOC_PHI",
        "note": "",
    },
    {
        "id": "HP_002",
        "category": "hoc_phi",
        "question": "Học phí K20 chương trình đào tạo bằng tiếng Anh là bao nhiêu?",
        "expected": "Học phí K20 chương trình đào tạo bằng tiếng Anh là 1.000.000 đồng/tín chỉ.",
        "keywords": ["1.000.000", "tiếng Anh", "K20"],
        "intent": "JSON_HOC_PHI",
        "note": "",
    },
    {
        "id": "HP_003",
        "category": "hoc_phi",
        "question": "Học phí K19 HaUI bao nhiêu một tín chỉ?",
        "expected": "Học phí K19 là 550.000 đồng/tín chỉ.",
        "keywords": ["550.000", "K19"],
        "intent": "JSON_HOC_PHI",
        "note": "",
    },
    {
        "id": "HP_004",
        "category": "hoc_phi",
        "question": "Học phí K18 hoặc cũ hơn của HaUI là bao nhiêu?",
        "expected": "K18 trở về trước học phí là 495.000 đồng/tín chỉ.",
        "keywords": ["495.000", "K18"],
        "intent": "JSON_HOC_PHI",
        "note": "",
    },
    {
        "id": "HP_005",
        "category": "hoc_phi",
        "question": "Học phí Thạc sĩ HaUI là bao nhiêu?",
        "expected": "Học phí Thạc sĩ là 900.000 đồng/tín chỉ.",
        "keywords": ["900.000", "Thạc sĩ"],
        "intent": "JSON_HOC_PHI",
        "note": "",
    },
    {
        "id": "HP_006",
        "category": "hoc_phi",
        "question": "Học phí Tiến sĩ HaUI là bao nhiêu một năm?",
        "expected": "Học phí Tiến sĩ là 35.000.000 đồng/năm.",
        "keywords": ["35.000.000", "Tiến sĩ"],
        "intent": "JSON_HOC_PHI",
        "note": "",
    },
    {
        "id": "HP_007",
        "category": "hoc_phi",
        "question": "Học phí đào tạo từ xa HaUI bao nhiêu một tín chỉ?",
        "expected": "Học phí đào tạo từ xa là 495.000 đồng/tín chỉ.",
        "keywords": ["495.000", "từ xa"],
        "intent": "JSON_HOC_PHI",
        "note": "",
    },
    {
        "id": "HP_008",
        "category": "hoc_phi",
        "question": "Học phí Kỹ sư K3 đại trà HaUI là bao nhiêu?",
        "expected": "Học phí Kỹ sư K3 chương trình đại trà là 700.000 đồng/tín chỉ.",
        "keywords": ["700.000", "Kỹ sư K3", "đại trà"],
        "intent": "JSON_HOC_PHI",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: to_hop — Tổ hợp môn và chỉ tiêu
    # Nguồn: chi_tieu_to_hop_2025.json, to_hop_mon_thi.json
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TH_001",
        "category": "to_hop",
        "question": "Ngành CNTT HaUI xét tuyển bằng tổ hợp nào?",
        "expected": "Ngành Công nghệ thông tin xét tuyển bằng 4 tổ hợp: A00 (Toán-Lý-Hóa), A01 (Toán-Lý-Anh), X06 (Toán-Tin-Công nghệ) và X07 (Toán-Lý-Công nghệ).",
        "keywords": ["A00", "A01", "X06", "X07"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "TH_002",
        "category": "to_hop",
        "question": "Tổ hợp D01 gồm những môn gì?",
        "expected": "Tổ hợp D01 gồm Toán, Ngữ văn và Tiếng Anh.",
        "keywords": ["Toán", "Ngữ văn", "Tiếng Anh"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "TH_003",
        "category": "to_hop",
        "question": "Ngành Kế toán HaUI có bao nhiêu chỉ tiêu năm 2025?",
        "expected": "Ngành Kế toán có 600 chỉ tiêu năm 2025.",
        "keywords": ["600", "Kế toán"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "TH_004",
        "category": "to_hop",
        "question": "Tổ hợp A00 có thể xét tuyển được những ngành nào thuộc nhóm CNTT?",
        "expected": "Tổ hợp A00 có thể xét các ngành CNTT như: Khoa học máy tính, Mạng máy tính, Kỹ thuật phần mềm, Hệ thống thông tin, Công nghệ kỹ thuật máy tính, Công nghệ thông tin, Công nghệ đa phương tiện, An toàn thông tin.",
        "keywords": ["Kỹ thuật phần mềm", "An toàn thông tin", "CNTT"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "TH_005",
        "category": "to_hop",
        "question": "Ngành Robot và Trí tuệ nhân tạo có bao nhiêu chỉ tiêu và tổ hợp gì?",
        "expected": "Ngành Robot và trí tuệ nhân tạo có 60 chỉ tiêu, xét tuyển tổ hợp A00, A01, X06, X07 theo PT1, PT2, PT3, PT5.",
        "keywords": ["60", "Robot", "A00"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "TH_006",
        "category": "to_hop",
        "question": "Ngành Logistics HaUI xét tuyển tổ hợp nào năm 2025?",
        "expected": "Ngành Logistics và quản lý chuỗi cung ứng xét tuyển tổ hợp A01, D01 và X25 theo PT1, PT2, PT3, PT4.",
        "keywords": ["A01", "D01", "X25", "Logistics"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "TH_007",
        "category": "to_hop",
        "question": "Tổng chỉ tiêu HaUI năm 2026 là bao nhiêu?",
        "expected": "Tổng chỉ tiêu tuyển sinh HaUI năm 2026 là 9.420 sinh viên, trong đó đại học chính quy 8.300, đại học từ xa 750, liên thông 250, Kỹ sư bậc 7 là 120.",
        "keywords": ["9.420", "8.300", "2026"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "TH_008",
        "category": "to_hop",
        "question": "Tổ hợp X25 gồm những môn gì?",
        "expected": "Tổ hợp X25 gồm Toán, Tiếng Anh và Giáo dục kinh tế pháp luật.",
        "keywords": ["Toán", "Tiếng Anh", "GDKT pháp luật"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "TH_009",
        "category": "to_hop",
        "question": "Ngành Ngôn ngữ Hàn Quốc HaUI xét tổ hợp gì?",
        "expected": "Ngành Ngôn ngữ Hàn Quốc xét tuyển tổ hợp D01 (Toán-Văn-Anh) và DD2 (Toán-Văn-Tiếng Hàn).",
        "keywords": ["D01", "DD2", "Tiếng Hàn"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },
    {
        "id": "TH_010",
        "category": "to_hop",
        "question": "Tổ hợp B00 gồm những môn gì và có thể đăng ký ngành nào tại HaUI?",
        "expected": "Tổ hợp B00 gồm Toán, Hóa học và Sinh học. Tại HaUI, B00 dùng để xét ngành Công nghệ kỹ thuật hóa học, Công nghệ kỹ thuật môi trường, Công nghệ thực phẩm và Hóa dược.",
        "keywords": ["Toán", "Hóa học", "Sinh học", "Hóa dược", "Thực phẩm"],
        "intent": "JSON_CHI_TIEU_TO_HOP",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: tinh_toan — Tính điểm quy đổi và điểm ưu tiên
    # Nguồn: diem_quy_doi.json, diem_uu_tien.json
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TT_001",
        "category": "tinh_toan",
        "question": "Điểm ĐGNL (HSA) 105 quy đổi được bao nhiêu điểm thang 30?",
        "expected": "Điểm HSA 105 quy đổi được 25.00 điểm (thang 30).",
        "keywords": ["25.00", "105", "HSA"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "TT_002",
        "category": "tinh_toan",
        "question": "Điểm ĐGTD (TSA) 60 điểm tương đương bao nhiêu điểm thang 30?",
        "expected": "Điểm TSA 60 quy đổi được 25.50 điểm (thang 30).",
        "keywords": ["25.50", "60", "TSA"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "TT_003",
        "category": "tinh_toan",
        "question": "Điểm học bạ trung bình môn 9.0 quy đổi thành bao nhiêu điểm thi THPT tương đương?",
        "expected": "Điểm học bạ 9.00 đến 9.09 quy đổi thành 7.23 điểm thi THPT tương đương (thang 10).",
        "keywords": ["7.23", "9.0", "học bạ"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "TT_004",
        "category": "tinh_toan",
        "question": "Tôi đạt 90 điểm HSA, ở KV2-NT, điểm xét tuyển vào HaUI là bao nhiêu?",
        "expected": "HSA 90 điểm quy đổi được 22.70 điểm (thang 30). Vì 22.70 ≥ 22.5, áp dụng công thức giảm dần: điểm ưu tiên KV2-NT = [(30-22.70)/7.5] × 0.50 ≈ 0.49 điểm. Điểm xét tuyển ≈ 22.70 + 0.49 = 23.19 điểm.",
        "keywords": ["22.70", "22.5", "KV2-NT"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "Câu tính toán phức hợp",
    },
    {
        "id": "TT_005",
        "category": "tinh_toan",
        "question": "Điểm TSA 85 điểm được quy đổi thành bao nhiêu?",
        "expected": "Điểm TSA từ 85 trở lên quy đổi được 30.00 điểm (điểm tối đa thang 30).",
        "keywords": ["30.00", "85", "TSA"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "TT_006",
        "category": "tinh_toan",
        "question": "Em thi THPT được 22 điểm, KV1, điểm xét tuyển là bao nhiêu?",
        "expected": "22 điểm < 22.5 nên cộng thẳng điểm ưu tiên KV1 = 0.75. Điểm xét tuyển = 22 + 0.75 = 22.75 điểm.",
        "keywords": ["22.75", "22", "KV1", "cộng thẳng"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "TT_007",
        "category": "tinh_toan",
        "question": "Điểm HSA 130 trở lên quy đổi được bao nhiêu điểm?",
        "expected": "Điểm HSA từ 130 đến 150 quy đổi được 30.00 điểm (điểm tối đa).",
        "keywords": ["30.00", "130"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "TT_008",
        "category": "tinh_toan",
        "question": "Điểm học bạ 8.5 quy đổi thành bao nhiêu điểm thi THPT tương đương?",
        "expected": "Điểm học bạ 8.50 đến 8.59 quy đổi thành 6.55 điểm thi THPT tương đương (thang 10).",
        "keywords": ["6.55", "8.5", "học bạ"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },
    {
        "id": "TT_009",
        "category": "tinh_toan",
        "question": "Thí sinh đạt 25 điểm thi THPT, KV2-NT, đối tượng 06 thì điểm xét tuyển là bao nhiêu?",
        "expected": "Tổng ưu tiên KV2-NT + ĐT06 = 0.50 + 1.00 = 1.50. Vì 25 ≥ 22.5, áp dụng giảm dần: [(30-25)/7.5] × 1.50 = 1.00. Điểm xét tuyển = 25 + 1.00 = 26.00 điểm.",
        "keywords": ["26.00", "25", "KV2-NT", "đối tượng 06"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "Câu phức hợp cả KV và đối tượng",
    },
    {
        "id": "TT_010",
        "category": "tinh_toan",
        "question": "Điểm TSA 55 điểm quy đổi thành bao nhiêu điểm thang 30?",
        "expected": "Điểm TSA từ 55.00 đến 55.99 quy đổi được 23.60 điểm (thang 30).",
        "keywords": ["23.60", "55", "TSA"],
        "intent": "JSON_QUY_DOI_DIEM",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: mo_ta_nganh — Mô tả ngành, cơ hội việc làm
    # Nguồn: các file nganh_*.md
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "MN_001",
        "category": "mo_ta_nganh",
        "question": "Ngành Kỹ thuật phần mềm HaUI ra trường làm được những vị trí gì?",
        "expected": "Sinh viên KTPM có thể làm: Lập trình viên (web/mobile/desktop), Kỹ sư phần mềm, Kỹ sư kiểm thử (QA/QC), Chuyên viên DevOps, Quản lý dự án phần mềm, Chuyên gia giải pháp (Solution Architect), hoặc khởi nghiệp công nghệ.",
        "keywords": ["Lập trình viên", "DevOps", "QA", "Solution Architect"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_002",
        "category": "mo_ta_nganh",
        "question": "Ngành Robot và Trí tuệ nhân tạo HaUI học những gì?",
        "expected": "Ngành Robot và AI học: lập trình Python/C++/ROS, machine learning, deep learning, computer vision, NLP, hệ thống nhúng, cảm biến, vi xử lý, điều khiển tự động và thiết kế robot công nghiệp/dịch vụ.",
        "keywords": ["Python", "machine learning", "ROS", "robot"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_003",
        "category": "mo_ta_nganh",
        "question": "Ngành Logistics HaUI có cơ hội việc làm như thế nào?",
        "expected": "Sinh viên Logistics có thể làm tại: doanh nghiệp logistics/vận tải/giao nhận, công ty sản xuất/thương mại/XNK, hãng tàu/cảng biển/hàng không, hoặc cơ quan nhà nước. Vị trí: điều phối vận tải, quản lý kho, khai báo hải quan, hoạch định chuỗi cung ứng...",
        "keywords": ["vận tải", "kho", "hải quan", "chuỗi cung ứng"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_004",
        "category": "mo_ta_nganh",
        "question": "Ngành An toàn thông tin khác Công nghệ thông tin như thế nào?",
        "expected": "An toàn thông tin tập trung vào bảo mật hệ thống, mã hóa, kiểm thử xâm nhập (pentest), phân tích mã độc, ứng cứu sự cố. CNTT có phạm vi rộng hơn gồm phát triển phần mềm, cơ sở dữ liệu, mạng, AI... Cả hai cùng hệ CNTT nhưng định hướng chuyên sâu khác nhau.",
        "keywords": ["bảo mật", "pentest", "mã hóa"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_005",
        "category": "mo_ta_nganh",
        "question": "Ngành Công nghệ thực phẩm HaUI ra làm gì?",
        "expected": "Sinh viên Công nghệ thực phẩm có thể làm: Kỹ sư công nghệ thực phẩm tại nhà máy chế biến, chuyên viên QA/QC, chuyên viên R&D phát triển sản phẩm mới, chuyên viên an toàn thực phẩm, giảng viên/nghiên cứu viên, hoặc khởi nghiệp trong ngành thực phẩm.",
        "keywords": ["QA/QC", "R&D", "an toàn thực phẩm"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_006",
        "category": "mo_ta_nganh",
        "question": "Ngành Công nghệ kỹ thuật điều khiển và tự động hóa HaUI học phần mềm gì?",
        "expected": "Ngành Điều khiển và tự động hóa sử dụng các phần mềm: AutoCAD Electrical, TIA Portal (Siemens), CX-Programmer, LabVIEW, MATLAB/Simulink để lập trình PLC, thiết kế và mô phỏng hệ thống điều khiển.",
        "keywords": ["TIA Portal", "PLC", "MATLAB", "LabVIEW"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_007",
        "category": "mo_ta_nganh",
        "question": "Ngành Thiết kế thời trang HaUI có cơ hội việc làm gì?",
        "expected": "Sinh viên Thiết kế thời trang có thể làm: Nhà thiết kế thời trang, Chuyên viên phát triển sản phẩm, Stylist, Nhà nghiên cứu xu hướng, Pattern maker, Marketing thời trang, Giảng viên hoặc khởi nghiệp thương hiệu riêng.",
        "keywords": ["Nhà thiết kế", "Stylist", "xu hướng", "thương hiệu"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_008",
        "category": "mo_ta_nganh",
        "question": "Ngành Hóa dược HaUI học những kỹ năng gì và sử dụng thiết bị nào?",
        "expected": "Ngành Hóa dược học tổng hợp hóa hữu cơ, phân tích hóa dược, kiểm soát chất lượng thuốc, sử dụng thiết bị HPLC, GC, phổ khối MS, NMR để phân tích và phát triển dược phẩm.",
        "keywords": ["HPLC", "GC", "NMR", "tổng hợp"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_009",
        "category": "mo_ta_nganh",
        "question": "Ngành Công nghệ kỹ thuật ô tô HaUI thuộc trường nào và tuyển bao nhiêu chỉ tiêu?",
        "expected": "Ngành Công nghệ kỹ thuật ô tô thuộc Trường Cơ khí - Ô tô, có 360 chỉ tiêu năm 2025, mã ngành 7510205.",
        "keywords": ["360", "Cơ khí - Ô tô", "7510205"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_010",
        "category": "mo_ta_nganh",
        "question": "Ngành Ngôn ngữ Hàn Quốc HaUI tốt nghiệp có thể làm gì?",
        "expected": "Sinh viên Ngôn ngữ Hàn Quốc có thể làm: Biên/phiên dịch tiếng Hàn tại doanh nghiệp Hàn Quốc/khu công nghiệp, Chuyên viên nhân sự/đối ngoại, Trợ lý giám đốc người Hàn, Giảng dạy tiếng Hàn, Chuyên viên xuất nhập khẩu thị trường Hàn.",
        "keywords": ["biên dịch", "khu công nghiệp", "Hàn Quốc"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_011",
        "category": "mo_ta_nganh",
        "question": "Ngành Phân tích dữ liệu kinh doanh HaUI dùng công cụ gì?",
        "expected": "Ngành Phân tích dữ liệu kinh doanh sử dụng: Excel nâng cao, SQL, Power BI, Tableau, Python, R để thu thập, xử lý, trực quan hóa dữ liệu và xây dựng dashboard hỗ trợ ra quyết định.",
        "keywords": ["Power BI", "Tableau", "Python", "SQL"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_012",
        "category": "mo_ta_nganh",
        "question": "Ngành Năng lượng tái tạo HaUI ra làm ở những đơn vị nào?",
        "expected": "Sinh viên Năng lượng tái tạo có thể làm tại: các đơn vị thuộc EVN, công ty điện lực các tỉnh, công ty tư vấn thiết kế/thi công hệ thống điện mặt trời/điện gió, trường đại học và viện nghiên cứu.",
        "keywords": ["EVN", "điện mặt trời", "điện gió"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_013",
        "category": "mo_ta_nganh",
        "question": "Chương trình liên kết 2+2 ngành Ngôn ngữ Trung Quốc HaUI là gì?",
        "expected": "Chương trình học 2 năm tại HaUI + 2 năm tại Đại học Khoa học Kỹ thuật Quảng Tây (Trung Quốc). Tốt nghiệp nhận 2 bằng: bằng cử nhân của HaUI và bằng Hán ngữ giáo dục quốc tế của trường Quảng Tây.",
        "keywords": ["2 năm", "Quảng Tây", "2 bằng"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },
    {
        "id": "MN_014",
        "category": "mo_ta_nganh",
        "question": "Ngành Kỹ thuật sản xuất thông minh HaUI ứng dụng công nghệ gì?",
        "expected": "Ngành Kỹ thuật sản xuất thông minh ứng dụng: IoT, AI, Big Data, điện toán đám mây, VR/AR vào sản xuất; sử dụng phần mềm CAD/CAM, PLM, MES, ERP, Siemens NX để thiết kế và quản lý nhà máy thông minh.",
        "keywords": ["IoT", "AI", "CAD/CAM", "MES", "nhà máy thông minh"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: dau_truot — Kiểm tra đậu/trượt
    # Nguồn: diem_chuan JSON + logic tính toán
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "DT_001",
        "category": "dau_truot",
        "question": "Em thi THPT được 25 điểm, KV1, có đậu ngành CNTT HaUI không?",
        "expected": "Điểm xét tuyển = 25 + ưu tiên KV1 (25≥22.5 nên giảm dần) ≈ 25.50. Điểm chuẩn CNTT 2025 là 23.09, điểm chuẩn 2024 là 25.22. So với 2024 thì sát nút, so với 2025 thì cao hơn đáng kể. Khả năng đậu cao nhưng cần theo dõi điểm chuẩn 2026.",
        "keywords": ["25", "KV1", "CNTT", "23.09"],
        "intent": "JSON_DAU_TRUOT",
        "note": "",
    },
    {
        "id": "DT_002",
        "category": "dau_truot",
        "question": "Điểm TSA 75 điểm có đủ vào ngành Kỹ thuật phần mềm không?",
        "expected": "Điểm TSA 75 quy đổi được 28.50 điểm (thang 30). Điểm chuẩn KTPM 2025 là 21.75, điểm chuẩn 2024 PT5 là 16.01. Điểm xét 28.50 cao hơn điểm chuẩn, khả năng đậu rất cao.",
        "keywords": ["28.50", "75", "KTPM", "21.75"],
        "intent": "JSON_DAU_TRUOT",
        "note": "",
    },
    {
        "id": "DT_003",
        "category": "dau_truot",
        "question": "Em thi THPT 20 điểm KV2-NT có đỗ ngành Kế toán không?",
        "expected": "Điểm xét = 20 + 0.50 (KV2-NT, 20<22.5 cộng thẳng) = 20.50. Điểm chuẩn Kế toán 2025 là 20.00. Điểm xét 20.50 cao hơn điểm chuẩn 2025 nên khả năng đậu, tuy nhiên điểm chuẩn 2026 chưa công bố.",
        "keywords": ["20.50", "20.00", "Kế toán", "KV2-NT"],
        "intent": "JSON_DAU_TRUOT",
        "note": "",
    },
    {
        "id": "DT_004",
        "category": "dau_truot",
        "question": "Điểm HSA 95, KV2, có vào được ngành Điều khiển tự động hóa HaUI không?",
        "expected": "HSA 95 quy đổi 23.50 điểm. KV2 được +0.25, 23.50≥22.5 nên giảm dần: [(30-23.50)/7.5]×0.25≈0.22. Điểm xét ≈ 23.72. Điểm chuẩn Tự động hóa 2025 là 26.27, điểm xét 23.72 thấp hơn nhiều → khả năng trượt cao.",
        "keywords": ["23.50", "23.72", "26.27", "trượt"],
        "intent": "JSON_DAU_TRUOT",
        "note": "Câu tính phức hợp, kết quả trượt",
    },
    {
        "id": "DT_005",
        "category": "dau_truot",
        "question": "Em 23.5 điểm thi THPT, KV3, có nộp hồ sơ ngành Marketing không?",
        "expected": "KV3 không được cộng điểm ưu tiên. Điểm xét = 23.5. Điểm chuẩn Marketing 2025 là 22.50, 2024 là 25.33 (PT3). So với 2025 thì đậu, nhưng so với 2024 thì trượt. Điểm 2026 chưa công bố, nên vẫn có rủi ro nếu điểm chuẩn tăng.",
        "keywords": ["23.5", "22.50", "Marketing", "KV3"],
        "intent": "JSON_DAU_TRUOT",
        "note": "",
    },
    {
        "id": "DT_006",
        "category": "dau_truot",
        "question": "Điểm học bạ 3 môn Toán 8.5, Lý 8.0, Hóa 8.0 có đủ xét tuyển ngành Cơ điện tử?",
        "expected": "Điểm học bạ cần quy đổi theo bảng: 8.5→6.55, 8.0→6.27. Tổng 3 môn quy đổi ≈ 19.09, chưa tính ưu tiên. Điểm chuẩn Cơ điện tử 2025 là 25.17 (điểm chung). Điểm xét 19.09 thấp hơn nhiều → không đủ điều kiện.",
        "keywords": ["19", "25.17", "Cơ điện tử", "không đủ"],
        "intent": "JSON_DAU_TRUOT",
        "note": "",
    },
    {
        "id": "DT_007",
        "category": "dau_truot",
        "question": "Em 22 điểm thi THPT KV1 có đỗ ngành Ngôn ngữ Nhật HaUI không?",
        "expected": "Điểm xét = 22 + 0.75 (KV1, 22<22.5 cộng thẳng) = 22.75. Điểm chuẩn Ngôn ngữ Nhật 2025 là 20.00. Điểm xét 22.75 cao hơn điểm chuẩn 2025, khả năng đậu tốt.",
        "keywords": ["22.75", "20.00", "Ngôn ngữ Nhật"],
        "intent": "JSON_DAU_TRUOT",
        "note": "",
    },
    {
        "id": "DT_008",
        "category": "dau_truot",
        "question": "Điểm TSA 50 điểm có vào được ngành nào tại HaUI?",
        "expected": "TSA 50-50.99 quy đổi được 21.35 điểm (thang 30). Với điểm này có thể đăng ký các ngành có điểm chuẩn 2025 thấp hơn như Công nghệ dệt may (18.00-18.25), Môi trường (18.75), Du lịch tiếng Anh (18.00)... Nên cộng thêm điểm ưu tiên nếu có.",
        "keywords": ["21.35", "50", "18.00", "18.75"],
        "intent": "JSON_DAU_TRUOT",
        "note": "",
    },
    {
        "id": "DT_009",
        "category": "dau_truot",
        "question": "Em 26 điểm THPT KV2-NT đăng ký ngành Logistics có chắc đậu không?",
        "expected": "Điểm xét: 26 ≥ 22.5 nên giảm dần: [(30-26)/7.5]×0.50 = 0.27. Điểm xét ≈ 26.27. Điểm chuẩn Logistics 2025 là 22.76, 2024 là 25.89. So với cả 2 năm đều cao hơn. Khả năng đậu cao, nhưng điểm 2026 chưa công bố.",
        "keywords": ["26.27", "22.76", "25.89", "Logistics"],
        "intent": "JSON_DAU_TRUOT",
        "note": "",
    },
    {
        "id": "DT_010",
        "category": "dau_truot",
        "question": "Điểm HSA 80, ở KV2, đăng ký ngành Robot AI có đậu không?",
        "expected": "HSA 80 quy đổi 21.02 điểm. KV2 được +0.25, 21.02<22.5 nên cộng thẳng: 21.02 + 0.25 = 21.27. Điểm chuẩn Robot AI 2025 là 24.30. Điểm xét 21.27 < 24.30 → trượt, thiếu khoảng 3 điểm.",
        "keywords": ["21.02", "21.27", "24.30", "trượt", "Robot"],
        "intent": "JSON_DAU_TRUOT",
        "note": "",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: edge — Câu hỏi khó, viết tắt, câu phức hợp
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "EDGE_001",
        "category": "edge",
        "question": "Mã DCN là trường nào, ở đâu?",
        "expected": "DCN là mã trường của Đại học Công nghiệp Hà Nội (HaUI), đặt tại số 298 Cầu Diễn, quận Bắc Từ Liêm, Hà Nội.",
        "keywords": ["DCN", "HaUI", "Cầu Diễn"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "Hỏi bằng mã trường thay vì tên",
    },
    {
        "id": "EDGE_002",
        "category": "edge",
        "question": "Tôi muốn học ngành KTPM, điểm TSA 60, KV1 thì được bao nhiêu và có đỗ không?",
        "expected": "TSA 60 quy đổi 25.50 điểm. KV1 +0.75, 25.50≥22.5 giảm dần: [(30-25.50)/7.5]×0.75 = 0.45. Điểm xét ≈ 25.95. Điểm chuẩn KTPM 2025 là 21.75. Khả năng đậu rất cao.",
        "keywords": ["25.50", "0.45", "25.95", "21.75", "KTPM"],
        "intent": "JSON_DAU_TRUOT",
        "note": "Câu hỏi kết hợp quy đổi + ưu tiên + đậu/trượt",
    },
    {
        "id": "EDGE_003",
        "category": "edge",
        "question": "HaUI là trường nào? Học phí bao nhiêu? Điểm chuẩn CNTT năm ngoái?",
        "expected": "HaUI là Đại học Công nghiệp Hà Nội, mã DCN, trực thuộc Bộ Công Thương. Học phí K20 đại trà 700.000đ/tín chỉ. Điểm chuẩn CNTT năm 2025 (năm gần nhất) là 23.09 điểm chung.",
        "keywords": ["HaUI", "700.000", "23.09", "Bộ Công Thương"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "Câu hỏi đa chủ đề trong một câu",
    },
    {
        "id": "EDGE_004",
        "category": "edge",
        "question": "Ngành tự động hóa và ngành cơ điện tử khác nhau như thế nào? Ngành nào điểm chuẩn cao hơn?",
        "expected": "Điều khiển và tự động hóa tập trung vào lập trình PLC, SCADA, hệ thống điều khiển công nghiệp. Cơ điện tử kết hợp cơ khí + điện tử + điều khiển tích hợp, thiên về thiết kế hệ thống cơ điện tử. Điểm chuẩn 2025: Tự động hóa 26.27 > Cơ điện tử 25.17.",
        "keywords": ["26.27", "25.17", "PLC", "tích hợp"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "So sánh 2 ngành",
    },
    {
        "id": "EDGE_005",
        "category": "edge",
        "question": "PT4 và PT5 khác nhau thế nào? Nên chọn phương thức nào?",
        "expected": "PT4 dựa trên điểm ĐGNL do ĐHQG Hà Nội tổ chức (thang 150, quy đổi sang thang 30). PT5 dựa trên điểm ĐGTD do ĐHBK Hà Nội tổ chức (thang 100, quy đổi sang thang 30). Nên đăng ký cả hai nếu đủ điều kiện để tăng cơ hội.",
        "keywords": ["ĐHQG Hà Nội", "ĐHBK Hà Nội", "thang 150", "thang 100"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "EDGE_006",
        "category": "edge",
        "question": "Giải HSG cấp tỉnh môn Tin học năm lớp 11 có dùng đăng ký PT2 không?",
        "expected": "Có. Giải HSG cấp tỉnh/TP môn Tin học được chấp nhận trong PT2 (không quá 3 năm tính đến thời điểm xét tuyển), kết hợp điểm học bạ trung bình từng môn tổ hợp ≥ 7.0.",
        "keywords": ["Tin học", "3 năm", "7.0", "PT2"],
        "intent": "RAG_FAQ",
        "note": "",
    },
    {
        "id": "EDGE_007",
        "category": "edge",
        "question": "Học ngành cntt hay ktpm ở haui tốt hơn?",
        "expected": "Cả hai ngành đều tốt. CNTT có phạm vi rộng hơn (phát triển phần mềm, mạng, AI, cơ sở dữ liệu...). KTPM chuyên sâu hơn vào quy trình phát triển phần mềm (Agile, DevOps, kiểm thử). Điểm chuẩn 2025: CNTT 23.09, KTPM 21.75. Chọn theo sở thích cá nhân.",
        "keywords": ["CNTT", "KTPM", "23.09", "21.75"],
        "intent": "RAG_MO_TA_NGANH",
        "note": "Câu hỏi viết tắt không dấu",
    },
    {
        "id": "EDGE_008",
        "category": "edge",
        "question": "Năm 2026 em có thể đăng ký xét tuyển bằng điểm ĐGNL 2025 không?",
        "expected": "Điều này phụ thuộc vào quy chế tuyển sinh 2026 chưa công bố đầy đủ. Thông thường HaUI chấp nhận kết quả ĐGNL trong năm tốt nghiệp. Nên theo dõi thông báo chính thức tại haui.edu.vn.",
        "keywords": ["2026", "ĐGNL", "quy chế", "theo dõi"],
        "intent": "RAG_FAQ",
        "note": "Câu hỏi về năm chưa có đủ thông tin",
    },
    {
        "id": "EDGE_009",
        "category": "edge",
        "question": "Em ở thành phố Hà Nội thuộc khu vực nào?",
        "expected": "Nếu ở các phường của Hà Nội (thành phố trực thuộc Trung ương) thì thuộc KV3 (không được cộng điểm ưu tiên). Nếu ở xã của Hà Nội (không phải phường) thì thuộc KV2. Cần xác định theo trường THPT đã học lâu nhất.",
        "keywords": ["KV3", "phường", "KV2", "xã"],
        "intent": "RAG_FAQ",
        "note": "Câu hỏi phụ thuộc vào địa chỉ cụ thể",
    },
    {
        "id": "EDGE_010",
        "category": "edge",
        "question": "Tổng chi phí học 4 năm đại học K20 đại trà tại HaUI khoảng bao nhiêu?",
        "expected": "Học phí K20 đại trà là 700.000đ/tín chỉ. Chương trình 4 năm khoảng 130-140 tín chỉ, tổng học phí ước tính 91-98 triệu đồng (chưa tính sinh hoạt phí, tài liệu). Nếu ở KTX tiêu chuẩn thêm khoảng 3.7-5.6 triệu/năm.",
        "keywords": ["700.000", "130", "91", "K20"],
        "intent": "JSON_HOC_PHI",
        "note": "Câu hỏi ước tính chi phí tổng",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY: van_bang — Văn bằng, quy mô đào tạo
    # Nguồn: van_bang.md, quy_mo_dao_tao.md
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "VB_001",
        "category": "van_bang",
        "question": "Tốt nghiệp 4 năm đại học tại HaUI được cấp bằng gì?",
        "expected": "Sinh viên tốt nghiệp chương trình 4 năm (thời gian chuẩn) được cấp bằng Cử nhân.",
        "keywords": ["Cử nhân", "4 năm"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "VB_002",
        "category": "van_bang",
        "question": "Sinh viên HaUI có thể học 2 bằng cùng lúc không?",
        "expected": "Có. Sinh viên hoàn thành năm thứ nhất có thể đăng ký học 2 chương trình đồng thời và được cấp 2 bằng tốt nghiệp của hai chương trình đào tạo khác nhau.",
        "keywords": ["2 bằng", "năm thứ nhất"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "VB_003",
        "category": "van_bang",
        "question": "HaUI hiện có bao nhiêu sinh viên đại học chính quy?",
        "expected": "Số sinh viên đại học chính quy đang theo học tại HaUI là 25.447 người.",
        "keywords": ["25.447", "chính quy"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "VB_004",
        "category": "van_bang",
        "question": "Học chương trình Kỹ sư tại HaUI cần điều kiện gì và được cấp bằng gì?",
        "expected": "Sinh viên tốt nghiệp đại học (bằng cử nhân) có thể dự tuyển chương trình đào tạo chuyên sâu đặc thù để lấy bằng Kỹ sư (bậc 7) theo thông báo riêng của trường.",
        "keywords": ["bằng cử nhân", "Kỹ sư", "bậc 7"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "VB_005",
        "category": "van_bang",
        "question": "Tỷ lệ sinh viên chính quy HaUI có việc làm sau 12 tháng tốt nghiệp là bao nhiêu?",
        "expected": "Tỷ lệ sinh viên đại học chính quy có việc làm sau 12 tháng tốt nghiệp là 92.78%.",
        "keywords": ["92.78%", "12 tháng"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
    {
        "id": "VB_006",
        "category": "van_bang",
        "question": "HaUI có bao nhiêu sinh viên sau đại học đang theo học?",
        "expected": "HaUI có 558 sinh viên sau đại học (cao học, NCS) đang theo học.",
        "keywords": ["558", "sau đại học"],
        "intent": "RAG_TRUONG_HOC_BONG",
        "note": "",
    },
]

# ── Thống kê nhanh ────────────────────────────────────────────────────────────

def print_stats():
    from collections import Counter
    cats = Counter(d["category"] for d in EVAL_DATASET)
    intents = Counter(d["intent"] for d in EVAL_DATASET)
    print(f"\n{'═'*60}")
    print(f"  HaUI RAGAs Dataset — Tổng: {len(EVAL_DATASET)} câu")
    print(f"{'═'*60}")
    print("\nPhân bố theo danh mục:")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        bar = "█" * n
        print(f"  {cat:<25} {n:>3}  {bar}")
    print("\nPhân bố theo intent:")
    for intent, n in sorted(intents.items(), key=lambda x: -x[1]):
        print(f"  {intent:<30} {n:>3}")
    print()


if __name__ == "__main__":
    print_stats()