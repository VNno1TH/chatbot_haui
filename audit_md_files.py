def audit_md_files(md_dir: str = "data/processed/nganh"):
    """
    Kiểm tra tất cả file .md ngành có truong_khoa trong frontmatter chưa.
    Chạy: python chunker_patch.py
    """
    from pathlib import Path
    try:
        import frontmatter as fm
    except ImportError:
        print("pip install python-frontmatter")
        return
 
    md_path = Path(md_dir)
    if not md_path.exists():
        print(f"Directory not found: {md_dir}")
        return
 
    print(f"Auditing {md_dir}...\n")
    missing = []
    present = []
 
    for f in sorted(md_path.glob("**/*.md")):
        try:
            post = fm.load(str(f))
            meta = post.metadata
            if meta.get("loai") != "mo_ta_nganh":
                continue
            tk = meta.get("truong_khoa", "")
            if tk:
                present.append((f.name, tk))
            else:
                missing.append(f.name)
        except Exception as e:
            print(f"ERROR {f.name}: {e}")
 
    print(f"✓ Có truong_khoa ({len(present)}):")
    for name, tk in present:
        print(f"  {name:<60} → {tk}")
 
    if missing:
        print(f"\n✗ THIẾU truong_khoa ({len(missing)}) — cần thêm vào frontmatter:")
        for name in missing:
            print(f"  {name}")
    else:
        print("\n✓ Tất cả file đều có truong_khoa!")
 
 
if __name__ == "__main__":
    audit_md_files()