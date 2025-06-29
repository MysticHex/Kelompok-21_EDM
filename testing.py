import requests
import csv
import time
import re

job_categories = [
    "software engineer", "frontend", "backend", "fullstack", "mobile developer", 
    "devops", "data scientist", "data analyst", "data engineer", 
    "machine learning", "cloud computing", "cyber security", "IT support", 
    "system administrator", "network engineer", "database", "UI/UX", "QA", 
    "project manager", "product manager", "business analyst", "consultant", 
    "business development", "strategic planning", "management", 
    
    "akuntan", "finance", "tax", "audit", "treasury", "financial planning", 
    "investment", "risk management", "payroll", 
    
    "digital marketing", "content marketing", "social media", "SEO", 
    "brand management", "public relations", "market research", "copywriter", 
    "PPC", 
    
    "recruitment", "talent management", "compensation benefit", "training", 
    "organization development", 
    
    "sales", "account executive", "customer service", "retail", 
    "business relationship", "telemarketing", 
    
    "supply chain", "logistik", "purchasing", "inventory", "warehouse", 
    "production", "maintenance", "quality control", 
    
    "dokter", "perawat", "bidan", "apoteker", "nutritionist", "physiotherapist", 
    "medical representative", 
    
    "guru", "dosen", "instruktur", "education", "training", "tutor", 
    
    "graphic designer", "motion designer", "illustrator", "photographer", 
    "videographer", "creative director", 
    
    "admin", "sekretaris", "legal", "pengacara", "arsitek", "interior designer", 
    "civil engineer", "mechanic", "technician", "chef", "hotel", "event"
]

def assign_category(text, categories):
    """
    Mengembalikan kategori yang cocok dari daftar categories jika ditemukan di text.
    Jika tidak ada yang cocok, mengembalikan 'Lainnya'.
    """
    text = str(text).lower()
    for cat in categories:
        if cat.lower() in text:
            return cat.title()
    return "Lainnya"

def parse_salary(s):
    """Ekstrak gaji min dan max dari string gaji."""
    if not s or s == "Tidak dicantumkan":
        return None, None
    nums = re.findall(r'(\d[\d\.]+)', str(s).replace(',', ''))
    if len(nums) >= 2:
        return int(nums[0].replace('.','')), int(nums[1].replace('.',''))
    elif len(nums) == 1:
        return int(nums[0].replace('.','')), int(nums[0].replace('.',''))
    return None, None

def extract_experience(text):
    """Ambil tahun pengalaman kerja, hanya jika ada kata kunci pengalaman/experience."""
    if not text: return None
    patterns = [
        r'(?:pengalaman|experience|berpengalaman|years of experience|work experience)[^\d]{0,20}(\d+)\s*(?:tahun|years?)',
        r'min(?:imum)?\.?\s*(\d+)\s*(?:tahun|years?)\s*(?:pengalaman|experience)?',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return int(m.group(1))
    return None

def extract_age(text):
    """Ambil usia, hanya jika ada kata kunci usia/umur/age."""
    if not text: return None
    patterns = [
        r'(?:usia|max(?:imal)?|umur|berusia|age)[^\d]{0,20}(\d+)\s*(?:tahun|years?)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return int(m.group(1))
    return None

def extract_education(text):
    """Ekstrak jenjang pendidikan dari kualifikasi."""
    if not text: return None
    degrees = ["S1", "S2", "D3", "Diploma", "Bachelor", "Master", "Sarjana", "SMK", "SMA"]
    found = [d for d in degrees if re.search(r'\b{}\b'.format(d), text, re.IGNORECASE)]
    if found:
        return ",".join(found)
    return None

def extract_skills(text):
    """Ekstrak skill utama dari kualifikasi."""
    if not text: return None
    skills = ["excel", "python", "sap", "finance", "accounting", "sql", "communication", "microsoft office", "erp", "tax", "java", "jira"]
    found = [s for s in skills if s.lower() in text.lower()]
    return ",".join(found) if found else None

def fetch_jobstreet_jobs(max_pages=100, pagesize=32, out_csv="jobstreet_jobs_cleaned_with_category.csv"):
    BASE_URL = "https://id.jobstreet.com/api/jobsearch/v5/search"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://id.jobstreet.com/id/jobs",
        
    }
    params = {
        "sitekey": "ID-Main",
        "sourcesystem": "houston",
        "locale": "id-ID",
        "page": 1,
        "pagesize": pagesize,
    }

    resp = requests.get(BASE_URL, headers=HEADERS, params=params)
    data = resp.json()
    userQueryId = data.get("userQueryId") or data.get("userqueryid")
    total_count = int(data.get("totalCount", 0))
    print(f"[INFO] Total lowongan ditemukan: {total_count}")

    if not userQueryId:
        print("[ERROR] Tidak dapat userQueryId!")
        print(data)
        return

    all_jobs = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        params["userqueryid"] = userQueryId
        resp = requests.get(BASE_URL, headers=HEADERS, params=params)
        data = resp.json()
        jobs = data.get("data", [])
        if not jobs:
            print("[INFO] No more jobs found, stopping.")
            break
        all_jobs.extend(jobs)
        print(f"[INFO] Page {page}: total jobs so far: {len(all_jobs)}")
        time.sleep(1)
        if len(all_jobs) >= total_count:
            break

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Kategori Lowongan", "Pekerjaan Lowongan", "Title", "Posisi", "Gaji", "Gaji Min", "Gaji Max",
            "Kualifikasi", "Tahun Pengalaman", "Umur", "Pendidikan", "Skill", "Link"
        ])
        for job in all_jobs:
            title = job.get("title", "")
            posisi = title
            salary_str = job.get("salaryLabel") or job.get("salary", {}).get("display") or "Tidak dicantumkan"
            gaji_min, gaji_max = parse_salary(salary_str)
            kualifikasi = job.get("teaser", "") or ""
            tahun_pengalaman = extract_experience(kualifikasi)
            umur = extract_age(kualifikasi)
            pendidikan = extract_education(kualifikasi)
            skill = extract_skills(kualifikasi)
            # Kategori: dicari dari gabungan Title dan Kualifikasi
            kategori_lowongan = assign_category(f"{title} {kualifikasi}", job_categories)
            # Kolom Pekerjaan Lowongan diisi dengan hasil assign_category, bukan "Semua Lowongan"
            pekerjaan_lowongan = kategori_lowongan
            link = f"https://id.jobstreet.com/id/job/{job.get('id')}"
            writer.writerow([
                kategori_lowongan, pekerjaan_lowongan, title, posisi, salary_str, gaji_min or "N/A", gaji_max or "N/A",
                kualifikasi if kualifikasi else "N/A", tahun_pengalaman or "N/A", umur or "N/A",
                pendidikan or "N/A", skill or "N/A", link
            ])
    print(f"[INFO] Saved {len(all_jobs)} cleaned jobs with category to {out_csv}")

if __name__ == "__main__":
    fetch_jobstreet_jobs(max_pages=16, pagesize=32, out_csv="jobstreet_jobs_cleaned_16.csv")