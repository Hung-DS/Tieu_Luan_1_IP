#!/usr/bin/env python3
"""
Script để chạy ứng dụng xử lý ảnh Streamlit
"""

import subprocess
import sys
import os

def main():
    print("🚀 Khởi động ứng dụng Xử lý Ảnh...")
    print("📱 Ứng dụng sẽ mở trong trình duyệt web")
    print("🔗 Địa chỉ: http://localhost:8501")
    print("⏹️  Nhấn Ctrl+C để dừng ứng dụng")
    print("-" * 50)
    
    try:
        # Chạy Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng ứng dụng")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("💡 Hãy đảm bảo đã cài đặt đầy đủ thư viện:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 