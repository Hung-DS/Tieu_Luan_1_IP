import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import pillow_heif

# ==============================
# Các hàm xử lý cường độ sáng
# ==============================

def negative_image(img):
    """Tạo ảnh âm"""
    if img is None:
        return None
    return 255 - img

def log_transform(img, c=1):
    """Biến đổi logarit"""
    if img is None:
        return None
    img_float = img.astype(np.float32) / 255.0
    log_img = c * np.log(1 + img_float)
    log_img = np.uint8(255 * (log_img / np.max(log_img)))
    return log_img

def gamma_transform(img, gamma=1.0):
    """Biến đổi gamma"""
    if img is None:
        return None
    img_float = img.astype(np.float32) / 255.0
    gamma_img = np.power(img_float, gamma)
    gamma_img = np.uint8(255 * gamma_img)
    return gamma_img

def piecewise_linear(img):
    """Biến đổi piecewise-linear"""
    if img is None:
        return None
    # tăng độ sáng vùng tối, giảm độ sáng vùng sáng
    img_float = img.astype(np.float32) / 255.0
    piecewise = np.piecewise(img_float,
                             [img_float < 0.3, (img_float >= 0.3) & (img_float < 0.7), img_float >= 0.7],
                             [lambda x: 0.5 * x, lambda x: x, lambda x: 0.2 + 0.8 * x])
    piecewise = np.uint8(255 * piecewise)
    return piecewise

# ==============================
# Các hàm cân bằng mức xám (Histogram Equalization)
# ==============================

def clahe_equalization(img, clip=2.0, tile=(8,8)):
    """Cân bằng histogram thích ứng (CLAHE)"""
    if img is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img)

def hist_equalization(img):
    """Cân bằng histogram toàn cục"""
    if img is None:
        return None
    return cv2.equalizeHist(img)

def custom_hist_equalization(img):
    """Tự viết hàm cân bằng histogram"""
    if img is None:
        return None
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    img_eq = cdf_normalized[img]
    return img_eq

# ==============================
# Ứng dụng biến đổi ảnh cơ bản trong thực tế: tăng cường chất lượng ảnh
# ==============================

def preprocess_license_plate(img):
    """Tiền xử lý ảnh cho nhận dạng biển số xe"""
    # Chuyển sang ảnh xám
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Cân bằng histogram thích ứng
    eq = clahe_equalization(gray)
    
    # Làm mịn ảnh(lọc nhiễu)
    blur = cv2.GaussianBlur(eq, (3,3), 0)
    
    # Tăng độ tương phản
    enhanced = cv2.convertScaleAbs(blur, alpha=1.2, beta=10)
    
    return enhanced

def enhance_satellite(img):
    """Cải thiện ảnh vệ tinh trong GIS"""
    # Chuyển sang LAB color space
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Cân bằng histogram cho kênh L (độ sáng)
        l_eq = clahe_equalization(l)
        
        # Tăng độ bão hòa cho kênh a và b
        a_enhanced = cv2.convertScaleAbs(a, alpha=1.3, beta=0)
        b_enhanced = cv2.convertScaleAbs(b, alpha=1.3, beta=0)
        
        # Ghép lại
        merged = cv2.merge((l_eq, a_enhanced, b_enhanced))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    else:
        # Nếu là ảnh xám
        return clahe_equalization(img)

def enhance_low_light(img):
    """Nâng cao chất lượng ảnh chụp trong điều kiện ánh sáng kém"""
    if len(img.shape) == 3:
        # Chuyển sang HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Cân bằng histogram cho kênh V (độ sáng)
        v_eq = clahe_equalization(v)
        
        # Tăng độ bão hòa
        s_enhanced = cv2.convertScaleAbs(s, alpha=1.2, beta=0)
        
        # Ghép lại
        final = cv2.merge((h, s_enhanced, v_eq))
        return cv2.cvtColor(final, cv2.COLOR_HSV2RGB)
    else:
        # Nếu là ảnh xám
        return clahe_equalization(img)




# ==============================
# Utility Functions
# ==============================

def load_image(uploaded_file):
    """Load ảnh từ file upload, hỗ trợ HEIC"""
    try:
        # Kiểm tra nếu là file HEIC
        if uploaded_file.name.lower().endswith(('.heic', '.heif')):
            # Sử dụng pillow-heif để đọc HEIC
            heif_file = pillow_heif.read_heif(uploaded_file.read())
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data, 
                "raw", 
                heif_file.mode, 
                heif_file.stride,
            )
            # Chuyển sang RGB nếu cần
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Chuyển sang numpy array
            image_array = np.array(image)
            return image_array
        else:
            # Xử lý các định dạng khác
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return None
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {str(e)}")
        return None

def plot_comparison(original, processed, title):
    """Vẽ so sánh ảnh gốc và ảnh đã xử lý"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if len(original.shape) == 3:
        axes[0].imshow(original)
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Ảnh gốc')
    axes[0].axis('off')
    
    if len(processed.shape) == 3:
        axes[1].imshow(processed)
    else:
        axes[1].imshow(processed, cmap='gray')
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def plot_histogram_comparison(original, processed, title):
    """Vẽ so sánh histogram trước và sau xử lý"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Ảnh gốc
    if len(original.shape) == 3:
        axes[0,0].imshow(original)
    else:
        axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Ảnh gốc')
    axes[0,0].axis('off')
    
    # Ảnh đã xử lý
    if len(processed.shape) == 3:
        axes[0,1].imshow(processed)
    else:
        axes[0,1].imshow(processed, cmap='gray')
    axes[0,1].set_title(title)
    axes[0,1].axis('off')
    
    # Histogram ảnh gốc
    if len(original.shape) == 3:
        gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    else:
        gray_orig = original.copy()
    hist_orig = cv2.calcHist([gray_orig], [0], None, [256], [0, 256])
    axes[1,0].plot(hist_orig, color='blue', alpha=0.7)
    axes[1,0].set_title('Histogram ảnh gốc')
    axes[1,0].set_xlabel('Mức xám')
    axes[1,0].set_ylabel('Số pixel')
    axes[1,0].grid(True, alpha=0.3)
    
    # Histogram ảnh đã xử lý
    if len(processed.shape) == 3:
        gray_proc = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    else:
        gray_proc = processed.copy()
    hist_proc = cv2.calcHist([gray_proc], [0], None, [256], [0, 256])
    axes[1,1].plot(hist_proc, color='red', alpha=0.7)
    axes[1,1].set_title('Histogram ảnh đã xử lý')
    axes[1,1].set_xlabel('Mức xám')
    axes[1,1].set_ylabel('Số pixel')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def get_image_download_link(img, filename, text):
    """Tạo link download cho ảnh"""
    if len(img.shape) == 3:
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray(img, mode='L')
    
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    buf.seek(0)
    
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ==============================
# Main App
# ==============================

def main():
    st.set_page_config(
        page_title="Ứng dụng Xử lý Ảnh - Tiểu luận 1",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Ứng dụng Xử lý Ảnh - Tiểu luận 1")
    st.markdown("**Xử lý ảnh dựa trên giá trị điểm ảnh (Point Processing) - Phần 3: Ứng dụng thực tế**")
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📁 Tải ảnh lên")
    uploaded_file = st.sidebar.file_uploader(
        "Chọn file ảnh",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'heic', 'heif'],
        help="Hỗ trợ các định dạng: PNG, JPG, JPEG, BMP, TIFF, HEIC, HEIF"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Cài đặt")
    
    # Main content
    if uploaded_file is not None:
        # Đọc ảnh
        image = load_image(uploaded_file)
        
        if image is None:
            st.error("Không thể đọc file ảnh. Vui lòng thử file khác.")
            return
        
        # Hiển thị ảnh gốc
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📸 Ảnh gốc")
            st.image(image, caption="Ảnh gốc", use_container_width=True)
            
        
        with col2:
            st.subheader("🔍 Thông tin ảnh")
            
            # Chuyển sang ảnh xám để phân tích
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Hiển thị thông tin chi tiết về ảnh
            st.write(f"**📏 Kích thước:** {image.shape[1]} × {image.shape[0]} pixels")
            st.write(f"**🎨 Loại ảnh:** {'Màu RGB' if len(image.shape) == 3 else 'Ảnh xám'}")
            st.write(f"**🔢 Số kênh màu:** {image.shape[2] if len(image.shape) == 3 else 1}")
            st.write(f"**💾 Kiểu dữ liệu:** {image.dtype}")
            st.write(f"**📊 Phạm vi giá trị:** {image.min()} - {image.max()}")
            st.write(f"**📁 Định dạng file:** {uploaded_file.name.split('.')[-1].upper()}")
            
            # Thống kê cơ bản
            st.markdown("**📈 Thống kê cơ bản:**")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Độ sáng trung bình", f"{np.mean(gray):.1f}")
                st.metric("Độ tương phản (std)", f"{np.std(gray):.1f}")
            with col2b:
                st.metric("Độ sáng min", f"{np.min(gray)}")
                st.metric("Độ sáng max", f"{np.max(gray)}")
        
            
        
        st.markdown("---")
        
        # Chọn chức năng xử lý
        st.subheader("🛠️ Chọn chức năng xử lý")
        
        # Nhóm các chức năng xử lý cơ bản
        basic_processing = {
            "🔄 Ảnh âm (Negative)": negative_image,
            "📊 Biến đổi Logarit": log_transform,
            "⚡ Biến đổi Gamma": gamma_transform,
            "📈 Biến đổi Piecewise-linear": piecewise_linear,
        }
        
        # Nhóm các chức năng cân bằng histogram
        histogram_processing = {
            "📊 Cân bằng Histogram toàn cục": hist_equalization,
            "🎯 Cân bằng Histogram thích ứng (CLAHE)": clahe_equalization,
            "✏️ Tự viết hàm cân bằng Histogram": custom_hist_equalization,
        }
        
        # Nhóm các chức năng ứng dụng thực tế
        application_processing = {
            "🚗 Tiền xử lý ảnh biển số xe": preprocess_license_plate,
            "🛰️ Cải thiện ảnh vệ tinh GIS": enhance_satellite,
            "🌙 Nâng cao ảnh ánh sáng kém": enhance_low_light,
        }
        
        # Tạo selectbox với các nhóm chức năng
        st.markdown("**🔧 Xử lý ảnh cơ bản:**")
        basic_option = st.selectbox(
            "Chọn chức năng xử lý cơ bản:",
            ["Không chọn"] + list(basic_processing.keys())
        )
        
        # Thêm thanh trượt cho gamma nếu chọn biến đổi gamma
        gamma_value = 1.0
        if basic_option == "⚡ Biến đổi Gamma":
            gamma_value = st.slider(
                "Chọn chỉ số Gamma (γ):",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="γ < 1: tăng độ sáng, γ > 1: giảm độ sáng, γ = 1: không thay đổi"
            )
            st.write(f"**Giá trị Gamma hiện tại:** {gamma_value}")
        
        st.markdown("**📊 Cân bằng Histogram:**")
        histogram_option = st.selectbox(
            "Chọn chức năng cân bằng histogram:",
            ["Không chọn"] + list(histogram_processing.keys())
        )
        
        st.markdown("**🎯 Ứng dụng thực tế:**")
        app_option = st.selectbox(
            "Chọn chức năng ứng dụng:",
            ["Không chọn"] + list(application_processing.keys())
        )
        
        # Hiển thị lý thuyết cho các chức năng được chọn
        st.markdown("---")
        st.subheader("📚 Lý thuyết và Phân tích")
        
        # Lý thuyết cho xử lý ảnh cơ bản
        if basic_option != "Không chọn":
            if "Negative" in basic_option:
                st.info(f"""
                **{basic_option}:**
                
                **📖 Lý thuyết:**
                Công thức: s = 255 - r
                Trong đó: r là giá trị pixel gốc, s là giá trị pixel sau xử lý
                
                
                **✨ Ý nghĩa:** Làm nổi chi tiết ở vùng sáng (x-quang, ảnh y khoa).

                """)
            elif "Logarit" in basic_option:
                st.info(f"""
                **{basic_option}:**
                
                **📖 Lý thuyết:**
                Công thức: s = c × log(1 + r)
                Trong đó: c là hằng số, r là giá trị pixel gốc (0-255)
                
                
                **✨ Ý nghĩa:** Nổi bật vùng tối, nén vùng quá sáng (ảnh thiên văn, vệ tinh).

                """)
            elif "Gamma" in basic_option:
                st.info(f"""
                **{basic_option}:**
                
                **📖 Lý thuyết:**
                Công thức: s = c × r^γ
                Trong đó: c là hằng số, γ = {gamma_value} (hệ số gamma), r là giá trị pixel gốc
                
                
                **✨ Ý nghĩa:** Điều chỉnh theo đặc tính hiển thị & cảm nhận thị giác.

                """)
            elif "Piecewise" in basic_option:
                st.info(f"""
                **{basic_option}:**
                
                **📖 Lý thuyết:**
                Công thức: s = f(r) với f(r) là hàm từng đoạn
                - r < 0.3: s = 0.5 × r (tăng độ sáng vùng tối)
                - 0.3 ≤ r < 0.7: s = r (giữ nguyên)
                - r ≥ 0.7: s = 0.2 + 0.8 × r (giảm độ sáng vùng sáng)
                
                
                **✨ Ý nghĩa:** Làm rõ chi tiết trong khoảng quan tâm, tăng tương phản cục bộ.
                """)
        
        # Lý thuyết cho cân bằng histogram
        if histogram_option != "Không chọn":
            if "Histogram toàn cục" in histogram_option:
                st.info(f"""
                **{histogram_option}:**
                
                **📖 Lý thuyết:**
                Công thức: s = T(r) = (L-1) × ∫₀ʳ pᵣ(w)dw
                Trong đó: L là số mức xám, pᵣ(w) là PDF của ảnh gốc
                
                **🎯 Ứng dụng:** Cân bằng histogram cho toàn bộ ảnh
                
                **✨ Cải thiện:** Tăng độ tương phản toàn cục, phân bố đều mức xám
                """)
            elif "CLAHE" in histogram_option:
                st.info(f"""
                **{histogram_option}:**
                
                **📖 Lý thuyết:**
                Công thức: s = T(r) với T(r) được tính trên từng tile nhỏ
                Clip limit: giới hạn độ tăng histogram để tránh nhiễu
                
                **🎯 Ứng dụng:** Cân bằng histogram thích ứng, xử lý ảnh có vùng sáng/tối khác nhau
                
                **✨ Cải thiện:** Tăng độ tương phản cục bộ, giữ chi tiết tốt hơn
                """)
            elif "Tự viết" in histogram_option:
                st.info(f"""
                **{histogram_option}:**
                
                **📖 Lý thuyết:**
                Công thức: s = T(r) = (L-1) × (CDF(r) - CDF_min) / (CDF_max - CDF_min)
                Trong đó: CDF là hàm phân phối tích lũy
                
                **🎯 Ứng dụng:** Hiểu rõ thuật toán cân bằng histogram
                
                **✨ Cải thiện:** Tương tự histogram toàn cục, nhưng tự implement
                """)
        
        # Lý thuyết cho ứng dụng thực tế
        if app_option != "Không chọn":
            if "biển số xe" in app_option:
                st.info(f"""
                **{app_option}:**
                
                **📖 Lý thuyết:**
                Sử dụng CLAHE để cân bằng histogram thích ứng
                Gaussian blur để làm mịn nhiễu
                Tăng độ tương phản với alpha và beta
                
                **🎯 Ứng dụng:** Tiền xử lý ảnh biển số xe trước khi nhận dạng ký tự (OCR)
                
                **✨ Cải thiện:** Tăng độ tương phản, làm mịn nhiễu, chuẩn bị cho bước tiếp theo
                """)
            elif "vệ tinh" in app_option:
                st.info(f"""
                **{app_option}:**
                
                **📖 Lý thuyết:**
                Chuyển sang LAB color space
                Cân bằng histogram cho kênh L (độ sáng)
                Tăng độ bão hòa cho kênh a và b
                
                **🎯 Ứng dụng:** Cải thiện ảnh vệ tinh trong hệ thống thông tin địa lý (GIS)
                
                **✨ Cải thiện:** Tăng độ sắc nét, cải thiện màu sắc, dễ dàng phân tích địa hình
                """)
            elif "ánh sáng kém" in app_option:
                st.info(f"""
                **{app_option}:**
                
                **📖 Lý thuyết:**
                Chuyển sang HSV color space
                Cân bằng histogram cho kênh V (độ sáng)
                Tăng độ bão hòa kênh S
                
                **🎯 Ứng dụng:** Nâng cao chất lượng ảnh chụp trong điều kiện thiếu sáng
                
                **✨ Cải thiện:** Tăng độ sáng, giảm nhiễu, cải thiện độ tương phản
                """)
            elif "tài liệu" in app_option:
                st.info(f"""
                **{app_option}:**
                
                **📖 Lý thuyết:**
                Cân bằng histogram toàn cục
                Làm sắc nét với kernel [[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]
                Tăng độ tương phản
                
                **🎯 Ứng dụng:** Cải thiện ảnh tài liệu bị mờ, nhòe
                
                **✨ Cải thiện:** Làm sắc nét chữ, tăng độ tương phản, dễ đọc hơn
                """)
            elif "sách cũ" in app_option:
                st.info(f"""
                **{app_option}:**
                
                **📖 Lý thuyết:**
                Chuyển sang LAB color space
                Cân bằng histogram cho kênh L
                Giảm độ vàng (kênh b) với alpha = 0.8
                
                **🎯 Ứng dụng:** Cải thiện ảnh sách cũ, giấy ố vàng
                
                **✨ Cải thiện:** Giảm độ vàng, tăng độ sắc nét, cải thiện khả năng đọc
                """)
            elif "hóa đơn" in app_option:
                st.info(f"""
                **{app_option}:**
                
                **📖 Lý thuyết:**
                CLAHE với clip limit cao hơn (3.0) và tile nhỏ hơn (4x4)
                Làm sắc nét với kernel
                Tăng độ tương phản với alpha = 1.4, beta = 5
                
                **🎯 Ứng dụng:** Cải thiện ảnh hóa đơn, chữ nhỏ
                
                **✨ Cải thiện:** Tăng độ sắc nét, cải thiện độ tương phản, dễ nhận dạng
                """)
            
        
        # Xử lý ảnh
        if st.button("🚀 Xử lý ảnh", type="primary"):
            with st.spinner("Đang xử lý ảnh..."):
                results = {}
                
                # Xử lý các chức năng được chọn
                if basic_option != "Không chọn":
                    if basic_option == "⚡ Biến đổi Gamma":
                        results[basic_option] = gamma_transform(gray, gamma_value)
                    else:
                        results[basic_option] = basic_processing[basic_option](gray)
                
                if histogram_option != "Không chọn":
                    results[histogram_option] = histogram_processing[histogram_option](gray)
                
                if app_option != "Không chọn":
                    results[app_option] = application_processing[app_option](image)
                
                if not results:
                    st.warning("Vui lòng chọn ít nhất một chức năng xử lý!")
                    return
                
                # Hiển thị kết quả
                st.markdown("---")
                st.subheader("✨ Kết quả xử lý")
                
                # Hiển thị từng kết quả
                for title, processed_image in results.items():
                    if processed_image is not None:
                        st.markdown(f"**{title}:**")
                        
                        # Kiểm tra nếu là chức năng cân bằng histogram thì hiển thị histogram so sánh
                        if any(keyword in title for keyword in ["Histogram", "CLAHE"]):
                            comparison_fig = plot_histogram_comparison(gray, processed_image, title)
                            st.pyplot(comparison_fig)
                            plt.close()
                        else:
                            # So sánh trước/sau thông thường
                            comparison_fig = plot_comparison(gray, processed_image, title)
                            st.pyplot(comparison_fig)
                            plt.close()
                        
                        # Thống kê
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Độ tương phản gốc", f"{np.std(gray):.2f}")
                            st.metric("Độ tương phản sau xử lý", f"{np.std(processed_image):.2f}")
                        
                        with col2:
                            st.metric("Độ sáng trung bình gốc", f"{np.mean(gray):.2f}")
                            st.metric("Độ sáng trung bình sau xử lý", f"{np.mean(processed_image):.2f}")
                        
                        # Download button
                        st.markdown(get_image_download_link(
                            processed_image, 
                            f"processed_{uploaded_file.name.split('.')[0]}_{title.replace(' ', '_')}.png", 
                            f"📥 Tải ảnh {title}"
                        ), unsafe_allow_html=True)
                        
                        st.markdown("---")
                
                # Phân tích trước và sau cho ứng dụng thực tế
                if app_option != "Không chọn":
                    st.markdown("---")
                    st.subheader("🔍 Phân tích trước và sau xử lý")
                    
                    if "biển số xe" in app_option:
                        st.info("""
                        **🚗 Tiền xử lý ảnh biển số xe - Phân tích:**
                        
                        **📊 Trước xử lý:**
                        - Ảnh gốc có thể bị mờ, thiếu độ tương phản
                        - Nhiễu và bóng mờ có thể che khuất ký tự
                        - Độ sáng không đều giữa các vùng
                        
                        **✨ Sau xử lý:**
                        - **CLAHE:** Tăng độ tương phản cục bộ, làm nổi bật ký tự
                        - **Gaussian Blur:** Làm mịn nhiễu, giữ nguyên chi tiết quan trọng
                        - **Tăng độ tương phản:** Làm rõ ranh giới giữa ký tự và nền
                        
                        **🎯 Kết quả mong đợi:**
                        - Ký tự biển số rõ ràng hơn, dễ nhận dạng
                        - Giảm nhiễu và bóng mờ
                        - Tăng độ tương phản giữa chữ và nền
                        """)
                    
                    elif "vệ tinh" in app_option:
                        st.info("""
                        **🛰️ Cải thiện ảnh vệ tinh GIS - Phân tích:**
                        
                        **📊 Trước xử lý:**
                        - Ảnh vệ tinh có thể bị mờ, thiếu độ sắc nét
                        - Màu sắc không tự nhiên, thiếu độ bão hòa
                        - Độ tương phản thấp, khó phân biệt địa hình
                        
                        **✨ Sau xử lý:**
                        - **LAB Color Space:** Tách riêng độ sáng và màu sắc
                        - **CLAHE cho kênh L:** Tăng độ sắc nét và độ tương phản
                        - **Tăng độ bão hòa:** Làm nổi bật đặc điểm địa hình
                        
                        **🎯 Kết quả mong đợi:**
                        - Địa hình rõ ràng hơn, dễ phân tích
                        - Màu sắc tự nhiên và sinh động
                        - Tăng khả năng nhận dạng đối tượng
                        """)
                    
                    elif "ánh sáng kém" in app_option:
                        st.info("""
                        **🌙 Nâng cao ảnh ánh sáng kém - Phân tích:**
                        
                        **📊 Trước xử lý:**
                        - Ảnh tối, thiếu chi tiết ở vùng bóng
                        - Nhiễu cao do ISO cao trong điều kiện thiếu sáng
                        - Màu sắc bị mất đi, thiếu độ bão hòa
                        
                        **✨ Sau xử lý:**
                        - **HSV Color Space:** Tách riêng màu sắc và độ sáng
                        - **CLAHE cho kênh V:** Tăng độ sáng vùng tối, giữ chi tiết
                        - **Tăng độ bão hòa:** Khôi phục màu sắc tự nhiên
                        
                        **🎯 Kết quả mong đợi:**
                        - Vùng tối được làm sáng, hiển thị chi tiết
                        - Giảm nhiễu và tăng độ mượt mà
                        - Màu sắc được khôi phục và tăng cường
                        """)
    
    else:
        # Hướng dẫn sử dụng
        st.info("👆 **Hướng dẫn sử dụng:** Tải ảnh lên từ sidebar bên trái để bắt đầu xử lý")
        
        st.markdown("---")
        
        # Giới thiệu các chức năng
        st.subheader("🎯 Các chức năng chính")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔧 Xử lý ảnh cơ bản
            - **🔄 Ảnh âm (Negative):** Tạo ảnh âm, hữu ích trong y tế
            - **📊 Biến đổi Logarit:** Nén động dải, xử lý ảnh tương phản cao
            - **⚡ Biến đổi Gamma:** Điều chỉnh độ sáng theo hàm mũ
            - **📈 Biến đổi Piecewise-linear:** Xử lý vùng sáng/tối khác nhau
            
            ### 📊 Cân bằng Histogram
            - **📊 Histogram toàn cục:** Cân bằng toàn bộ ảnh
            - **🎯 CLAHE:** Cân bằng thích ứng, giữ chi tiết tốt hơn
            - **✏️ Tự viết hàm:** Hiểu rõ thuật toán
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Ứng dụng thực tế
            - **🚗 Biển số xe:** Tiền xử lý cho OCR
            - **🛰️ Ảnh vệ tinh:** Cải thiện cho GIS
            - **🌙 Ánh sáng kém:** Nâng cao chất lượng
            """)
        
        st.markdown("---")
        
        # Yêu cầu bài tập
        st.subheader("📋 Yêu cầu bài tập")
        st.markdown("""
        **Phần 3: Ứng dụng biến đổi ảnh cơ bản trong thực tế**
        
        **🔧 Xử lý ảnh cơ bản:**
        1. **Negative Image (Ảnh âm)**
        2. **Log Transformation (Biến đổi log)**
        3. **Power-law / Gamma Correction**
        4. **Piecewise-linear Transformation**
        
        **📊 Thuật toán cân bằng histogram:**
        5. **Histogram Equalization toàn cục**
        6. **Adaptive Histogram Equalization (CLAHE)**
        7. **Tự viết hàm cân bằng histogram**
        
        **🎯 Ứng dụng thực tế:**
        8. **Tiền xử lý ảnh cho nhận dạng biển số xe**
        9. **Cải thiện ảnh vệ tinh trong GIS**
        10. **Nâng cao chất lượng ảnh chụp trong điều kiện ánh sáng kém**
        
        **Mục tiêu:** Phân tích kết quả và so sánh trước/sau xử lý
        """)
    


if __name__ == "__main__":
    main() 