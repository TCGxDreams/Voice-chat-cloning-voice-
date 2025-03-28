import speech_recognition as sr
import numpy as np
import torch
import os
import time
import platform
import soundfile as sf
from google import genai
from google.genai import types
import hashlib
import shutil

# Thêm thư viện TTS từ Mozilla (cung cấp khả năng voice cloning)
from TTS.api import TTS

# Xác định hệ điều hành để sử dụng player phù hợp
system = platform.system()

# Kiểm tra và thiết lập thiết bị tính toán tốt nhất cho Mac
if system == 'Darwin':  # macOS
    # Kiểm tra MPS (Metal Performance Shaders) cho Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\033[32mSử dụng GPU Apple Silicon (MPS) cho tăng tốc\033[0m")
        use_gpu = True
    else:
        device = torch.device("cpu")
        print("\033[33mSử dụng CPU (không tìm thấy GPU trên Mac)\033[0m")
        use_gpu = False
else:
    # Kiểm tra CUDA cho Windows/Linux
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\033[32mSử dụng NVIDIA GPU (CUDA)\033[0m")
        use_gpu = True
    else:
        device = torch.device("cpu")
        print("\033[33mSử dụng CPU (không tìm thấy GPU)\033[0m") 
        use_gpu = False

def play_audio(filename):
    """Phát file âm thanh dựa trên hệ điều hành"""
    try:
        if system == 'Windows':
            from playsound import playsound
            playsound(filename)
        elif system == 'Darwin':  # macOS
            os.system(f"afplay {filename}")
        else:  # Linux và các hệ điều hành khác
            os.system(f"mpg123 {filename}")
    except Exception as e:
        print(f"Lỗi khi phát âm thanh: {e}")
        print("Hãy chắc chắn bạn đã cài đặt thư viện phát âm thanh phù hợp.")
        if system == 'Linux':
            print("Trên Linux, chạy: sudo apt-get install mpg123")

class CustomVoiceTTS:
    def __init__(self, reference_audio_path=None):
        """Khởi tạo mô hình TTS với khả năng sao chép giọng nói.
        
        Args:
            reference_audio_path: Đường dẫn đến file âm thanh mẫu để sao chép giọng nói.
                                 Nếu None, sẽ sử dụng giọng mặc định.
        """
        print("Đang khởi tạo mô hình TTS...")
        
        # Tạo thư mục cache nếu chưa tồn tại
        os.makedirs("tts_cache", exist_ok=True)
        
        # Khởi tạo cache cho các phản hồi đã tổng hợp
        self.response_cache = {}
        self.tts = None
        
        try:
            # Lần lượt thử các mô hình từ đơn giản đến phức tạp
            models_to_try = [
                # Mô hình tiếng Anh cơ bản - thường được cài đặt sẵn
                "tts_models/en/ljspeech/fast_pitch",
                # Mô hình tiếng Anh phổ biến
                "tts_models/en/ljspeech/tacotron2-DDC",
                # Mô hình đa ngôn ngữ với khả năng voice cloning
                "tts_models/multilingual/multi-dataset/your_tts"
            ]
            
            # Nếu là Mac không có GPU, thì chỉ thử mô hình đơn giản
            if system == 'Darwin' and not use_gpu:
                print("\033[33mMac không có GPU, sử dụng mô hình nhẹ\033[0m")
                models_to_try = models_to_try[:2]  # Chỉ dùng 2 mô hình đầu
            
            for model_name in models_to_try:
                try:
                    print(f"Đang thử tải mô hình: {model_name}")
                    self.tts = TTS(
                        model_name=model_name,
                        progress_bar=True,
                        gpu=use_gpu
                    )
                    print(f"\033[32mĐã tải thành công mô hình: {model_name}\033[0m")
                    break  # Thoát vòng lặp nếu tải thành công
                except Exception as e:
                    print(f"\033[31mKhông thể tải mô hình {model_name}: {e}\033[0m")
                    continue
            
            # Kiểm tra nếu không tải được mô hình nào
            if self.tts is None:
                print("\033[31mKhông thể tải bất kỳ mô hình TTS nào. Chuyển sang gTTS.\033[0m")
                
            # Lưu lại đường dẫn âm thanh tham chiếu
            self.reference_audio_path = reference_audio_path
            
            if self.tts:
                print("\033[32mKhởi tạo mô hình TTS thành công!\033[0m")
                # Hiển thị thông tin về mô hình
                print(f"Mô hình: {self.tts.model_name}")
                print(f"Ngôn ngữ hỗ trợ: {self.tts.languages if hasattr(self.tts, 'languages') else 'Không rõ'}")
            
        except Exception as e:
            print(f"\033[31mLỗi khi khởi tạo mô hình TTS: {e}\033[0m")
            print("Sẽ sử dụng gTTS làm phương án dự phòng.")
            self.tts = None
    
    def create_voice_sample(self, sample_text="Xin chào, đây là mẫu giọng nói của tôi để sử dụng cho trợ lý ảo."):
        """Tạo mẫu giọng nói mới từ microphone.
        
        Args:
            sample_text: Văn bản đề nghị người dùng đọc để lấy mẫu giọng nói.
        
        Returns:
            Đường dẫn đến file âm thanh mẫu.
        """
        print(f"\nĐể tạo mẫu giọng nói, vui lòng đọc đoạn văn bản sau:")
        print(f"\033[36m{sample_text}\033[0m")
        
        # Khởi tạo recognizer
        r = sr.Recognizer()
        
        # Ghi âm từ microphone
        with sr.Microphone() as source:
            print("\n\033[33mĐang ghi âm... Hãy nói rõ ràng.\033[0m")
            # Điều chỉnh theo tiếng ồn xung quanh
            r.adjust_for_ambient_noise(source, duration=1)
            
            # Ghi âm mẫu giọng nói
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
            
            print("\033[33mĐã ghi âm xong, đang lưu file...\033[0m")
            
            # Lưu file âm thanh
            sample_path = "voice_sample.wav"
            with open(sample_path, "wb") as f:
                f.write(audio.get_wav_data())
            
            print(f"\033[32mĐã lưu mẫu giọng nói tại: {sample_path}\033[0m")
            
            # Cập nhật đường dẫn
            self.reference_audio_path = sample_path
            
            return sample_path
    
    def speak(self, text):
        """Chuyển văn bản thành giọng nói sử dụng voice cloning và cache.
        
        Args:
            text: Văn bản cần chuyển thành giọng nói.
        """
        # Tạo key cho cache
        cache_key = text.strip().lower()
        hash_object = hashlib.md5(cache_key.encode())
        cache_hash = hash_object.hexdigest()
        cache_file = f"tts_cache/cached_{cache_hash}.wav"
        
        # Kiểm tra cache
        if os.path.exists(cache_file):
            print("\033[32mSử dụng phản hồi từ cache...\033[0m")
            play_audio(cache_file)
            return
        
        output_path = "response.wav"
        
        # Kiểm tra xem mô hình TTS có sẵn sàng không
        if self.tts is None:
            self._fallback_gtts(text, output_path)
            # Lưu vào cache
            try:
                shutil.copy(output_path, cache_file)
            except:
                pass
            return
        
        try:
            # Kiểm tra xem mô hình có hỗ trợ voice cloning không
            model_supports_cloning = hasattr(self.tts, 'voice_cloning')
            
            # Kiểm tra ngôn ngữ hỗ trợ
            supported_languages = getattr(self.tts, 'languages', [])
            
            # Kiểm tra ngôn ngữ và thiết lập tiếng Việt nếu được hỗ trợ
            language = "vi" if "vi" in supported_languages else "en"
            
            # Thử sử dụng voice cloning nếu được hỗ trợ
            if self.reference_audio_path and model_supports_cloning:
                # Sử dụng voice cloning
                speaker_wav = self.reference_audio_path
                
                print(f"\033[36mĐang tổng hợp giọng nói sử dụng mẫu với ngôn ngữ: {language}\033[0m")
                
                # Kiểm tra xem phương thức có hỗ trợ các tham số này không
                try:
                    self.tts.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=speaker_wav,
                        language=language
                    )
                except TypeError:
                    # Nếu không, thử với ít tham số hơn
                    self.tts.tts_to_file(
                        text=text,
                        file_path=output_path
                    )
            else:
                # Sử dụng giọng mặc định
                print(f"\033[36mĐang tổng hợp giọng nói với ngôn ngữ: {language}\033[0m")
                
                # Kiểm tra xem có cần tham số language không
                try:
                    self.tts.tts_to_file(text=text, file_path=output_path, language=language)
                except TypeError:
                    self.tts.tts_to_file(text=text, file_path=output_path)
            
            # Lưu vào cache
            try:
                shutil.copy(output_path, cache_file)
            except Exception as e:
                print(f"\033[31mLỗi khi lưu vào cache: {e}\033[0m")
            
            # Phát âm thanh
            play_audio(output_path)
            
            # Xóa file tạm sau khi phát
            time.sleep(0.5)
            try:
                os.remove(output_path)
            except:
                pass
                
        except Exception as e:
            print(f"\033[31mLỗi khi tổng hợp giọng nói: {e}\033[0m")
            print("Chuyển sang sử dụng gTTS...")
            self._fallback_gtts(text, output_path)
            # Lưu vào cache nếu thành công
            try:
                if os.path.exists(output_path):
                    shutil.copy(output_path, cache_file)
            except:
                pass
    
    def _fallback_gtts(self, text, output_path):
        """Sử dụng gTTS làm phương án dự phòng khi TTS không hoạt động.
        
        Args:
            text: Văn bản cần chuyển thành giọng nói.
            output_path: Đường dẫn lưu file âm thanh.
        """
        try:
            from gtts import gTTS
            print("\033[33mSử dụng gTTS thay thế...\033[0m")
            tts = gTTS(text=text, lang='vi', slow=False)
            tts.save(output_path)
            
            # Phát âm thanh
            play_audio(output_path)
            
            # Xóa file sau khi phát
            time.sleep(0.5)
            try:
                os.remove(output_path)
            except:
                pass
        except Exception as e:
            print(f"\033[31mLỗi khi sử dụng gTTS: {e}\033[0m")

def generate_gemini_response(text):
    """Tạo phản hồi sử dụng Gemini API."""
    try:
        # Kiểm tra API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("\033[31mLỗi: Không tìm thấy GEMINI_API_KEY trong biến môi trường\033[0m")
            return "Xin lỗi, tôi không thể kết nối với dịch vụ AI. Vui lòng kiểm tra cài đặt API key."
        
        # Khởi tạo client
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-pro-exp-03-25"
        
        # Chuẩn bị nội dung để gửi tới API
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"""Người dùng nói bằng tiếng Việt: "{text}". 
Hãy trả lời ngắn gọn, thân thiện và tự nhiên bằng tiếng Việt. 
Giữ câu trả lời dưới 50 từ."""
                    ),
                ],
            ),
        ]
        
        # Cấu hình và gọi API
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )
        
        # Sử dụng phương thức không stream để nhận phản hồi đầy đủ
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Lấy phản hồi
        if response and response.text:
            return response.text
        else:
            return "Tôi không thể tạo ra phản hồi lúc này. Vui lòng thử lại sau."
            
    except Exception as e:
        print(f"\033[31mLỗi khi gọi Gemini API: {e}\033[0m")
        return "Xin lỗi, có lỗi xảy ra khi kết nối với dịch vụ AI. Vui lòng thử lại sau."

def initialize_recognizer():
    """Khởi tạo bộ nhận dạng giọng nói và kiểm tra microphone"""
    r = sr.Recognizer()
    
    # Kiểm tra microphone
    try:
        with sr.Microphone() as source:
            print("Kiểm tra microphone... ", end="")
            r.adjust_for_ambient_noise(source, duration=1)
            print("\033[32mOK\033[0m")
    except Exception as e:
        print(f"\033[31mLỗi khi khởi tạo microphone: {e}\033[0m")
        print("Hãy chắc chắn rằng:")
        print("1. Microphone đã được kết nối")
        print("2. Bạn đã cài đặt các thư viện cần thiết (PyAudio)")
        print("3. Bạn đã cấp quyền truy cập microphone cho terminal")
        return None
    
    return r

def check_api_key():
    """Kiểm tra xem API key của Gemini đã được cài đặt chưa"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\033[31mCảnh báo: Không tìm thấy GEMINI_API_KEY trong biến môi trường!\033[0m")
        print("Vui lòng cài đặt API key bằng cách chạy lệnh sau trong terminal:")
        print("export GEMINI_API_KEY=your_api_key_here")
        print("\nBạn có muốn tiếp tục mà không có API key không? (y/n): ", end="")
        choice = input().lower().strip()
        return choice == 'y'
    return True

def main():
    """Hàm chính của chương trình"""
    print("\n" + "="*60)
    print("\033[1m TRỢ LÝ GIỌNG NÓI TÍCH HỢP GEMINI API VÀ SAO CHÉP GIỌNG NÓI \033[0m")
    print("="*60)
    
    # Kiểm tra API key
    if not check_api_key():
        print("Chương trình kết thúc vì thiếu API key.")
        return
    
    # Khởi tạo bộ nhận dạng
    r = initialize_recognizer()
    if not r:
        return
    
    # Khởi tạo TTS engine với khả năng sao chép giọng nói
    voice_engine = CustomVoiceTTS()
    
    # Hỏi người dùng có muốn tạo mẫu giọng nói mới không
    print("\nBạn có muốn tạo mẫu giọng nói mới không? (y/n): ", end="")
    create_new_sample = input().lower().strip() == 'y'
    
    if create_new_sample:
        voice_engine.create_voice_sample()
    else:
        # Hỏi người dùng có muốn sử dụng mẫu giọng nói đã có không
        print("\nBạn có muốn sử dụng mẫu giọng nói đã có không? (y/n): ", end="")
        use_existing_sample = input().lower().strip() == 'y'
        
        if use_existing_sample:
            print("\nNhập đường dẫn đến file âm thanh mẫu: ", end="")
            sample_path = input().strip()
            
            if os.path.exists(sample_path):
                voice_engine.reference_audio_path = sample_path
                print(f"\033[32mĐã nạp mẫu giọng nói từ: {sample_path}\033[0m")
            else:
                print(f"\033[31mKhông tìm thấy file: {sample_path}\033[0m")
                print("Sẽ sử dụng giọng nói mặc định.")
    
    print("\nHướng dẫn sử dụng:")
    print("- Nói vào microphone khi thấy 'Đang nghe...'")
    print("- Nói 'dừng lại' hoặc 'tạm biệt' để thoát chương trình")
    print("- Hỏi bất kỳ câu hỏi nào bằng tiếng Việt để nhận phản hồi từ Gemini API")
    
    voice_engine.speak("Xin chào! Tôi là trợ lý ảo tích hợp với Gemini API. Tôi đang lắng nghe bạn đây.")
    
    # Vòng lặp chính
    while True:
        # Sử dụng microphone làm nguồn âm thanh
        with sr.Microphone() as source:
            print("\n\033[33mĐang nghe...\033[0m")
            # Điều chỉnh theo tiếng ồn xung quanh để nhận dạng tốt hơn
            r.adjust_for_ambient_noise(source, duration=0.5)
            try:
                # Lắng nghe âm thanh từ microphone
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                print("\033[33mĐang xử lý...\033[0m")
                
                # Cố gắng nhận dạng giọng nói bằng Google Speech Recognition (tiếng Việt)
                text = r.recognize_google(audio, language='vi-VN')
                print(f"\033[36mBạn đã nói: {text}\033[0m")
                
                # Kiểm tra lệnh dừng lại
                if "dừng lại" in text.lower() or "tạm biệt" in text.lower():
                    voice_engine.speak("Tạm biệt bạn! Hẹn gặp lại.")
                    break
                
                # Thông báo đang gửi đến API
                print("\033[35mĐang gửi đến Gemini API...\033[0m")
                
                # Tạo phản hồi từ Gemini API
                response = generate_gemini_response(text)
                print(f"\033[32mPhản hồi: {response}\033[0m")
                
                # Phát phản hồi bằng giọng nói đã sao chép
                voice_engine.speak(response)
                    
            except sr.WaitTimeoutError:
                # Bỏ qua lỗi timeout, tiếp tục vòng lặp để nghe lại
                continue
                
            except sr.UnknownValueError:
                print("\033[31mKhông thể nhận dạng được giọng nói.\033[0m")
                voice_engine.speak("Xin lỗi, tôi không hiểu bạn nói gì.")
                
            except sr.RequestError as e:
                print(f"\033[31mLỗi kết nối đến dịch vụ Google Speech Recognition: {e}\033[0m")
                voice_engine.speak("Lỗi kết nối dịch vụ nhận dạng giọng nói.")
                
            except Exception as e:
                print(f"\033[31mĐã xảy ra lỗi không xác định: {e}\033[0m")
                # Tạm dừng một chút nếu có lỗi khác để tránh lặp lỗi quá nhanh
                time.sleep(1)
    
    print("\nChương trình đã kết thúc.")

if __name__ == "__main__":
    main()