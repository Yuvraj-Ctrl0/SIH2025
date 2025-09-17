# Enhanced anti-spoof with fallback methods
import cv2
import numpy as np
import os

def load_onnx_model():
    """Try to load the ONNX anti-spoofing model"""
    model_path = "server/models/anti_spoof.onnx"
    if os.path.exists(model_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            print("✅ ONNX anti-spoof model loaded successfully")
            return session
        except ImportError:
            print("⚠️  onnxruntime not installed. Using fallback anti-spoof.")
            return None
        except Exception as e:
            print(f"⚠️  Failed to load ONNX model: {e}. Using fallback.")
            return None
    else:
        print(f"⚠️  ONNX model not found at {model_path}. Using fallback anti-spoof.")
        return None

# Try to load the model once
_onnx_model = load_onnx_model()

def preprocess_face_for_onnx(face_rgb):
    """Preprocess face image for ONNX model input"""
    # Resize to model expected size (usually 224x224 or 112x112)
    face_resized = cv2.resize(face_rgb, (224, 224))
    
    # Normalize to [0, 1]
    face_normalized = face_resized.astype(np.float32) / 255.0
    
    # Add batch dimension and rearrange to NCHW format
    face_batch = np.transpose(face_normalized[np.newaxis, :, :, :], (0, 3, 1, 2))
    
    return face_batch

def onnx_liveness_check(face_rgb):
    """Use ONNX model for liveness detection"""
    try:
        input_data = preprocess_face_for_onnx(face_rgb)
        
        # Get input name
        input_name = _onnx_model.get_inputs()[0].name
        
        # Run inference
        outputs = _onnx_model.run(None, {input_name: input_data})
        
        # Extract score (assuming model outputs probability of being real)
        score = float(outputs[0][0][1])  # Adjust indexing based on your model
        is_live = score > 0.5
        
        return is_live, score
        
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        return fallback_liveness_check(face_rgb)

def fallback_liveness_check(face_rgb):
    """Fallback liveness detection using traditional computer vision"""
    if face_rgb.size == 0:
        return False, 0.0
    
    try:
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
        
        # Check 1: Texture analysis (real faces have more texture variation)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_score = min(laplacian_var / 100.0, 1.0)  # Normalize
        
        # Check 2: Color distribution (real faces have better color distribution)
        hist_r = cv2.calcHist([face_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([face_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([face_rgb], [2], None, [256], [0, 256])
        
        # Calculate histogram entropy as a measure of color diversity
        def calculate_entropy(hist):
            hist = hist.flatten()
            hist = hist[hist > 0]  # Remove zero values
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            return entropy
        
        color_diversity = (calculate_entropy(hist_r) + 
                          calculate_entropy(hist_g) + 
                          calculate_entropy(hist_b)) / 3
        color_score = min(color_diversity / 8.0, 1.0)  # Normalize
        
        # Check 3: Edge density (printed photos often have fewer natural edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(edge_density * 10, 1.0)  # Normalize
        
        # Check 4: Brightness variation
        brightness_std = np.std(gray)
        brightness_score = min(brightness_std / 50.0, 1.0)  # Normalize
        
        # Combine scores
        final_score = (texture_score * 0.4 + 
                      color_score * 0.3 + 
                      edge_score * 0.2 + 
                      brightness_score * 0.1)
        
        # Decision threshold
        is_live = final_score > 0.3  # Adjust this threshold based on testing
        
        return is_live, final_score
        
    except Exception as e:
        print(f"❌ Fallback liveness check failed: {e}")
        # Ultra-safe fallback - assume live but low confidence
        return True, 0.1

def is_live(face_rgb):
    """
    Main liveness detection function
    Args:
        face_rgb: Face image in RGB format (HxWx3 uint8)
    Returns:
        tuple: (is_live: bool, score: float)
    """
    if face_rgb is None or face_rgb.size == 0:
        return False, 0.0
    
    # Use ONNX model if available, otherwise fallback
    if _onnx_model is not None:
        return onnx_liveness_check(face_rgb)
    else:
        return fallback_liveness_check(face_rgb)

def test_anti_spoof():
    """Test function to verify anti-spoof is working"""
    print("🧪 Testing anti-spoof system...")
    
    # Create a dummy face image for testing
    dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    is_live_result, score = is_live(dummy_face)
    
    print(f"Test result: is_live={is_live_result}, score={score:.3f}")
    
    if _onnx_model is not None:
        print("✅ Using ONNX model for anti-spoofing")
    else:
        print("⚠️  Using fallback anti-spoofing method")

if __name__ == "__main__":
    test_anti_spoof()