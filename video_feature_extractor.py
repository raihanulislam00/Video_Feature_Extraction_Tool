import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def download_video(url, output_path="downloaded_video.mp4"):
    import urllib.request
    print("Downloading video from URL...")
    print("URL: %s" % url)
    try:
        urllib.request.urlretrieve(url, output_path)
        print(" Video downloaded: %s" % output_path)
        return output_path
    except Exception as e:
        print(" Error downloading video: %s" % str(e))
        return None

class VideoFeatureExtractor:
    def __init__(self, video_path, sample_rate=30):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.cap = cv2.VideoCapture(video_path)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        self.features = {}
        self.features['video_info'] = {
            'path': video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration_seconds': self.duration,
            'resolution': str(self.w) + 'x' + str(self.h)
        }
        
        print("Video loaded:", os.path.basename(video_path))
        print("Duration: %.2fs | FPS: %.2f | Frames: %d" % (self.duration, self.fps, self.total_frames))
        print("Resolution: %dx%d" % (self.w, self.h))
    
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def detect_shot_cuts(self, threshold=30.0):
        print("\n[1/4] Detecting shot cuts...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cuts = []
        diffs = []
        prev = None
        fc = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev is not None:
                h1 = cv2.calcHist([prev], [0], None, [256], [0, 256])
                h2 = cv2.calcHist([gray], [0], None, [256], [0, 256])
                
                h1 = cv2.normalize(h1, h1).flatten()
                h2 = cv2.normalize(h2, h2).flatten()
                
                d = np.sum(np.abs(h1 - h2))
                diffs.append(d)
                
                if d > threshold:
                    ts = fc / self.fps
                    cuts.append({
                        'frame': fc,
                        'timestamp': ts,
                        'difference': float(d)
                    })
            
            prev = gray
            fc += 1
        
        avg_diff = float(np.mean(diffs)) if len(diffs) > 0 else 0
        max_diff = float(np.max(diffs)) if len(diffs) > 0 else 0
        
        result = {
            'total_cuts': len(cuts),
            'cut_timestamps': cuts,
            'average_frame_difference': avg_diff,
            'max_frame_difference': max_diff,
            'threshold_used': threshold
        }
        
        print(" Detected %d hard cuts" % len(cuts))
        return result
    
    
    def analyze_motion(self):
        print("\n[2/4] Analyzing motion using optical flow...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        prev_g = None
        motion_vals = []
        fc = 0
        sf = 0
        
        params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if fc % self.sample_rate != 0:
                fc = fc + 1
                continue
            
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_g is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_g, g, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                avg_mag = np.mean(mag)
                motion_vals.append(avg_mag)
                sf = sf + 1
            
            prev_g = g
            fc = fc + 1
        
        avg_m = float(np.mean(motion_vals)) if len(motion_vals) > 0 else 0
        
        result = {
            'average_motion': avg_m,
            'max_motion': float(np.max(motion_vals)) if len(motion_vals) > 0 else 0,
            'min_motion': float(np.min(motion_vals)) if len(motion_vals) > 0 else 0,
            'std_motion': float(np.std(motion_vals)) if len(motion_vals) > 0 else 0,
            'frames_analyzed': sf,
            'motion_category': self.get_motion_category(avg_m)
        }
        
        print("✓ Average motion: %.2f (%s)" % (result['average_motion'], result['motion_category']))
        return result
    
    def get_motion_category(self, avg_motion):
        if avg_motion < 1.0:
            return "Static/Very Low"
        elif avg_motion < 3.0:
            return "Low"
        elif avg_motion < 7.0:
            return "Moderate"
        elif avg_motion < 12.0:
            return "High"
        else:
            return "Very High"
    
    
    def detect_text(self, sample_frames=20):
        print("\n[3/4] Detecting text using OCR...")
        
        try:
            import pytesseract
        except ImportError:
            print("⚠ pytesseract not available. Skipping text detection.")
            return {
                'error': 'pytesseract not installed',
                'text_present_ratio': 0,
                'frames_with_text': 0,
                'frames_analyzed': 0
            }
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        txt_frames = 0
        all_txt = []
        interval = max(1, self.total_frames // sample_frames)
        analyzed = 0
        
        for i in range(sample_frames):
            pos = i * interval
            if pos >= self.total_frames:
                break
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            txt = pytesseract.image_to_string(th, config='--psm 11').strip()
            
            if txt and len(txt) > 3:
                txt_frames += 1
                all_txt.append(txt)
            
            analyzed += 1
        
        words = []
        for t in all_txt:
            words.extend([w.lower() for w in t.split() if len(w) > 3])
        
        wc = Counter(words)
        kw = [w for w, c in wc.most_common(10) if c > 1]
        
        ratio = txt_frames / analyzed if analyzed > 0 else 0
        result = {
            'text_present_ratio': ratio,
            'frames_with_text': txt_frames,
            'frames_analyzed': analyzed,
            'keywords': kw,
            'total_text_instances': len(all_txt)
        }
        
        print("✓ Text detected in %d/%d frames (%.1f%%)" % (txt_frames, analyzed, ratio*100))
        return result
    
    
    def detect_objects_and_people(self, confidence=0.5, sample_frames=30):
        print("\n[4/4] Detecting objects and people using YOLO...")
        
        try:
            yolo_weights = "yolov3.weights"
            yolo_cfg = "yolov3.cfg"
            
            if not (os.path.exists(yolo_weights) and os.path.exists(yolo_cfg)):
                print("⚠ YOLO model files not found. Using alternative method...")
                return self.detect_people_cascade(sample_frames)
            
            net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
            ln = net.getLayerNames()
            out_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
            
            class_names = []
            with open("coco.names", "r") as f:
                class_names = [line.strip() for line in f.readlines()]
            
        except Exception as e:
            print("⚠ Could not load YOLO: %s. Using alternative method..." % str(e))
            return self.detect_people_cascade(sample_frames)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        people_cnt = 0
        obj_cnt = 0
        detections = []
        interval = max(1, self.total_frames // sample_frames)
        analyzed = 0
        
        for i in range(sample_frames):
            pos = i * interval
            if pos >= self.total_frames:
                break
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(out_layers)
            
            fp = 0
            fo = 0
            
            for out in outs:
                for det in out:
                    sc = det[5:]
                    cid = np.argmax(sc)
                    conf = sc[cid]
                    
                    if conf > confidence:
                        if class_names[cid] == "person":
                            fp = fp + 1
                            people_cnt = people_cnt + 1
                        else:
                            fo = fo + 1
                            obj_cnt = obj_cnt + 1
            
            detections.append({'people': fp, 'objects': fo})
            analyzed = analyzed + 1
        
        total = people_cnt + obj_cnt
        p_ratio = people_cnt / total if total > 0 else 0
        
        dom = 'people' if p_ratio > 0.5 else ('objects' if p_ratio < 0.5 else 'balanced')
        
        result = {
            'total_people_detected': people_cnt,
            'total_objects_detected': obj_cnt,
            'person_to_object_ratio': p_ratio,
            'object_to_person_ratio': 1 - p_ratio,
            'average_people_per_frame': people_cnt / analyzed if analyzed > 0 else 0,
            'average_objects_per_frame': obj_cnt / analyzed if analyzed > 0 else 0,
            'frames_analyzed': analyzed,
            'dominance': dom
        }
        
        print("✓ People: %d | Objects: %d | Dominance: %s" % (people_cnt, obj_cnt, dom))
        return result
    
    def detect_people_cascade(self, sample_frames=30):
        print("Using Haar Cascade for person detection...")
        
        try:
            face_casc = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            body_casc = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            )
        except:
            print("⚠ Could not load cascade classifiers")
            return {
                'error': 'No detection method available',
                'total_people_detected': 0,
                'frames_analyzed': 0
            }
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        ppl_detected = 0
        interval = max(1, self.total_frames // sample_frames)
        analyzed = 0
        
        for i in range(sample_frames):
            pos = i * interval
            if pos >= self.total_frames:
                break
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_casc.detectMultiScale(g, 1.1, 4)
            bodies = body_casc.detectMultiScale(g, 1.1, 4)
            
            ppl_detected = ppl_detected + max(len(faces), len(bodies))
            analyzed = analyzed + 1
        
        dom = 'people' if ppl_detected > 0 else 'unknown'
        result = {
            'total_people_detected': ppl_detected,
            'total_objects_detected': 0,
            'person_to_object_ratio': 1.0 if ppl_detected > 0 else 0,
            'average_people_per_frame': ppl_detected / analyzed if analyzed > 0 else 0,
            'frames_analyzed': analyzed,
            'detection_method': 'Haar Cascade',
            'dominance': dom
        }
        
        print("✓ People detected: %d in %d frames" % (ppl_detected, analyzed))
        return result
    
    
    def extract_all_features(self, shot_cut_threshold=30.0, motion_sample_rate=None,
                           text_sample_frames=20, object_sample_frames=30,
                           object_confidence=0.5):
        print("\n" + "="*60)
        print("VIDEO FEATURE EXTRACTION")
        print("="*60)
        
        self.features['shot_cuts'] = self.detect_shot_cuts(shot_cut_threshold)
        self.features['motion_analysis'] = self.analyze_motion()
        self.features['text_detection'] = self.detect_text(text_sample_frames)
        self.features['object_detection'] = self.detect_objects_and_people(
            object_confidence, object_sample_frames
        )
        
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60 + "\n")
        
        return self.features
    
    def save_features(self, output_path=None):
        if output_path is None:
            base = os.path.splitext(os.path.basename(self.video_path))[0]
            output_path = base + "_features.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.features, f, indent=2)
        
        print("Features saved to: %s" % output_path)
        return output_path
    
    
    def print_summary(self):
        if not self.features:
            print("No features extracted yet. Run extract_all_features() first.")
            return
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60 + "\n")
        
        info = self.features['video_info']
        print("Video: %s" % os.path.basename(info['path']))
        print("Duration: %.2fs" % info['duration_seconds'])
        print("Resolution: %s @ %.2f FPS\n" % (info['resolution'], info['fps']))
        
        if 'shot_cuts' in self.features:
            cuts = self.features['shot_cuts']
            print("Shot Cuts: %d" % cuts['total_cuts'])
            print("  - Average frame difference: %.2f" % cuts['average_frame_difference'])
            print("  - Max frame difference: %.2f\n" % cuts['max_frame_difference'])
        
        if 'motion_analysis' in self.features:
            motion = self.features['motion_analysis']
            print("Motion Analysis: %s" % motion['motion_category'])
            print("  - Average motion: %.2f" % motion['average_motion'])
            print("  - Max motion: %.2f\n" % motion['max_motion'])
        
        if 'text_detection' in self.features:
            text = self.features['text_detection']
            if 'error' not in text:
                print("Text Detection: %.1f%% of frames" % (text['text_present_ratio']*100))
                print("  - Frames with text: %d/%d" % (text['frames_with_text'], text['frames_analyzed']))
                if text['keywords']:
                    kw_str = ', '.join(text['keywords'][:5])
                    print("  - Keywords: %s\n" % kw_str)
                else:
                    print("  - No keywords detected\n")
        
        if 'object_detection' in self.features:
            obj = self.features['object_detection']
            if 'error' not in obj:
                print("Object Detection: %s dominant" % obj['dominance'].capitalize())
                print("  - People detected: %d" % obj['total_people_detected'])
                if 'total_objects_detected' in obj:
                    print("  - Objects detected: %d" % obj['total_objects_detected'])
                print("  - Avg people/frame: %.2f\n" % obj['average_people_per_frame'])
        
        print("="*60 + "\n")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_feature_extractor.py <video_path_or_url>")
        print("\nExamples:")
        print("  python video_feature_extractor.py sample_video.mp4")
        print("  python video_feature_extractor.py https://example.com/video.mp4")
        sys.exit(1)
    
    vpath = sys.argv[1]
    
    if vpath.startswith('http://') or vpath.startswith('https://'):
        print("Detected URL input...")
        vpath = download_video(vpath, "downloaded_video.mp4")
        if vpath is None:
            print("Failed to download video")
            sys.exit(1)
    
    try:
        ext = VideoFeatureExtractor(vpath, sample_rate=30)
        
        feats = ext.extract_all_features()
        
        ext.print_summary()
        
        out_file = ext.save_features()
        
        print("\n All features extracted successfully!")
        print(" Results saved to: %s" % out_file)
        
    except Exception as e:
        print("\n✗ Error: %s" % str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
