#!/usr/bin/env python3
"""
Professional Camera Test Script
Tests webcam functionality and performance
"""

import cv2
import time
import sys

def test_camera():
    """Test camera functionality with professional setup"""
    print("🎥 Professional Camera Test Starting...")
    print("=" * 50)
    
    # Try different camera backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Media Foundation (Windows)"),
        (cv2.CAP_V4L2, "Video4Linux2 (Linux)"),
        (0, "Default Backend")
    ]
    
    cap = None
    working_backend = None
    
    for backend, name in backends:
        print(f"🔍 Trying {name}...")
        try:
            if backend == 0:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(0, backend)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ {name} works!")
                    working_backend = name
                    break
                else:
                    cap.release()
            else:
                if cap:
                    cap.release()
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            if cap:
                cap.release()
    
    if not cap or not cap.isOpened():
        print("❌ No working camera found!")
        return False
    
    print(f"\n🎯 Using: {working_backend}")
    
    # Optimize camera settings
    print("\n⚙️  Optimizing camera settings...")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
    
    # Get actual settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📐 Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps}")
    
    # Warm up camera
    print("\n🔥 Warming up camera...")
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame {i+1}")
            cap.release()
            return False
        print(f"  Frame {i+1}/10 ✓")
    
    print("\n🚀 Starting live test...")
    print("Controls:")
    print("  'q' or ESC - Quit")
    print("  's' - Save screenshot")
    print("  'f' - Show FPS info")
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Add info overlay
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Calculate FPS every second
            fps_counter += 1
            if current_time - last_fps_time >= 1.0:
                current_fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            
            # Draw info
            cv2.putText(frame, f"Camera Test - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Camera Test', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):
                filename = f"camera_test_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('f'):
                print(f"📊 Current FPS: {current_fps:.2f}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Final stats
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\n📊 Test Results:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Backend used: {working_backend}")
    
    if avg_fps > 15:
        print("✅ Camera performance: GOOD")
        return True
    elif avg_fps > 10:
        print("⚠️  Camera performance: ACCEPTABLE")
        return True
    else:
        print("❌ Camera performance: POOR")
        return False

if __name__ == "__main__":
    print("🎥 Professional Camera Test")
    print("This will test your webcam functionality and performance")
    print("Make sure no other applications are using the camera\n")
    
    success = test_camera()
    
    if success:
        print("\n✅ Camera test completed successfully!")
        print("Your camera should work fine with the attendance system.")
    else:
        print("\n❌ Camera test failed!")
        print("Please check:")
        print("  1. Camera is connected properly")
        print("  2. No other apps are using the camera")
        print("  3. Camera drivers are installed")
        print("  4. Try running as administrator")
    
    input("\nPress Enter to exit...")