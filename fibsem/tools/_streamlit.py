import subprocess
import threading
import time

def run_streamlit_simple(script_path: str, port: int = 8501, streamlit_args: list = None) -> subprocess.Popen:
    """Simple version to run Streamlit app in background thread."""
    
    def start_streamlit():
        cmd = [
            "streamlit", "run", script_path,
            "--server.port", str(port),
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
        ]
    
        # Add any additional arguments for the streamlit script
        if streamlit_args:
            cmd.append("--")  # Separator between streamlit args and script args
            cmd.extend(streamlit_args)
        
        subprocess.run(cmd)
    
    # Start in daemon thread so it stops when main program exits
    thread = threading.Thread(target=start_streamlit, daemon=True)
    thread.start()
    
    # Give it time to start
    time.sleep(2)
    
    return thread

# Usage
def main():

    script_path = "/home/patrick/github/autolamella/autolamella/tools/review.py"
    experiment_path = "/home/patrick/github/autolamella/autolamella/log/AutoLamella-2025-05-30-17-56"

    # Start Streamlit app
    streamlit_args = ["--experiment_path", experiment_path]
    thread = run_streamlit_simple(script_path=script_path,
                                  port=8502, 
                                  streamlit_args=streamlit_args)
    
    # Your main application continues here
    print("Streamlit is running in background...")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()