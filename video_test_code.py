import multiprocessing
import subprocess


def run_script(script_name):
    """
    Runs a Python script using subprocess.

    Args:
        script_name (str): The name of the script to be executed.

    Returns:
        None
    """
    subprocess.call(['python', script_name])


if __name__ == '__main__':
    # Define the names of the two scripts to be executed
    script1 = 'capture_images_on_pi.py'
    script2 = 'backend_processing.py'

    # Create two separate processes
    process1 = multiprocessing.Process(target=run_script, args=(script1,))
    process2 = multiprocessing.Process(target=run_script, args=(script2,))

    # The start() method initializes the process and calls the target function
    process1.start()
    process2.start()

    # Wait for both processes to finish
    process1.join()
    process2.join()
